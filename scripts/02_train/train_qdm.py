#!/usr/bin/env python
"""
Fit the K density-matrix behavioral profiles by maximum likelihood.

Upstream:
  scripts/00_preprocess/merge_chunks.py -> data/processed/full_qdm.parquet
Downstream:
  scripts/03_infer/infer.py, scripts/04_analysis/*

Objective, per frame:

  L = -log p(x_t)  -  gamma * sum_k S(rho_k)

where p(x_t) = phi_t^T rho_t phi_t is the Born-rule likelihood and
S(rho_k) = -tr(rho_k log rho_k) is the von Neumann entropy of profile k.

gamma is the weight on the entropy term. It is subtracted, so a larger gamma
pushes the profiles to spread their spectral mass across several eigenvectors.
Without it each rho_k collapses to rank one and a profile can only represent a
single behavioral mode. gamma is a penalty weight and has nothing to do with the
eigenvalues lambda_i of rho_k.

Only V_k and beta are fit. alpha and eta are fixed hyperparameters; see
src/qdm/model.py for why no gradient can reach eta.

Model selection uses the unregularized NLL, not the regularized objective. A
large gamma would otherwise select whichever epoch maximized entropy.

Writes to --model-dir:
  best_model.pt        state_dict at the lowest unregularized NLL
  checkpoint.pt        resumable training state
  rho_profile_{k}.npy  the K profiles
  model_params.npz     beta, alpha, eta, V_k
  rff_sampler.pkl      the fitted RFF map
  scaler_x.pkl         behavioral scaler
  scaler_c.pkl         context scaler
  training_log.csv     per-epoch NLL, profile entropy, per-profile effective rank

The objective is a per-chunk quantity: the NLL is a sum over the chunk's
observations, while the entropy is one scalar for the whole chunk, since it is a
property of the profiles and not of the data. Dividing an accumulated objective
by the observation count therefore shrinks the entropy term by a factor of
chunk_size and makes it look as though gamma did nothing. The gradient is
unaffected, but the printed number is misleading, so no per-observation
"objective" is reported. The entropy is logged on its own instead, measured on
the profiles as they stand at the end of each epoch, which is the model that
gets saved.

The whole parquet is loaded into memory, then subsampled to --n-egos whole
trajectories. On a machine that cannot hold the file, use --n-egos to cut the
training set; the load itself still needs the RAM.

Usage:
  python train_qdm.py --parquet data/processed/full_qdm.parquet
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qdm.config import (
    ALPHA,
    BEHAVIORAL_COLS,
    CHUNK_SIZE,
    CONTEXT_COLS,
    D,
    EPOCHS,
    ETA,
    GAMMA,
    K,
    LR,
    MODEL_DIR,
    N_TRAIN_EGOS,
    PARQUET_NAME,
    PROCESSED_DIR,
    RANK,
    SEED,
)
from qdm.features import fit_rff, fit_scalers, phi, save_features
from qdm.io import behavioral_context_arrays, load_parquet, sample_egos
from qdm.model import QuantumDriverModel, enforce_density_matrix, von_neumann_entropy
from qdm.signals import log


def train(Phi, C, ids, args, device):
    Phi_t = torch.tensor(Phi, dtype=torch.float32, device=device)
    C_t = torch.tensor(C, dtype=torch.float32, device=device)
    n_rows = len(Phi)

    torch.manual_seed(args.seed)
    model = QuantumDriverModel(
        K=args.K, D=args.D, q=C.shape[1], rank=args.rank,
        alpha=args.alpha, eta=args.eta,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_dir / "checkpoint.pt"

    start_epoch, best_nll = 0, float("inf")
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_nll = ckpt["best_nll"]
        log(f"[resume] from epoch {start_epoch + 1}, best NLL {best_nll:.6f}")

    log("=" * 70)
    log(f"TRAIN  K={args.K}  D={args.D}  rank={args.rank}  gamma={args.gamma}")
    log(f"       alpha={args.alpha} (fixed)   eta={args.eta} (fixed)")
    log(f"       {n_rows:,} rows  seed={args.seed}  lr={args.lr}")
    log("=" * 70)

    history = []
    n_chunks = (n_rows + args.chunk_size - 1) // args.chunk_size

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        log(f"\n[epoch {epoch + 1}/{args.epochs}]")

        # Reset the per-driver states each epoch: every ego starts from the
        # maximally mixed state, as it does at inference.
        driver_states = {}
        sum_nll = sum_obj = 0.0

        for c in range(n_chunks):
            lo = c * args.chunk_size
            hi = min(lo + args.chunk_size, n_rows)

            optimizer.zero_grad()
            nll, driver_states = model.forward_chunk(
                Phi_t[lo:hi], C_t[lo:hi], ids[lo:hi], driver_states
            )

            entropy = von_neumann_entropy(model.build_profiles())
            objective = nll - args.gamma * entropy
            objective.backward()
            optimizer.step()

            sum_nll += nll.item()
            sum_obj += objective.item()

            if (c + 1) % 50 == 0 or (c + 1) == n_chunks:
                log(f"  chunk {c + 1:>5d}/{n_chunks}  "
                    f"NLL/obs {sum_nll / hi:.6f}")

        nll_per_obs = sum_nll / n_rows

        # Entropy of the profiles AS THEY STAND at the end of the epoch, which
        # is the model that gets saved. An average over the epoch would describe
        # no particular model: the profiles change under every chunk.
        with torch.no_grad():
            end_entropy = von_neumann_entropy(model.build_profiles()).item()
            eigvals = [
                np.sort(np.linalg.eigvalsh(
                    model.build_profiles()[k].cpu().numpy()))[::-1]
                for k in range(args.K)
            ]
        eff_ranks = [float(np.exp(-(e[e > 1e-12] * np.log(e[e > 1e-12])).sum()))
                     for e in eigvals]

        # The objective is a per-chunk quantity: nll is a sum over the chunk's
        # observations, entropy is one scalar for the whole chunk. Dividing the
        # accumulated objective by n_rows therefore shrinks the entropy term by
        # a factor of chunk_size and makes it look negligible. Report the
        # entropy separately instead of folding it into a per-observation number.
        log(f"[epoch {epoch + 1}] NLL/obs {nll_per_obs:.6f}   "
            f"profile entropy {end_entropy:.4f} nats   "
            f"effective ranks {[round(r, 2) for r in eff_ranks]}   "
            f"{(time.time() - t0) / 60:.1f} min")

        history.append({
            "epoch": epoch + 1,
            "nll_per_obs": nll_per_obs,
            "profile_entropy": end_entropy,
            **{f"eff_rank_P{k + 1}": eff_ranks[k] for k in range(args.K)},
        })

        # Select on the unregularized NLL, i.e. fit quality.
        if nll_per_obs < best_nll:
            best_nll = nll_per_obs
            torch.save(model.state_dict(), model_dir / "best_model.pt")
            log("[epoch] new best NLL -> best_model.pt")

        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "best_nll": best_nll},
                   ckpt_path)

    pd.DataFrame(history).to_csv(model_dir / "training_log.csv", index=False)

    best_path = model_dir / "best_model.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device,
                                         weights_only=False))
        log(f"\n[load] best model, NLL/obs {best_nll:.6f}")

    return model


def save_model(model, model_dir):
    model_dir = Path(model_dir)
    Vs = [V.detach().cpu().numpy() for V in model.Vs]

    rho_profiles = []
    for V in Vs:
        M = V @ V.T
        tr = np.trace(M)
        rho = M / tr if tr > 1e-12 else np.eye(model.D) / model.D
        rho_profiles.append(enforce_density_matrix(rho))

    for k, rho in enumerate(rho_profiles):
        np.save(model_dir / f"rho_profile_{k + 1}.npy", rho)

    np.savez(model_dir / "model_params.npz",
             beta=model.beta.detach().cpu().numpy(),
             alpha=model.get_alpha(),
             eta=model.get_eta(),
             **{f"V_{k}": V for k, V in enumerate(Vs)})

    log(f"[save] {len(rho_profiles)} profiles -> {model_dir}")
    return rho_profiles


def report_spectra(rho_profiles):
    """Eigenvalue spectrum of each profile. A profile that occupies several
    eigendirections is what gamma is there to produce."""
    log("\n" + "=" * 70)
    log("PROFILE SPECTRA")
    log("=" * 70)
    for k, rho in enumerate(rho_profiles):
        eigvals = np.sort(np.linalg.eigvalsh(rho))[::-1]
        log(f"\n  Profile {k + 1}")
        log(f"    top 10 eigenvalues: "
            f"{np.array2string(eigvals[:10], precision=4, suppress_small=True)}")
        log(f"    mass in top 1 / 5 / 10: "
            f"{eigvals[0]:.4f} / {eigvals[:5].sum():.4f} / {eigvals[:10].sum():.4f}")
        log(f"    purity tr(rho^2):   {float((rho * rho).sum()):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(PROCESSED_DIR / PARQUET_NAME))
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--K", type=int, default=K)
    ap.add_argument("--D", type=int, default=D)
    ap.add_argument("--rank", type=int, default=RANK)
    ap.add_argument("--gamma", type=float, default=GAMMA,
                    help="Weight on the von Neumann entropy penalty. Larger "
                         "gamma spreads each profile over more eigenmodes.")
    ap.add_argument("--alpha", type=float, default=ALPHA,
                    help="State-evolution blend. Held fixed; a learnable alpha "
                         "collapses toward zero.")
    ap.add_argument("--eta", type=float, default=ETA,
                    help="Behavioral adaptation. Held fixed: no gradient "
                         "reaches eta, see src/qdm/model.py.")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--n-egos", type=int, default=N_TRAIN_EGOS,
                    help="Whole trajectories to train on.")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="Force CPU.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu
                          else "cpu")
    log(f"[env] torch {torch.__version__}, device {device}")

    df = load_parquet(args.parquet)
    X_raw, C_raw, ids, times = behavioral_context_arrays(df)
    del df

    # Scalers and the RFF map are fit on the FULL file, before subsampling, so
    # the feature space does not depend on which egos happen to be drawn.
    scaler_x, scaler_c = fit_scalers(X_raw, C_raw)
    X_std = scaler_x.transform(X_raw)
    C_std = scaler_c.transform(C_raw)
    log(f"[scale] {X_std.shape[1]} behavioral {BEHAVIORAL_COLS}")
    log(f"[scale] {C_std.shape[1]} context    {CONTEXT_COLS}")

    rff = fit_rff(X_std, n_components=args.D)
    Phi = phi(X_std, rff)
    log(f"[rff] D={Phi.shape[1]}, L2-normalized")

    ids, times, (Phi, C_std) = sample_egos(
        ids, times, args.n_egos, args.seed, arrays=(Phi, C_std)
    )

    save_features(args.model_dir, rff, scaler_x, scaler_c)
    log(f"[save] rff_sampler.pkl, scaler_x.pkl, scaler_c.pkl -> {args.model_dir}")

    model = train(Phi, C_std, ids, args, device)
    rho_profiles = save_model(model, args.model_dir)
    report_spectra(rho_profiles)

    log("\n[done]")


if __name__ == "__main__":
    main()
