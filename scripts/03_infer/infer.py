#!/usr/bin/env python
"""
Run the trained model forward over a set of egos and save the per-frame state.

Upstream:
  scripts/00_preprocess/merge_chunks.py -> data/processed/full_qdm.parquet
  scripts/02_train/train_qdm.py         -> data/models/best_model.pt
                                           data/models/rff_sampler.pkl
                                           data/models/scaler_x.pkl
                                           data/models/scaler_c.pkl
Downstream:
  scripts/04_analysis/*, scripts/05_macroscopic/*

The RFF map and both scalers are loaded from the training run, never refit, so
inference sees the exact feature space the model was fit in.

Per ego, per frame:
  pi        (T, K)  profile mixture pi_k(c_t)
  p         (T,)    Born-rule likelihood phi_t^T rho_t phi_t
  modes     (T, M)  |<v_m, phi_t>|^2 against the top M eigenvectors of the
                    profile named by --mode-profile
  speed, headway, jerk, context, time

Writes:
  catalog.csv           one row per ego, summary statistics for ranking
  by_ego/{tid}_*.npy    the arrays above
  profile_eigvecs.npy   (K, D, M)
  profile_eigvals.npy   (K, M)

Usage:
  python infer.py --parquet data/processed/full_qdm.parquet
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qdm.config import (
    ALPHA,
    BEHAVIORAL_COLS,
    CONTEXT_COLS,
    D,
    ETA,
    INFERENCE_DIR,
    K,
    MODEL_DIR,
    PARQUET_NAME,
    PROCESSED_DIR,
    RANK,
    SEED,
)
from qdm.features import load_features, phi
from qdm.io import load_parquet
from qdm.model import QuantumDriverModel, top_eigenmodes
from qdm.signals import log


@torch.no_grad()
def infer_ego(model, sub, rff, scaler_x, scaler_c, eigvecs_focus, device):
    """Forward pass over one ego's frames, in time order."""
    b_raw = sub[BEHAVIORAL_COLS].to_numpy()
    c_raw = sub[CONTEXT_COLS].to_numpy()
    T = len(sub)

    Phi = torch.tensor(phi(scaler_x.transform(b_raw), rff),
                       dtype=torch.float32, device=device)
    C = torch.tensor(scaler_c.transform(c_raw),
                     dtype=torch.float32, device=device)

    rho_k = model.build_profiles()
    alpha = model.get_alpha()
    eta = model.get_eta()

    pi_traj = np.zeros((T, model.K), dtype=np.float32)
    p_traj = np.zeros(T, dtype=np.float32)
    modes_traj = np.zeros((T, eigvecs_focus.shape[1]), dtype=np.float32)

    # Start from the maximally mixed state, as in training.
    rho_prev = torch.eye(model.D, device=device) / model.D

    for t in range(T):
        phi_t = Phi[t]
        pi = model.softmax_activation(C[t])
        mixture = torch.einsum("k,kde->de", pi, rho_k)
        rho_t = (1 - alpha) * rho_prev + alpha * mixture

        p = torch.clamp(phi_t @ rho_t @ phi_t, min=1e-12)
        proj = eigvecs_focus.T @ phi_t

        pi_traj[t] = pi.cpu().numpy()
        p_traj[t] = p.item()
        modes_traj[t] = (proj ** 2).cpu().numpy()

        rho_prev = (1 - eta) * rho_t + eta * torch.outer(phi_t, phi_t)

    return {
        "pi": pi_traj,
        "p": p_traj,
        "modes": modes_traj,
        "speed": b_raw[:, 0],
        "headway": b_raw[:, 1],
        "jerk": b_raw[:, 2],
        "context": c_raw,
        "time": sub["time"].to_numpy(),
    }


def summarize(tid, res, eigvals_focus):
    """Statistics that surface egos worth plotting."""
    pi = res["pi"]
    mean_pi = pi.mean(axis=0)

    # Mean per-profile variance of pi over time: how much the mixture actually
    # moves. The entropy of the MEAN mixture would instead reward egos that sit
    # at a near-uniform mixture at every single frame.
    pi_var = float(pi.var(axis=0).mean())
    arg = pi.argmax(axis=1)
    switches = int((arg[1:] != arg[:-1]).sum())

    # Mode responsibility r_m(t) = lambda_m |<v_m, phi_t>|^2 / sum_j (...).
    # The raw argmax of |<v_m, phi_t>|^2 is biased toward eigenvectors aligned
    # with high-variance directions of the data regardless of how much spectral
    # mass lambda_m they carry. Weighting by lambda_m gives the probability that
    # mode m generated phi_t.
    weighted = res["modes"] * eigvals_focus[None, :]
    resp = weighted / weighted.sum(axis=1, keepdims=True).clip(min=1e-12)
    mode2_resp = float(resp[:, 1].mean()) if resp.shape[1] > 1 else np.nan
    m_arg = resp[:, :2].argmax(axis=1) if resp.shape[1] > 1 else np.zeros(len(resp), int)
    mode_flips = int((m_arg[1:] != m_arg[:-1]).sum())

    row = {
        "trajectory_id": tid,
        "length": len(pi),
        "nll_mean": float(-np.log(res["p"].clip(min=1e-9)).mean()),
        "pi_var_mean": pi_var,
        "switches": switches,
        "mode2_resp": mode2_resp,
        "mode_flips": mode_flips,
        "mean_speed": float(res["speed"].mean()),
        "mean_headway": float(res["headway"].mean()),
        "mean_jerk": float(res["jerk"].mean()),
    }
    for k in range(pi.shape[1]):
        row[f"dwell_P{k + 1}"] = float(mean_pi[k])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(PROCESSED_DIR / PARQUET_NAME))
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--out", default=str(INFERENCE_DIR))
    ap.add_argument("--K", type=int, default=K)
    ap.add_argument("--D", type=int, default=D)
    ap.add_argument("--rank", type=int, default=RANK)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--eta", type=float, default=ETA)
    ap.add_argument("--n-egos", type=int, default=50_000,
                    help="Egos to run inference on. Any value; the downstream "
                         "analyses simply use whatever is in the catalog.")
    ap.add_argument("--min-len", type=int, default=200,
                    help="Skip egos shorter than this many frames.")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-modes", type=int, default=5,
                    help="Eigenmodes to project each frame onto.")
    ap.add_argument("--mode-profile", type=int, default=2,
                    help="1-indexed profile whose eigenmodes the per-frame "
                         "projections are taken against. Default 2: the profile "
                         "whose spectrum showed more than one significant "
                         "eigenvalue in the trained model.")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if not 1 <= args.mode_profile <= args.K:
        ap.error(f"--mode-profile must be in [1, {args.K}]")

    out = Path(args.out)
    (out / "by_ego").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu
                          else "cpu")
    log(f"[env] device {device}")

    rff, scaler_x, scaler_c = load_features(args.model_dir)
    log(f"[load] rff and scalers from {args.model_dir}")

    model = QuantumDriverModel(
        K=args.K, D=args.D, q=len(CONTEXT_COLS), rank=args.rank,
        alpha=args.alpha, eta=args.eta,
    ).to(device)
    ckpt = Path(args.model_dir) / "best_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found. Train first.")
    model.load_state_dict(torch.load(ckpt, map_location=device,
                                     weights_only=False))
    model.eval()
    log(f"[load] {ckpt}")
    log(f"[model] K={args.K} D={args.D} rank={args.rank} "
        f"alpha={model.get_alpha():.4f} eta={model.get_eta():.4f}")

    # Eigendecompose each profile once. All egos project onto the same basis.
    rho_np = model.build_profiles().detach().cpu().numpy()
    eigvals, eigvecs = [], []
    for k in range(args.K):
        w, v = top_eigenmodes(rho_np[k], args.n_modes)
        eigvals.append(w)
        eigvecs.append(v)
    eigvals = np.stack(eigvals)
    eigvecs = np.stack(eigvecs)
    np.save(out / "profile_eigvals.npy", eigvals)
    np.save(out / "profile_eigvecs.npy", eigvecs)

    focus = args.mode_profile - 1
    log(f"[modes] projecting onto profile {args.mode_profile}, "
        f"eigenvalues {np.round(eigvals[focus], 4)}")
    eigvecs_focus = torch.tensor(eigvecs[focus], dtype=torch.float32,
                                 device=device)

    df = load_parquet(args.parquet)
    by_id = df.groupby("trajectory_id")

    lengths = by_id.size()
    eligible = lengths[lengths >= args.min_len].index.to_numpy()
    log(f"[select] {len(eligible):,} egos with >= {args.min_len} frames")

    n_take = min(args.n_egos, len(eligible))
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(eligible, size=n_take, replace=False)
    log(f"[select] running inference on {n_take:,} egos")

    rows = []
    for i, tid in enumerate(chosen, start=1):
        sub = by_id.get_group(tid).sort_values("time").reset_index(drop=True)
        res = infer_ego(model, sub, rff, scaler_x, scaler_c, eigvecs_focus, device)

        for name in ("pi", "p", "modes", "speed", "headway", "jerk",
                     "context", "time"):
            np.save(out / "by_ego" / f"{tid}_{name}.npy", res[name])

        rows.append(summarize(tid, res, eigvals[focus]))

        if i == 1 or i % 500 == 0 or i == n_take:
            log(f"[infer] {i:>6,}/{n_take:,}")

    catalog = pd.DataFrame(rows)
    catalog.to_csv(out / "catalog.csv", index=False)
    log(f"[save] catalog.csv, {len(catalog):,} egos")

    dwell = [f"dwell_P{k + 1}" for k in range(args.K)]

    print("\n" + "=" * 70)
    print("Top 10 by pi variance (the mixture moves most over time)")
    print("=" * 70)
    print(catalog.nlargest(10, "pi_var_mean")[
        ["trajectory_id", "length", "pi_var_mean", "switches"] + dwell
    ].to_string(index=False))

    print("\n" + "=" * 70)
    print(f"Top 10 by profile-{args.mode_profile} mode flips")
    print("=" * 70)
    print(catalog.nlargest(10, "mode_flips")[
        ["trajectory_id", "length", "mode_flips", "mode2_resp", "mean_jerk"]
    ].to_string(index=False))

    for k in range(args.K):
        print("\n" + "=" * 70)
        print(f"Top 10 by dwell in profile {k + 1}")
        print("=" * 70)
        print(catalog.nlargest(10, f"dwell_P{k + 1}")[
            ["trajectory_id", "length", f"dwell_P{k + 1}",
             "mean_speed", "mean_headway", "mean_jerk"]
        ].to_string(index=False))

    log("\n[done]")


if __name__ == "__main__":
    main()
