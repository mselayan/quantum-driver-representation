#!/usr/bin/env python
"""
Purity and pairwise mutual information under each trained profile.

Upstream:
  scripts/02_train/train_qdm.py -> data/models/rho_profile_{k}.npy
                                   data/models/rff_sampler.pkl
                                   data/models/scaler_x.pkl
Downstream:
  none

Purity: tr(rho_k^2) = ||rho_k||_F^2, in [1/D, 1]. A pure (rank-1) profile scores
1 and represents one behavioral mode. A lower value means the spectral mass is
spread over several eigendirections, which is what the gamma penalty produces.

Mutual information: each profile implies a joint density over the behavioral
variables through the Born rule,

    p(v, s, j) ~ phi(v, s, j)^T rho_k phi(v, s, j)

evaluated on a grid and normalized. I(v;s), I(v;j) and I(s;j) are then computed
from the marginals of that joint. High MI means the profile couples that pair of
variables: it does not merely say each is high or low, it says how one moves with
the other. A profile with high purity but low MI is a single point in behavior
space; high MI means the profile encodes a relationship.

Grid ranges are in raw units and are set to cover the observed data.

Usage:
  python compute_interactions.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qdm.config import ANALYSIS_DIR, K, MODEL_DIR
from qdm.features import load_features, phi_of_raw
from qdm.signals import log

# Grid over the raw behavioral variables: speed ft/s, headway ft, jerk ft/s^3.
SPEED_RANGE = (0.0, 100.0)
HEADWAY_RANGE = (0.0, 300.0)
JERK_RANGE = (-6.0, 6.0)
N_SPEED, N_HEADWAY, N_JERK = 30, 30, 15


def load_profiles(model_dir, K):
    rhos = []
    for k in range(1, K + 1):
        path = Path(model_dir) / f"rho_profile_{k}.npy"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Train first.")
        rho = np.load(path)
        tr = float(np.trace(rho))
        if abs(tr - 1.0) > 1e-3:
            log(f"[warn] rho_{k} trace {tr:.6f}, renormalizing")
            rho = rho / tr
        rhos.append(rho)
    return np.stack(rhos)


def joint_distributions(rho_k, rff, scaler_x):
    """p(v, s, j | rho_k) on the grid, one per profile. Shape (K, Nv, Ns, Nj)."""
    v = np.linspace(*SPEED_RANGE, N_SPEED)
    s = np.linspace(*HEADWAY_RANGE, N_HEADWAY)
    j = np.linspace(*JERK_RANGE, N_JERK)

    V, S, J = np.meshgrid(v, s, j, indexing="ij")
    X = np.stack([V.ravel(), S.ravel(), J.ravel()], axis=1)
    Phi = phi_of_raw(X, rff, scaler_x)

    out = np.empty((len(rho_k), N_SPEED, N_HEADWAY, N_JERK))
    for k in range(len(rho_k)):
        w = np.clip(np.einsum("md,de,me->m", Phi, rho_k[k], Phi), 0.0, None)
        total = w.sum()
        if total > 0:
            w = w / total
        out[k] = w.reshape(N_SPEED, N_HEADWAY, N_JERK)
    return out


def mutual_information(p_joint):
    """I(A;B) in nats from a 2D joint. Zero iff the two axes are independent."""
    p = p_joint / max(p_joint.sum(), 1e-12)
    p_indep = p.sum(axis=1, keepdims=True) * p.sum(axis=0, keepdims=True)
    mask = (p > 0) & (p_indep > 0)
    if not mask.any():
        return 0.0
    return float((p[mask] * np.log(p[mask] / p_indep[mask])).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--out", default=str(ANALYSIS_DIR))
    ap.add_argument("--K", type=int, default=K)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rho_k = load_profiles(args.model_dir, args.K)
    rff, scaler_x, _ = load_features(args.model_dir)

    log(f"[grid] {N_SPEED} x {N_HEADWAY} x {N_JERK} over (speed, headway, jerk)")
    joints = joint_distributions(rho_k, rff, scaler_x)

    rows = []
    for k in range(args.K):
        p = joints[k]
        i_vs = mutual_information(p.sum(axis=2))   # speed vs headway
        i_vj = mutual_information(p.sum(axis=1))   # speed vs jerk
        i_sj = mutual_information(p.sum(axis=0))   # headway vs jerk

        rows.append({
            "profile": k + 1,
            "purity": float((rho_k[k] * rho_k[k]).sum()),
            "I_speed_headway": i_vs,
            "I_speed_jerk": i_vj,
            "I_headway_jerk": i_sj,
            "I_total": i_vs + i_vj + i_sj,
        })

        print(f"\n  Profile {k + 1}")
        print(f"    purity              {rows[-1]['purity']:.4f}")
        print(f"    I(speed; headway)   {i_vs:.4f} nats")
        print(f"    I(speed; jerk)      {i_vj:.4f} nats")
        print(f"    I(headway; jerk)    {i_sj:.4f} nats")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "interaction_indicators.csv", index=False)

    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    log(f"\n[done] -> {out_dir / 'interaction_indicators.csv'}")


if __name__ == "__main__":
    main()
