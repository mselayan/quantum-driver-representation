#!/usr/bin/env python
"""
Back-project each profile's eigenmodes into behavioral space.

Upstream:
  scripts/02_train/train_qdm.py -> data/models/rho_profile_{k}.npy
                                   data/models/rff_sampler.pkl
                                   data/models/scaler_x.pkl
Downstream:
  none

An eigenvector v_m of rho_k lives in the D-dimensional RFF space and means
nothing on its own. To read it, evaluate |<v_m, phi(x)>|^2 over a grid of raw
behavioral values x = (speed, headway, jerk). That squared projection is the
weight mode m assigns to each point in behavior space, so its peak and its
weighted moments say what the mode is: which speed, which headway, which jerk.

Reported per mode:
  lambda_m   the mode's share of the profile's spectral mass
  peak       the (speed, headway, jerk) with the largest |<v_m, phi(x)>|^2
  mean, std  weighted by |<v_m, phi(x)>|^2, so the spread of the mode

A profile with one dominant lambda is one behavioral mode. Several comparable
lambdas mean the profile holds several modes at once, which is the point of
gamma.

Grid is in standardized units, then inverse-transformed to report raw values.

Usage:
  python profile_signatures.py
"""

import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qdm.config import ANALYSIS_DIR, BEHAVIORAL_COLS, K, MODEL_DIR
from qdm.features import load_features, phi
from qdm.model import top_eigenmodes
from qdm.signals import log

N_GRID = 30          # per standardized dimension
GRID_SPAN = 3.0      # +/- standard deviations
MIN_EIGENVALUE = 1e-3


def load_profiles(model_dir, K):
    rhos = []
    for k in range(1, K + 1):
        path = Path(model_dir) / f"rho_profile_{k}.npy"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Train first.")
        rhos.append(np.load(path))
    return rhos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--out", default=str(ANALYSIS_DIR))
    ap.add_argument("--K", type=int, default=K)
    ap.add_argument("--n-modes", type=int, default=5,
                    help="Maximum modes to report per profile.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rhos = load_profiles(args.model_dir, args.K)
    rff, scaler_x, _ = load_features(args.model_dir)

    axis = np.linspace(-GRID_SPAN, GRID_SPAN, N_GRID)
    grid = np.array(list(product(axis, axis, axis)))     # standardized units
    Phi = phi(grid, rff)
    log(f"[grid] {N_GRID}^3 = {len(grid):,} points, standardized")

    rows = []
    for k, rho in enumerate(rhos):
        eigvals, eigvecs = top_eigenmodes(rho, args.n_modes)
        n_real = max(1, int(np.sum(eigvals > MIN_EIGENVALUE)))

        print(f"\n  Profile {k + 1}  "
              f"({n_real} mode{'s' if n_real > 1 else ''} above "
              f"lambda = {MIN_EIGENVALUE})")

        for m in range(n_real):
            # |<v_m, phi(x)>|^2 is the mode's weight over behavior space.
            weight = (Phi @ eigvecs[:, m]) ** 2
            total = weight.sum()
            if total <= 0:
                continue
            w = weight / total

            peak_std = grid[np.argmax(weight)]
            mean_std = (w[:, None] * grid).sum(axis=0)
            var_std = (w[:, None] * (grid - mean_std) ** 2).sum(axis=0)

            peak = scaler_x.inverse_transform(peak_std.reshape(1, -1))[0]
            mean = scaler_x.inverse_transform(mean_std.reshape(1, -1))[0]
            std = np.sqrt(var_std) * scaler_x.scale_

            print(f"    mode {m + 1}  lambda = {eigvals[m]:.4f}")
            for i, col in enumerate(BEHAVIORAL_COLS):
                print(f"      {col:<9s} peak {peak[i]:>8.2f}   "
                      f"mean {mean[i]:>8.2f}   std {std[i]:>7.2f}")

            row = {"profile": k + 1, "mode": m + 1, "eigenvalue": eigvals[m]}
            for i, col in enumerate(BEHAVIORAL_COLS):
                row[f"{col}_peak"] = peak[i]
                row[f"{col}_mean"] = mean[i]
                row[f"{col}_std"] = std[i]
            rows.append(row)

        np.savez(out_dir / f"profile_{k + 1}_modes.npz",
                 eigvals=eigvals, eigvecs=eigvecs,
                 behavioral_cols=np.array(BEHAVIORAL_COLS))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "profile_signatures.csv", index=False)
    log(f"\n[done] -> {out_dir / 'profile_signatures.csv'}")


if __name__ == "__main__":
    main()
