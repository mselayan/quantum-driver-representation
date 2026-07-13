#!/usr/bin/env python
"""
Pairwise Frobenius distances between the K trained profiles.

Upstream:
  scripts/02_train/train_qdm.py -> data/models/rho_profile_{k}.npy
Downstream:
  none

||rho_i - rho_j||_F measures how far apart two profiles sit as operators. A
small distance means the two profiles are close to redundant, so this is the
check on whether K was chosen too large.

Usage:
  python compute_frobenius.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qdm.config import ANALYSIS_DIR, K, MODEL_DIR
from qdm.signals import log


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
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rhos = load_profiles(args.model_dir, args.K)

    dist = np.zeros((args.K, args.K))
    for i in range(args.K):
        for j in range(args.K):
            dist[i, j] = np.linalg.norm(rhos[i] - rhos[j], ord="fro")

    labels = [f"P{k + 1}" for k in range(args.K)]
    table = pd.DataFrame(dist, index=labels, columns=labels)
    table.to_csv(out_dir / "frobenius_distances.csv")

    print(f"\nFrobenius distance matrix (K = {args.K})")
    print("-" * 40)
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))

    print("\nPairwise:")
    for i in range(args.K):
        for j in range(i + 1, args.K):
            print(f"  ||rho_{i + 1} - rho_{j + 1}||_F = {dist[i, j]:.4f}")

    log(f"\n[done] -> {out_dir / 'frobenius_distances.csv'}")


if __name__ == "__main__":
    main()
