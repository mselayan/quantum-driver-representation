#!/usr/bin/env python
"""
Generate a small synthetic parquet with the schema of the real one.

The pipeline can then be run end to end on a laptop in about a minute, which
verifies that the stages connect and that every file a downstream stage expects
actually gets written. It says nothing about the findings: the data is invented.

Three latent driver types with different speed, headway and jerk tendencies, so
the profiles have something to separate. Positions are integrated from speed so
that the FD, time-space, and hysteresis stages have a coherent x(t).

Usage:
  python make_synthetic_data.py --out data/processed/full_qdm.parquet
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qdm.config import (
    FINAL_COLUMNS,
    LANE_CENTERS,
    PARQUET_NAME,
    PROCESSED_DIR,
    SAMPLE_DT,
)
from qdm.signals import log

# (mean speed ft/s, mean headway ft, jerk scale)
TYPES = [
    (85.0, 180.0, 0.5),   # fast, large gap, smooth
    (45.0,  60.0, 2.5),   # slow, tight gap, jerky
    (65.0, 110.0, 1.2),   # in between
]


def make_ego(tid, rng, n_frames, t0, x0):
    kind = rng.integers(0, len(TYPES))
    v_mu, h_mu, j_sd = TYPES[kind]

    t = t0 + np.arange(n_frames) * SAMPLE_DT

    # Speed: an AR(1) around the type's mean, so it has temporal structure.
    v = np.empty(n_frames)
    v[0] = rng.normal(v_mu, 5.0)
    for k in range(1, n_frames):
        v[k] = 0.98 * v[k - 1] + 0.02 * v_mu + rng.normal(0, 1.2)
    v = np.clip(v, 5.0, 110.0)

    # Westbound: x decreases.
    x = x0 - np.cumsum(v) * SAMPLE_DT

    lane = int(rng.integers(1, 5))
    y = LANE_CENTERS[lane - 1] + rng.normal(0, 1.0, n_frames)

    headway = np.clip(rng.normal(h_mu, h_mu * 0.25, n_frames), 5.0, None)
    jerk = rng.normal(0.0, j_sd, n_frames)

    # Context, loosely anticorrelated with speed so it carries signal.
    density = np.clip(
        rng.poisson(np.clip(28.0 - 0.22 * v, 1.0, None)), 0, None
    ).astype(float)
    sp_entropy = np.clip(rng.normal(1.0 + 0.02 * density, 0.25), 0.0, None)
    accel_entropy = np.clip(rng.normal(0.8 + 0.015 * density, 0.25), 0.0, None)

    return pd.DataFrame({
        "trajectory_id": tid,
        "time": t,
        "x": x,
        "y": y,
        "speed": v,
        "headway": headway,
        "jerk": jerk,
        "density": density,
        "sp_entropy": sp_entropy,
        "accel_entropy": accel_entropy,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROCESSED_DIR / PARQUET_NAME))
    ap.add_argument("--n-egos", type=int, default=300)
    ap.add_argument("--min-frames", type=int, default=250)
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    frames = []
    for tid in range(args.n_egos):
        n = int(rng.integers(args.min_frames, args.max_frames))
        t0 = 1669120000.0 + float(rng.uniform(0, 300))
        x0 = 322000.0 + float(rng.uniform(0, 4000))
        frames.append(make_ego(tid, rng, n, t0, x0))

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["trajectory_id", "time"]).reset_index(drop=True)
    df = df[FINAL_COLUMNS]

    # A few NaNs, so the imputation path in io.load_parquet is exercised.
    n_nan = max(1, len(df) // 200)
    df.loc[rng.choice(len(df), n_nan, replace=False), "headway"] = np.nan
    df.loc[rng.choice(len(df), n_nan, replace=False), "accel_entropy"] = np.nan

    df.to_parquet(out, index=False)
    log(f"[done] {len(df):,} rows, {df['trajectory_id'].nunique():,} egos -> {out}")
    log(f"[done] injected {n_nan:,} NaN headway and {n_nan:,} NaN accel_entropy")


if __name__ == "__main__":
    main()
