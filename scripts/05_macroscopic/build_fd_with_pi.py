#!/usr/bin/env python
"""
Overlay the mean profile mixture onto each fundamental-diagram cell.

Upstream:
  scripts/05_macroscopic/build_fd.py -> fd_points.csv, fd_grid.npz
  scripts/03_infer/infer.py          -> catalog.csv, by_ego/{tid}_pi.npy
                                        by_ego/{tid}_time.npy
  scripts/00_preprocess/merge_chunks.py -> full_qdm.parquet  (for x, y per frame)
Downstream:
  none

For every FD cell, average pi_k(c_t) over all inference frames that fall inside
it. This asks whether the profiles the model learned from individual behavior
line up with the macroscopic state of the traffic: if the free-flow branch is
dominated by one profile and the congested branch by another, then a microscopic
mixture weight is carrying macroscopic information.

Cells with no inference coverage get NaN, not zero. A cell no sampled ego drove
through is not a cell where the mixture was uniform.

Frames are matched to their saved pi by timestamp, not by position. Inference
writes one row per parquet frame, so the two agree, but joining on time means a
mismatch shows up as missing frames rather than a silent shift.

Writes:
  fd_with_pi.csv   fd_points.csv plus pi_bar_P{k} and n_pi_frames

Usage:
  python build_fd_with_pi.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qdm.config import (
    ANALYSIS_DIR,
    INFERENCE_DIR,
    K,
    PARQUET_NAME,
    PROCESSED_DIR,
)
from qdm.io import align_by_time
from qdm.lanes import assign_lane_arr
from qdm.signals import log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(PROCESSED_DIR / PARQUET_NAME))
    ap.add_argument("--fd-dir", default=str(ANALYSIS_DIR / "fd"))
    ap.add_argument("--inference-dir", default=str(INFERENCE_DIR))
    ap.add_argument("--out", default=None,
                    help="Default: <fd-dir>/fd_with_pi.csv")
    ap.add_argument("--K", type=int, default=K)
    args = ap.parse_args()

    fd_dir = Path(args.fd_dir)
    inf_dir = Path(args.inference_dir)
    by_ego = inf_dir / "by_ego"
    out_path = Path(args.out) if args.out else fd_dir / "fd_with_pi.csv"

    fd = pd.read_csv(fd_dir / "fd_points.csv")
    grid = np.load(fd_dir / "fd_grid.npz")
    x_edges, t_edges = grid["x_edges"], grid["t_edges"]
    nx, nt = len(x_edges) - 1, len(t_edges) - 1
    log(f"[grid] {nx} x-cells x {nt} t-cells, from fd_grid.npz")

    catalog = pd.read_csv(inf_dir / "catalog.csv")
    inf_tids = catalog["trajectory_id"].astype(int).to_numpy()
    log(f"[inference] {len(inf_tids):,} trajectories in the catalog")

    log(f"[load] {args.parquet}")
    df = pd.read_parquet(args.parquet,
                         columns=["trajectory_id", "time", "x", "y"])
    df = df[df["trajectory_id"].isin(set(inf_tids.tolist()))]
    df = df.sort_values(["trajectory_id", "time"], kind="mergesort")
    log(f"[load] {len(df):,} rows for the inference trajectories")

    lanes = (1, 2, 3, 4)
    sum_pi = {ln: np.zeros((nx, nt, args.K)) for ln in lanes}
    n_pi = {ln: np.zeros((nx, nt), dtype=np.int64) for ln in lanes}

    grouped = dict(list(df.groupby("trajectory_id", sort=False)))
    n_missing = n_dropped = 0

    for i, tid in enumerate(inf_tids, start=1):
        if i == 1 or i % 2_000 == 0:
            log(f"[bin] {i:>7,}/{len(inf_tids):,}")

        pi_path = by_ego / f"{tid}_pi.npy"
        time_path = by_ego / f"{tid}_time.npy"
        if not pi_path.exists() or not time_path.exists() or tid not in grouped:
            n_missing += 1
            continue

        pi = np.load(pi_path)                 # (T, K)
        t_saved = np.load(time_path)          # (T,)
        g = grouped[tid]

        # Join by timestamp: which parquet rows have a saved pi?
        idx_pq, idx_pi = align_by_time(g["time"].to_numpy(), t_saved)
        n_dropped += len(g) - len(idx_pq)
        if len(idx_pq) == 0:
            continue

        x = g["x"].to_numpy()[idx_pq]
        y = g["y"].to_numpy()[idx_pq]
        t = g["time"].to_numpy()[idx_pq]
        pi_rows = pi[idx_pi]

        lane = assign_lane_arr(y)
        xi = np.searchsorted(x_edges, x, side="right") - 1
        ti = np.searchsorted(t_edges, t, side="right") - 1

        ok = ((xi >= 0) & (xi < nx) & (ti >= 0) & (ti < nt)
              & (lane >= 1) & (lane <= 4))

        for j in np.nonzero(ok)[0]:
            ln = int(lane[j])
            sum_pi[ln][xi[j], ti[j]] += pi_rows[j]
            n_pi[ln][xi[j], ti[j]] += 1

    log(f"[bin] {n_missing:,} trajectories skipped (no saved pi)")
    if n_dropped:
        log(f"[bin] {n_dropped:,} parquet frames had no matching saved pi")

    # Merge onto the FD rows. build_fd.py wrote x_index and t_index, so no
    # recomputation from the float cell corners.
    pi_cols = {f"pi_bar_P{k + 1}": [] for k in range(args.K)}
    n_frames = []

    for _, row in fd.iterrows():
        ln = int(row["lane"])
        xi, ti = int(row["x_index"]), int(row["t_index"])

        n = n_pi[ln][xi, ti] if ln in n_pi else 0
        n_frames.append(int(n))

        if n > 0:
            mean_pi = sum_pi[ln][xi, ti] / n
            for k in range(args.K):
                pi_cols[f"pi_bar_P{k + 1}"].append(float(mean_pi[k]))
        else:
            for k in range(args.K):
                pi_cols[f"pi_bar_P{k + 1}"].append(np.nan)

    fd["n_pi_frames"] = n_frames
    for col, vals in pi_cols.items():
        fd[col] = vals

    fd.to_csv(out_path, index=False)

    covered = int((fd["n_pi_frames"] > 0).sum())
    log(f"[done] {len(fd):,} cells, {covered:,} with inference coverage "
        f"({100 * covered / max(len(fd), 1):.1f}%)")
    log(f"[done] -> {out_path}")


if __name__ == "__main__":
    main()
