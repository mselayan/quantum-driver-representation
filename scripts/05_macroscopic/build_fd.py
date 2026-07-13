#!/usr/bin/env python
"""
Fundamental diagram by Edie's generalized definitions on rectangular cells.

Upstream:
  scripts/00_preprocess/merge_chunks.py -> data/processed/full_qdm.parquet
Downstream:
  scripts/05_macroscopic/build_fd_with_pi.py  reads fd_points.csv and fd_grid.npz

For a space-time region A of area |A|:

  density(A) = t(A) / |A|      total vehicle-time in A, per unit area
  flow(A)    = d(A) / |A|      total vehicle-distance in A, per unit area
  speed(A)   = d(A) / t(A)

d(A) and t(A) sum over every trajectory that crosses A. Trajectories that only
partly cross a cell are clipped at the boundary by linear interpolation, in time
first and then in space, so a vehicle contributes exactly the distance and time
it spent inside.

Limitation: rectangular cells do not respect kinematic-wave stationarity, so the
congested branch is smeared. He and Wu (2025) aggregate over parallelograms
aligned with the wave speed instead; scripts/05_macroscopic/hysteresis.py does
that for a single wave.

References:
  Edie, L. (1965). Discussion of traffic stream measurements and definitions.
    Proc. 2nd Int. Symp. on Transportation and Traffic Theory, 139-154.
  He, Z. and Wu, C. (2025). Constructing the fundamental diagrams of traffic
    flow from large-scale vehicle trajectory data. arXiv:2507.09648.

Writes:
  fd_points.csv  one row per non-empty (lane, x-cell, t-cell)
  fd_grid.npz    the cell edges, so downstream code reuses this exact grid
  fig_fd.pdf/png flow against density

Usage:
  python build_fd.py --parquet data/processed/full_qdm.parquet
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qdm.config import (
    ANALYSIS_DIR,
    FD_DT_SEC,
    FD_DX_FT,
    FT_PER_MILE,
    MPH_PER_FPS,
    PARQUET_NAME,
    PROCESSED_DIR,
    SEC_PER_HR,
)
from qdm.lanes import assign_lane_arr
from qdm.signals import log


def cell_edges(df, dx, dt):
    x_min, x_max = float(df["x"].min()), float(df["x"].max())
    t_min, t_max = float(df["time"].min()), float(df["time"].max())
    x_edges = np.arange(np.floor(x_min / dx) * dx, x_max + dx, dx)
    t_edges = np.arange(np.floor(t_min / dt) * dt, t_max + dt, dt)
    return x_edges, t_edges


def edie_contributions(t, x, t_lo, t_hi, x_lo, x_hi):
    """Distance and time one trajectory segment spends inside one cell.

    Clip in time by interpolating entry and exit samples at t_lo and t_hi, then
    clip each consecutive pair against [x_lo, x_hi].
    """
    if len(t) < 2 or t[-1] <= t_lo or t[0] >= t_hi:
        return 0.0, 0.0

    inside = (t > t_lo) & (t < t_hi)
    tt, xx = t[inside], x[inside]

    if t[0] < t_lo:                          # interpolate the entry sample
        i = max(np.searchsorted(t, t_lo) - 1, 0)
        j = min(i + 1, len(t) - 1)
        if t[j] > t[i]:
            f = (t_lo - t[i]) / (t[j] - t[i])
            tt = np.concatenate([[t_lo], tt])
            xx = np.concatenate([[x[i] + f * (x[j] - x[i])], xx])

    if t[-1] > t_hi:                         # interpolate the exit sample
        i = max(np.searchsorted(t, t_hi) - 1, 0)
        j = min(i + 1, len(t) - 1)
        if t[j] > t[i]:
            f = (t_hi - t[i]) / (t[j] - t[i])
            tt = np.concatenate([tt, [t_hi]])
            xx = np.concatenate([xx, [x[i] + f * (x[j] - x[i])]])

    if len(tt) < 2:
        return 0.0, 0.0

    d_total = t_total = 0.0
    for k in range(len(tt) - 1):
        ta, tb = tt[k], tt[k + 1]
        xa, xb = xx[k], xx[k + 1]
        if tb <= ta:
            continue
        if (xa < x_lo and xb < x_lo) or (xa > x_hi and xb > x_hi):
            continue

        dx_seg = xb - xa
        if dx_seg == 0.0:                    # stationary within the cell
            if x_lo <= xa <= x_hi:
                t_total += tb - ta
            continue

        # x(s) = xa + s * dx_seg for s in [0, 1]. Intersect with [x_lo, x_hi].
        s_lo = (x_lo - xa) / dx_seg
        s_hi = (x_hi - xa) / dx_seg
        s_a, s_b = (s_lo, s_hi) if s_lo <= s_hi else (s_hi, s_lo)
        s_a, s_b = max(s_a, 0.0), min(s_b, 1.0)
        if s_b <= s_a:
            continue

        t_total += (tb - ta) * (s_b - s_a)
        d_total += abs(dx_seg) * (s_b - s_a)

    return d_total, t_total


def inlane_runs(t, x, lane):
    """Maximal runs of consecutive frames sharing one nonzero lane.

    A lane change splits a trajectory: the two halves belong to different lanes
    and must not be aggregated into the same lane's cells.
    """
    n = len(lane)
    i = 0
    while i < n:
        if lane[i] == 0:
            i += 1
            continue
        j = i
        while j < n and lane[j] == lane[i]:
            j += 1
        if j - i >= 2:
            yield t[i:j], x[i:j], int(lane[i])
        i = j


def build_fd(df, x_edges, t_edges, dx, dt, lanes):
    nx = len(x_edges) - 1
    nt = len(t_edges) - 1
    log(f"[fd] {nx} x-cells (dx={dx} ft) x {nt} t-cells (dt={dt} s)")

    accum_d = {ln: np.zeros((nx, nt)) for ln in lanes}
    accum_t = {ln: np.zeros((nx, nt)) for ln in lanes}
    n_seg = {ln: np.zeros((nx, nt), dtype=np.int64) for ln in lanes}

    df = df.sort_values(["trajectory_id", "time"], kind="mergesort")
    grouped = list(df.groupby("trajectory_id", sort=False))
    log(f"[fd] {len(grouped):,} trajectories")

    for i, (_, g) in enumerate(grouped, start=1):
        if i == 1 or i % 10_000 == 0:
            log(f"[fd] trajectory {i:>8,}/{len(grouped):,}")

        t = g["time"].to_numpy()
        x = g["x"].to_numpy()
        lane = assign_lane_arr(g["y"].to_numpy())

        for t_run, x_run, ln in inlane_runs(t, x, lane):
            if ln not in accum_d:
                continue

            # Only the cells this run actually touches.
            ti_lo = max(int(np.searchsorted(t_edges, t_run[0], side="right") - 1), 0)
            ti_hi = min(int(np.searchsorted(t_edges, t_run[-1], side="right")), nt)
            xi_lo = max(int(np.searchsorted(x_edges, x_run.min(), side="right") - 1), 0)
            xi_hi = min(int(np.searchsorted(x_edges, x_run.max(), side="right")), nx)

            for ti in range(ti_lo, ti_hi):
                for xi in range(xi_lo, xi_hi):
                    d, tt = edie_contributions(
                        t_run, x_run,
                        t_edges[ti], t_edges[ti + 1],
                        x_edges[xi], x_edges[xi + 1],
                    )
                    if tt > 0:
                        accum_d[ln][xi, ti] += d
                        accum_t[ln][xi, ti] += tt
                        n_seg[ln][xi, ti] += 1

    area = dx * dt      # ft * s
    rows = []
    for ln in lanes:
        for xi in range(nx):
            for ti in range(nt):
                time_in = accum_t[ln][xi, ti]
                if time_in <= 0:
                    continue
                dist_in = accum_d[ln][xi, ti]
                rows.append({
                    "lane": ln,
                    "x_cell_ft": float(x_edges[xi]),
                    "t_cell_s": float(t_edges[ti]),
                    "x_index": xi,
                    "t_index": ti,
                    "n_segments": int(n_seg[ln][xi, ti]),
                    "density_veh_per_ft": time_in / area,
                    "flow_veh_per_s": dist_in / area,
                    "speed_ft_per_s": dist_in / time_in,
                })

    fd = pd.DataFrame(rows)
    fd["density_veh_per_mile"] = fd["density_veh_per_ft"] * FT_PER_MILE
    fd["flow_veh_per_hour"] = fd["flow_veh_per_s"] * SEC_PER_HR
    fd["speed_mph"] = fd["speed_ft_per_s"] * MPH_PER_FPS
    return fd


def plot_fd(fd, out_dir):
    fig, ax = plt.subplots(figsize=(7.0, 5.5), constrained_layout=True)
    ax.scatter(fd["density_veh_per_mile"], fd["flow_veh_per_hour"],
               s=6, alpha=0.30, c="#6C88C4", edgecolor="none")
    ax.set_xlabel("density (veh/mile)", fontsize=11)
    ax.set_ylabel("flow (veh/h)", fontsize=11)
    ax.set_title("Fundamental diagram, I-24 MOTION westbound\n"
                 "Edie (1965) on rectangular cells", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(out_dir / "fig_fd.pdf")
    fig.savefig(out_dir / "fig_fd.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(PROCESSED_DIR / PARQUET_NAME))
    ap.add_argument("--out", default=str(ANALYSIS_DIR / "fd"))
    ap.add_argument("--dx-ft", type=float, default=FD_DX_FT)
    ap.add_argument("--dt-sec", type=float, default=FD_DT_SEC)
    ap.add_argument("--lanes", type=int, nargs="*", default=[1, 2, 3, 4])
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"[load] {args.parquet}")
    df = pd.read_parquet(args.parquet,
                         columns=["trajectory_id", "time", "x", "y"])
    log(f"[load] {len(df):,} rows, {df['trajectory_id'].nunique():,} trajectories")

    x_edges, t_edges = cell_edges(df, args.dx_ft, args.dt_sec)
    fd = build_fd(df, x_edges, t_edges, args.dx_ft, args.dt_sec, args.lanes)

    fd.to_csv(out_dir / "fd_points.csv", index=False)

    # Save the grid so build_fd_with_pi.py bins onto these exact cells rather
    # than rederiving edges from the parquet and hoping dx and dt still match.
    np.savez(out_dir / "fd_grid.npz",
             x_edges=x_edges, t_edges=t_edges,
             dx_ft=args.dx_ft, dt_sec=args.dt_sec)

    log(f"[done] {len(fd):,} non-empty cells -> {out_dir / 'fd_points.csv'}")
    plot_fd(fd, out_dir)
    log(f"[done] fig_fd.pdf/.png -> {out_dir}")


if __name__ == "__main__":
    main()
