#!/usr/bin/env python
"""
Per-lane time-space diagrams, and the per-lane parquets the analyses need.

Upstream:
  scripts/00_preprocess/merge_chunks.py -> data/processed/full_qdm.parquet
Downstream:
  scripts/05_macroscopic/hysteresis.py  reads lane{n}.parquet

Per lane:
  lane{n}.parquet         every frame in that lane (trajectory_id, time, x, y, speed)
  tx_diagram_lane{n}.pdf  trajectories drawn as lines colored by speed
  tx_diagram_lane{n}.png

The parquet always holds the full lane. Only the figure subsamples, since
drawing every trajectory is unreadable and slow.

Position is plotted on an inverted axis so that westbound travel, which runs
toward decreasing x, reads downward. Stop-and-go waves then appear as the usual
stripes moving upstream.

Usage:
  python build_xt_diagrams.py --parquet data/processed/full_qdm.parquet
  python build_xt_diagrams.py --tmin 1669119900 --tmax 1669120400   # zoom
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qdm.config import ANALYSIS_DIR, PARQUET_NAME, PROCESSED_DIR
from qdm.lanes import assign_lane_arr
from qdm.signals import log


def render_lane(df_lane, lane, out_dir, vmax, n_trajectories, seed,
                tmin=None, tmax=None):
    tids = df_lane["trajectory_id"].unique()
    rng = np.random.default_rng(seed)
    if len(tids) > n_trajectories:
        picked = rng.choice(tids, size=n_trajectories, replace=False)
    else:
        picked = tids
    log(f"[plot] lane {lane}: drawing {len(picked):,} of {len(tids):,} trajectories")

    sub = df_lane[df_lane["trajectory_id"].isin(picked)]
    sub = sub.sort_values(["trajectory_id", "time"], kind="mergesort")

    # One line per trajectory, split into segments so each can take its own
    # color. A LineCollection draws them all in a single pass.
    segments, colors = [], []
    for _, g in sub.groupby("trajectory_id", sort=False):
        if len(g) < 2:
            continue
        pts = np.column_stack([g["time"].to_numpy(), g["x"].to_numpy()])
        v = g["speed"].to_numpy()
        segments.append(np.stack([pts[:-1], pts[1:]], axis=1))
        colors.append(0.5 * (v[:-1] + v[1:]))

    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = matplotlib.colormaps["turbo_r"]

    if segments:
        lc = LineCollection(np.concatenate(segments, axis=0),
                            cmap=cmap, norm=norm, linewidths=0.5, alpha=0.85)
        lc.set_array(np.concatenate(colors, axis=0))
        ax.add_collection(lc)
        ax.autoscale()
        if tmin is not None:
            ax.set_xlim(left=tmin)
        if tmax is not None:
            ax.set_xlim(right=tmax)
        ax.invert_yaxis()     # westbound travels toward decreasing x

    ax.set_xlabel("time (s)", fontsize=11)
    ax.set_ylabel("position x (ft)", fontsize=11)
    ax.set_title(f"I-24 MOTION westbound, lane {lane} "
                 f"({len(picked):,} trajectories, colored by speed)", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.01)
    cbar.set_label("speed (ft/s)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.savefig(out_dir / f"tx_diagram_lane{lane}.pdf")
    fig.savefig(out_dir / f"tx_diagram_lane{lane}.png", dpi=200)
    plt.close(fig)
    log(f"[plot] lane {lane}: wrote tx_diagram_lane{lane}.pdf/.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(PROCESSED_DIR / PARQUET_NAME))
    ap.add_argument("--out", default=str(ANALYSIS_DIR / "xt"))
    ap.add_argument("--vmax", type=float, default=90.0,
                    help="Top of the speed colormap, ft/s.")
    ap.add_argument("--n-trajectories", type=int, default=1000,
                    help="Trajectories to draw per lane. The parquet is always "
                         "written in full.")
    ap.add_argument("--tmin", type=float, default=None,
                    help="Crop the figure to this start time (UTC seconds).")
    ap.add_argument("--tmax", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"[load] {args.parquet}")
    df = pd.read_parquet(args.parquet,
                         columns=["trajectory_id", "time", "x", "y", "speed"])
    log(f"[load] {len(df):,} rows")

    df["lane"] = assign_lane_arr(df["y"].to_numpy())
    df = df[df["lane"].between(1, 4)].copy()
    log(f"[load] {len(df):,} rows in lanes 1-4")

    for lane in (1, 2, 3, 4):
        sub = df[df["lane"] == lane]
        if len(sub) == 0:
            log(f"[skip] lane {lane} is empty")
            continue
        log(f"[lane {lane}] {len(sub):,} rows, "
            f"{sub['trajectory_id'].nunique():,} trajectories")

        path = out_dir / f"lane{lane}.parquet"
        sub[["trajectory_id", "time", "x", "y", "speed"]].to_parquet(
            path, index=False)
        log(f"[lane {lane}] wrote {path.name}")

        render_lane(sub, lane, out_dir, args.vmax, args.n_trajectories,
                    args.seed, args.tmin, args.tmax)

    log("[done]")


if __name__ == "__main__":
    main()
