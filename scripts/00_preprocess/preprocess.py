#!/usr/bin/env python
"""
Compute the six modeled variables for a chunk of eligible ego trajectories.

Upstream:
  data/raw/<dataset>.json                      I-24 MOTION westbound trajectories
Downstream:
  scripts/00_preprocess/merge_chunks.py

Behavioral variables (per ego frame):
  speed          longitudinal speed magnitude          ft/s
  headway        gap to same-lane leader, net of ego length   ft
  jerk           rate of change of acceleration        ft/s^3

Context variables (per ego frame):
  density        vehicles within 150 m (euclidean)     count
  sp_entropy     entropy of Tesla-zone neighbors' 1 s speed changes   nats
  accel_entropy  entropy of forward-zone neighbors' accelerations     nats

The spacetime index is built from EVERY westbound vehicle: every class, every
duration. Only the egos that are scored are filtered (HDV class, >= 10 s). So a
short-lived truck still acts as a leader and still contributes to a neighbor
entropy, it just never gets a row of its own.

Chunking splits the ego list only. Every chunk rebuilds the full index, so
neighbor lookups are complete in every chunk and the chunks are independent.

Runs on one core. Chunks are independent, so run them in parallel with a job
array on a cluster, or sequentially with --chunk 0 --n-chunks 1 on a laptop
(slow: the index holds every vehicle in memory).

Usage:
  python preprocess.py --json data/raw/i24.json --out data/processed \
      --chunk 0 --n-chunks 32
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qdm.config import (
    FINAL_COLUMNS,
    FORWARD_LON_DIST,
    MAX_SEARCH_RADIUS,
    OMNI_RADIUS,
    PAST_OFFSETS,
    PROCESSED_DIR,
    WARMUP_FRAMES,
    WARMUP_SECS,
)
from qdm.lanes import assign_lane, assign_lane_arr, lanes_to_scan
from qdm.signals import central_diff, log, shannon_entropy
from qdm.spacetime import (
    build_spacetime_index,
    find_leader,
    load_trajectories,
    neighbors_in_x_window,
    tesla_neighbor_union,
)


def compute_kinematics(traj):
    """speed, accel, jerk for one trajectory. Computed for EVERY vehicle:
    neighbors need speed for the index and accel for accel_entropy."""
    t = np.asarray(traj["timestamp"], dtype=np.float64)
    x = np.asarray(traj["x_position"], dtype=np.float64)
    speed = np.abs(central_diff(x, t))
    accel = central_diff(speed, t)
    return {"speed": speed, "accel": accel, "jerk": central_diff(accel, t)}


def rows_for_ego(ego_idx, trajs_with_vars, time_index, traj_lookup):
    """One row per scored frame of a single ego."""
    traj, vars_ = trajs_with_vars[ego_idx]
    ts = traj["timestamp"]
    xs = traj["x_position"]
    ys = traj["y_position"]
    ego_length = traj.get("length", 5.0)
    speed, jerk = vars_["speed"], vars_["jerk"]

    rows = []
    for k in range(len(ts)):
        t = ts[k]
        # Warm-up: drop the first ~1 s so windowed quantities have history.
        if k < WARMUP_FRAMES or t < ts[0] + WARMUP_SECS:
            continue

        t_key = round(t, 2)
        snap = time_index.get(t_key)
        if snap is None or t_key not in traj_lookup[ego_idx]:
            continue

        ego_x, ego_y = xs[k], ys[k]
        ego_h = traj_lookup[ego_idx][t_key][3]

        # -- headway: nearest same-lane leader, net of ego length --
        headway = np.nan
        leader = find_leader(ego_x, ego_y, snap, ego_idx)
        if leader is not None:
            _, _, gap = leader
            headway = gap - ego_length

        # -- neighbors in the x-window, self excluded --
        win = neighbors_in_x_window(snap, ego_x, MAX_SEARCH_RADIUS)
        nidx, nx, ny = win["idx"], win["x"], win["y"]
        keep = nidx != ego_idx
        nidx, nx, ny = nidx[keep], nx[keep], ny[keep]

        density = 0
        accel_entropy = 0.0

        if len(nidx) > 0:
            dxn = nx - ego_x
            dyn = ny - ego_y
            dist = np.sqrt(dxn * dxn + dyn * dyn)

            # -- density: everything within the omni radius --
            density = int(np.sum(dist <= OMNI_RADIUS))

            # -- accel_entropy: forward zone, ego's 3-lane window --
            ego_lane = assign_lane(ego_y)
            if ego_lane is not None:
                scan = lanes_to_scan(ego_lane)
                ahead = (nx < ego_x) & ((ego_x - nx) <= FORWARD_LON_DIST)
                fwd_idx = nidx[ahead & np.isin(assign_lane_arr(ny), scan)]

                accels = []
                for ni in fwd_idx:
                    lk = traj_lookup.get(int(ni))
                    if lk is None or t_key not in lk:
                        continue
                    frame = lk[t_key][4]
                    a_n = trajs_with_vars[int(ni)][1]["accel"][frame]
                    if not np.isnan(a_n):
                        accels.append(a_n)
                accel_entropy = shannon_entropy(np.asarray(accels))

        # -- sp_entropy: 1 s speed change of every Tesla-zone neighbor --
        neighbor_set = tesla_neighbor_union(
            time_index, ego_x, ego_y, ego_h, ego_idx, t_key, PAST_OFFSETS
        )
        t_past = round(t_key - 1.0, 2)
        speed_changes = []
        for ni in neighbor_set:
            lk = traj_lookup.get(ni)
            if lk is None or t_key not in lk or t_past not in lk:
                continue
            speed_changes.append(lk[t_key][2] - lk[t_past][2])
        sp_entropy = shannon_entropy(np.asarray(speed_changes))

        rows.append({
            "trajectory_id": ego_idx,
            "time": t,
            "x": ego_x,
            "y": ego_y,
            "speed": speed[k],
            "headway": headway,
            "jerk": jerk[k],
            "density": density,
            "sp_entropy": sp_entropy,
            "accel_entropy": accel_entropy,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="I-24 MOTION trajectory JSON")
    ap.add_argument("--out", default=str(PROCESSED_DIR))
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--n-chunks", type=int, default=1)
    ap.add_argument("--limit-egos", type=int, default=None,
                    help="Score only the first N eligible egos. For a laptop "
                         "smoke test; the index still holds every vehicle.")
    args = ap.parse_args()

    if not 0 <= args.chunk < args.n_chunks:
        ap.error(f"--chunk must be in [0, {args.n_chunks})")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log(f"PREPROCESS chunk {args.chunk + 1}/{args.n_chunks}")
    log("=" * 70)

    trajs, is_eligible = load_trajectories(args.json)

    eligible = np.nonzero(is_eligible)[0]
    eligible.sort()
    if args.limit_egos is not None:
        eligible = eligible[:args.limit_egos]
        log(f"[limit] scoring only the first {len(eligible):,} eligible egos")

    # np.array_split handles a remainder that does not divide evenly, which a
    # floor-division slice would drop from the last chunk.
    my_egos = np.array_split(eligible, args.n_chunks)[args.chunk]
    log(f"[chunk] {len(my_egos):,} of {len(eligible):,} egos assigned")

    if len(my_egos) == 0:
        log("[chunk] nothing to do")
        return

    log(f"[kinematics] speed/accel/jerk for all {len(trajs):,} vehicles")
    trajs_with_vars = [(t, compute_kinematics(t)) for t in trajs]

    time_index, traj_lookup = build_spacetime_index(trajs_with_vars)

    log("[compute] six variables for the assigned egos")
    rows = []
    every = max(1, len(my_egos) // 50)
    for n, ego_idx in enumerate(my_egos, start=1):
        rows.extend(rows_for_ego(int(ego_idx), trajs_with_vars,
                                 time_index, traj_lookup))
        if n % every == 0 or n == len(my_egos):
            log(f"[compute] {n:,}/{len(my_egos):,} egos, {len(rows):,} rows")

    df = pd.DataFrame(rows, columns=FINAL_COLUMNS)
    path = out_dir / f"qdm_chunk_{args.chunk:04d}.parquet"
    df.to_parquet(path, index=False)
    log(f"[done] {len(df):,} rows, {df['trajectory_id'].nunique():,} egos -> {path}")


if __name__ == "__main__":
    main()
