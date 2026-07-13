#!/usr/bin/env python
"""
Hysteresis loop of one platoon through one stop-and-go wave.

Upstream:
  scripts/05_macroscopic/build_xt_diagrams.py -> lane{n}.parquet
  scripts/03_infer/infer.py                   -> catalog.csv
                                                 by_ego/{tid}_pi.npy
                                                 by_ego/{tid}_time.npy
  scripts/05_macroscopic/wave_lane1.json      -> the wave and platoon geometry
Downstream:
  none

A platoon that decelerates into a wave and accelerates out of it does not retrace
its path in the flow-density plane: the congested branch it follows on the way in
sits below the one it follows on the way out. That loop is traffic hysteresis.

Edie's quantities are aggregated over parallelograms rather than rectangles. One
edge follows the local vehicle trajectory, the other follows the wave front. A
rectangle straddling a wave boundary averages congested and uncongested traffic
together and smears the loop; a parallelogram aligned with the wave does not.

For each parallelogram this reports the flow, the density, and the mean profile
mixture pi_bar of the inference egos inside it, so the loop can be read against
the microscopic profile composition at every point around it.

The geometry comes from --wave-json, read off a time-space diagram. Nothing
about the wave is inferred; it is specified.

Writes:
  hysteresis_data.npz  labels, k, q, and pi_bar_P{k} for each parallelogram
  hysteresis.csv       the same, as a table

Usage:
  python hysteresis.py --wave-json wave_lane1.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qdm.config import (
    ANALYSIS_DIR,
    FT_PER_MILE,
    INFERENCE_DIR,
    K,
    SAMPLE_DT,
    SEC_PER_HR,
)
from qdm.io import align_by_time
from qdm.signals import log

# Frames of one trajectory more than this far apart are treated as separate
# passes through the region, not one continuous run.
MAX_FRAME_GAP = SAMPLE_DT * 1.5


def build_parallelograms(cfg):
    """One parallelogram per platoon point.

    u spans the local vehicle direction, w spans the wave direction. A point is
    inside when its (u, w) coordinates both lie in [-1, 1].
    """
    t_off = cfg["t_offset"]
    wave = cfg["wave"]
    pts = cfg["platoon"]

    wave_dxdt = (wave["x2"] - wave["x1"]) / (wave["t2"] - wave["t1"])
    log(f"[wave] speed {wave_dxdt:.2f} ft/s "
        f"({wave_dxdt * SEC_PER_HR / FT_PER_MILE:.1f} mph)")

    def local_vehicle_dxdt(i):
        """Vehicle speed at point i, by central difference over its neighbors."""
        if i == 0:
            a, b = pts[0], pts[1]
        elif i == len(pts) - 1:
            a, b = pts[-2], pts[-1]
        else:
            a, b = pts[i - 1], pts[i + 1]
        return (b["x"] - a["x"]) / (b["t"] - a["t"])

    out = []
    for i, p in enumerate(pts):
        v_dxdt = local_vehicle_dxdt(i)
        # Extent along the vehicle direction is fixed in space, so convert the
        # half-length in feet to a half-length in time using the local speed.
        dt_v = cfg["half_space_veh"] / max(abs(v_dxdt), 1.0)
        out.append({
            "label": p["label"],
            "center": np.array([t_off + p["t"], p["x"]]),
            "u": np.array([dt_v, v_dxdt * dt_v]),
            "w": np.array([cfg["half_time_wave"],
                           wave_dxdt * cfg["half_time_wave"]]),
            "v_dxdt": v_dxdt,
        })
    return out


def inside(t, x, par):
    """Solve [u w] c = p - center for c; inside iff |c| <= 1 on both axes."""
    p = np.column_stack([t - par["center"][0], x - par["center"][1]])
    M = np.column_stack([par["u"], par["w"]])
    c = np.linalg.solve(M, p.T).T
    return (np.abs(c[:, 0]) <= 1.0) & (np.abs(c[:, 1]) <= 1.0)


def edie_in_parallelogram(par, all_t, all_x, all_tid):
    """Frames inside the parallelogram, and Edie's k and q over it.

    Distance and time accumulate per continuous run of one trajectory, so a
    vehicle that leaves and re-enters does not contribute the jump between the
    two as travelled distance.
    """
    corners = np.array([
        par["center"] + par["u"] + par["w"],
        par["center"] + par["u"] - par["w"],
        par["center"] - par["u"] + par["w"],
        par["center"] - par["u"] - par["w"],
    ])
    t_lo, t_hi = corners[:, 0].min(), corners[:, 0].max()
    x_lo, x_hi = corners[:, 1].min(), corners[:, 1].max()

    bbox = np.nonzero((all_t >= t_lo) & (all_t <= t_hi)
                      & (all_x >= x_lo) & (all_x <= x_hi))[0]
    if len(bbox) == 0:
        return np.array([], dtype=int), 0.0, 0.0

    idx = bbox[inside(all_t[bbox], all_x[bbox], par)]
    if len(idx) == 0:
        return idx, 0.0, 0.0

    tid, t, x = all_tid[idx], all_t[idx], all_x[idx]
    order = np.lexsort((t, tid))
    tid, t, x = tid[order], t[order], x[order]

    total_d = total_t = 0.0
    i = 0
    while i < len(tid):
        j = i + 1
        while (j < len(tid) and tid[j] == tid[i]
               and (t[j] - t[j - 1]) <= MAX_FRAME_GAP):
            j += 1
        if j - i >= 2:
            total_t += t[j - 1] - t[i]
            total_d += abs(x[j - 1] - x[i])
        i = j

    # |A| = 4 |u x w|, since u and w are half-extents.
    area = 4.0 * abs(par["u"][0] * par["w"][1] - par["u"][1] * par["w"][0])
    if area <= 0:
        return idx, 0.0, 0.0

    return idx, total_t / area, total_d / area     # k veh/ft, q veh/s


def profile_mixture(par_idx, all_tid, tid_to_rows, pi_by_tid, time_by_tid,
                    all_t, K):
    """Mean pi over the inference frames inside one parallelogram."""
    if len(par_idx) == 0:
        return np.full(K, np.nan), 0

    total = np.zeros(K)
    n = 0
    for tid in np.unique(all_tid[par_idx]):
        if tid not in pi_by_tid:
            continue

        rows = tid_to_rows[tid]                     # this ego's rows in lane df
        frames = par_idx[all_tid[par_idx] == tid]   # its rows inside the region

        # Match by timestamp, not position, against the saved per-frame pi.
        local, saved = align_by_time(all_t[frames], time_by_tid[tid])
        if len(local) == 0:
            continue

        total += pi_by_tid[tid][saved].sum(axis=0)
        n += len(saved)

    return (total / n, n) if n > 0 else (np.full(K, np.nan), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave-json",
                    default=str(Path(__file__).parent / "wave_lane1.json"))
    ap.add_argument("--xt-dir", default=str(ANALYSIS_DIR / "xt"),
                    help="Directory holding lane{n}.parquet")
    ap.add_argument("--inference-dir", default=str(INFERENCE_DIR))
    ap.add_argument("--out", default=str(ANALYSIS_DIR / "hysteresis"))
    ap.add_argument("--K", type=int, default=K)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.wave_json) as f:
        cfg = json.load(f)
    lane = cfg["lane"]
    log(f"[cfg] {args.wave_json}, lane {lane}, "
        f"{len(cfg['platoon'])} platoon points")

    pars = build_parallelograms(cfg)

    lane_pq = Path(args.xt_dir) / f"lane{lane}.parquet"
    if not lane_pq.exists():
        raise FileNotFoundError(
            f"{lane_pq} not found. Run build_xt_diagrams.py first."
        )
    log(f"[load] {lane_pq}")
    df = pd.read_parquet(lane_pq)
    df = df.sort_values(["trajectory_id", "time"], kind="mergesort")
    df = df.reset_index(drop=True)

    all_t = df["time"].to_numpy()
    all_x = df["x"].to_numpy()
    all_tid = df["trajectory_id"].to_numpy()
    tid_to_rows = df.groupby("trajectory_id", sort=False).indices
    log(f"[load] {len(df):,} frames, {len(tid_to_rows):,} trajectories in lane {lane}")

    # Edie's k and q per parallelogram.
    par_idx = []
    for p in pars:
        idx, k, q = edie_in_parallelogram(p, all_t, all_x, all_tid)
        p["k_veh_per_ft"] = k
        p["q_veh_per_s"] = q
        p["n_frames"] = len(idx)
        par_idx.append(idx)

    # Load the saved pi only for the egos that actually appear in a region.
    inf_dir = Path(args.inference_dir)
    by_ego = inf_dir / "by_ego"
    catalog = pd.read_csv(inf_dir / "catalog.csv")
    inf_tids = set(catalog["trajectory_id"].astype(int).tolist())

    needed = set()
    for idx in par_idx:
        if len(idx):
            needed.update(np.unique(all_tid[idx]).tolist())
    needed &= inf_tids
    log(f"[inference] {len(needed):,} inference egos inside the parallelograms")

    pi_by_tid, time_by_tid = {}, {}
    for tid in needed:
        pi_path = by_ego / f"{tid}_pi.npy"
        t_path = by_ego / f"{tid}_time.npy"
        if pi_path.exists() and t_path.exists():
            pi_by_tid[tid] = np.load(pi_path)
            time_by_tid[tid] = np.load(t_path)

    rows = []
    for i, (p, idx) in enumerate(zip(pars, par_idx), start=1):
        pi_bar, n_pi = profile_mixture(idx, all_tid, tid_to_rows,
                                       pi_by_tid, time_by_tid, all_t, args.K)
        p["pi_bar"] = pi_bar

        row = {
            "point": i,
            "label": f"{i}. {p['label']}",
            "density_veh_per_mile": p["k_veh_per_ft"] * FT_PER_MILE,
            "flow_veh_per_hour": p["q_veh_per_s"] * SEC_PER_HR,
            "vehicle_speed_ft_per_s": abs(p["v_dxdt"]),
            "n_frames": p["n_frames"],
            "n_pi_frames": n_pi,
        }
        for k in range(args.K):
            row[f"pi_bar_P{k + 1}"] = pi_bar[k]
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "hysteresis.csv", index=False)

    np.savez_compressed(
        out_dir / "hysteresis_data.npz",
        labels=table["label"].to_numpy(),
        ks=table["density_veh_per_mile"].to_numpy(),
        qs=table["flow_veh_per_hour"].to_numpy(),
        **{f"p{k + 1}": table[f"pi_bar_P{k + 1}"].to_numpy()
           for k in range(args.K)},
    )

    print("\n" + table.to_string(index=False,
                                 float_format=lambda v: f"{v:.3f}"))
    log(f"\n[done] -> {out_dir / 'hysteresis.csv'}")
    log(f"[done] -> {out_dir / 'hysteresis_data.npz'}")


if __name__ == "__main__":
    main()
