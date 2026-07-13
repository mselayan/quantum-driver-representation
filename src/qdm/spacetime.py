"""
Spacetime index for neighbor lookup.

The index is built from EVERY westbound vehicle in the file: every class, every
duration. A leader or neighbor influences an ego regardless of its class or how
long it was tracked, so no surrounding vehicle is ever excluded. The ego filter
(HDV class, minimum duration) is applied separately when choosing which
trajectories to score, never when building this index.

Per timestep the index stores parallel arrays sorted by x, so an x-window around
any ego is a binary search rather than a scan over all vehicles at that instant.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import (
    HDV_CLASSES,
    LEADER_WINDOW,
    MAX_SEARCH_RADIUS,
    MIN_DURATION,
    TESLA_ZONES,
    WESTBOUND,
)
from .lanes import assign_lane, assign_lane_arr
from .signals import log, normalize_angle


REQUIRED_FIELDS = [
    "direction",
    "coarse_vehicle_class",
    "first_timestamp",
    "last_timestamp",
    "timestamp",
    "x_position",
    "y_position",
]


def _validate_schema(data, json_path):
    """Fail immediately, with a readable message, on the wrong file or format.

    Without this a wrong file surfaces as a KeyError somewhere deep in a loop,
    after the full JSON has already been parsed.
    """
    if not isinstance(data, list):
        raise ValueError(
            f"{json_path} holds a {type(data).__name__}, expected a list of "
            f"trajectory objects. This is probably not an I-24 MOTION raw "
            f"trajectory file."
        )
    if len(data) == 0:
        raise ValueError(f"{json_path} is empty.")

    missing = [f for f in REQUIRED_FIELDS if f not in data[0]]
    if missing:
        raise ValueError(
            f"{json_path} is missing required field(s): {missing}\n"
            f"Found instead: {sorted(data[0].keys())}\n"
            f"This code expects the I-24 MOTION raw trajectory JSON, one object "
            f"per vehicle, with 25 Hz timestamp / x_position / y_position arrays."
        )

    n = len(data[0]["timestamp"])
    for f in ("x_position", "y_position"):
        if len(data[0][f]) != n:
            raise ValueError(
                f"{json_path}: first trajectory has {n} timestamps but "
                f"{len(data[0][f])} {f} values. The per-frame arrays must align."
            )


def load_trajectories(json_path):
    """Load westbound trajectories and mark which ones are eligible egos.

    Returns (trajs, is_eligible). `trajs` is every westbound vehicle in the
    file; `is_eligible` is a boolean mask marking the subset that may be scored
    as an ego. Both arrays are index-aligned: position i in `is_eligible`
    refers to trajs[i], and that index is used as the trajectory_id throughout.

    See data/raw/README.md for where to get the file and what must be in it.
    """
    import json

    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(
            f"{json_path} not found.\n"
            f"The I-24 MOTION data is not distributed with this repository. "
            f"Register at https://i24motion.org, download a westbound trajectory "
            f"file, and place it here. See data/raw/README.md."
        )

    log(f"[load] {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    log(f"[load] {len(data):,} trajectories in file")

    _validate_schema(data, json_path)

    trajs = []
    is_eligible = []
    for traj in data:
        if traj.get("direction") != WESTBOUND:
            continue
        cls = traj.get("coarse_vehicle_class")
        dur = traj["last_timestamp"] - traj["first_timestamp"]
        trajs.append(traj)
        is_eligible.append(cls in HDV_CLASSES and dur >= MIN_DURATION)

    if len(trajs) == 0:
        raise ValueError(
            f"{json_path} holds no westbound trajectories "
            f"(direction == {WESTBOUND}). Check that this is the westbound file; "
            f"eastbound is direction == 1."
        )

    is_eligible = np.asarray(is_eligible, dtype=bool)
    log(f"[load] {len(trajs):,} westbound (all classes, all durations) -> index")
    log(f"[load] {int(is_eligible.sum()):,} eligible egos "
        f"(class in {HDV_CLASSES}, duration >= {MIN_DURATION}s)")

    if not is_eligible.any():
        raise ValueError(
            f"{json_path} holds westbound trajectories but none are eligible "
            f"egos. Expected coarse_vehicle_class in {HDV_CLASSES} and duration "
            f">= {MIN_DURATION}s."
        )

    return trajs, is_eligible


def build_spacetime_index(trajs_with_vars, speed_key="speed"):
    """Build the per-timestep x-sorted index and the per-trajectory lookup.

    trajs_with_vars: list of (traj_dict, vars_dict). vars_dict must contain
    `speed_key` as a per-frame array.

    Returns:
      time_index[t_key]        -> dict of x-sorted arrays: idx, x, y, v, h
      traj_lookup[traj_idx][t] -> (x, y, v, heading, frame_index)
    """
    log("[index] building spacetime index")
    raw = defaultdict(lambda: {"idx": [], "x": [], "y": [], "v": [], "h": []})
    traj_lookup = {}

    for traj_idx, (traj, vars_) in enumerate(trajs_with_vars):
        ts = np.asarray(traj["timestamp"], dtype=np.float64)
        xs = np.asarray(traj["x_position"], dtype=np.float64)
        ys = np.asarray(traj["y_position"], dtype=np.float64)
        v = vars_[speed_key]

        dx = np.diff(xs)
        dy = np.diff(ys)
        if len(dx) > 0:
            head = np.degrees(np.arctan2(dy, dx)) % 360.0
            head = np.concatenate([[head[0]], head])
        else:
            head = np.array([0.0])

        traj_lookup[traj_idx] = {}
        for k in range(len(ts)):
            t_key = round(ts[k], 2)
            traj_lookup[traj_idx][t_key] = (xs[k], ys[k], v[k], head[k], k)
            b = raw[t_key]
            b["idx"].append(traj_idx)
            b["x"].append(xs[k])
            b["y"].append(ys[k])
            b["v"].append(v[k])
            b["h"].append(head[k])

    time_index = {}
    for t_key, b in raw.items():
        x = np.asarray(b["x"], dtype=np.float64)
        order = np.argsort(x)
        time_index[t_key] = {
            "idx": np.asarray(b["idx"], dtype=np.int64)[order],
            "x": x[order],
            "y": np.asarray(b["y"], dtype=np.float64)[order],
            "v": np.asarray(b["v"], dtype=np.float64)[order],
            "h": np.asarray(b["h"], dtype=np.float64)[order],
        }

    log(f"[index] {len(time_index):,} unique timesteps")
    return time_index, traj_lookup


def neighbors_in_x_window(snapshot, ego_x, radius=MAX_SEARCH_RADIUS):
    """Slice a snapshot to [ego_x - radius, ego_x + radius] via binary search."""
    x = snapshot["x"]
    lo = np.searchsorted(x, ego_x - radius, side="left")
    hi = np.searchsorted(x, ego_x + radius, side="right")
    return {k: snapshot[k][lo:hi] for k in ("idx", "x", "y", "v", "h")}


def find_leader(ego_x, ego_y, snapshot, ego_idx):
    """Nearest same-lane vehicle ahead (lower x, westbound) within LEADER_WINDOW.

    Returns (leader_idx, leader_speed, gap) or None. `gap` is the front-bumper
    to front-bumper distance; subtract the ego length to get the net headway.
    """
    ego_lane = assign_lane(ego_y)
    if ego_lane is None:
        return None

    x = snapshot["x"]
    hi = np.searchsorted(x, ego_x, side="left")     # x < ego_x is ahead
    if hi == 0:
        return None
    lo = np.searchsorted(x, ego_x - LEADER_WINDOW, side="left")

    cidx = snapshot["idx"][lo:hi]
    cx = snapshot["x"][lo:hi]
    cy = snapshot["y"][lo:hi]
    cv = snapshot["v"][lo:hi]

    mask = (assign_lane_arr(cy) == ego_lane) & (cidx != ego_idx)
    if not np.any(mask):
        return None

    sx, sv, sidx = cx[mask], cv[mask], cidx[mask]
    gap = ego_x - sx                                # positive by construction
    j = int(np.argmin(gap))
    return int(sidx[j]), float(sv[j]), float(gap[j])


def in_tesla_zone_vec(rel_angles, dists, ego_heading):
    """Boolean mask: does each (angle, distance) fall in any Tesla zone?"""
    in_any = np.zeros(len(dists), dtype=bool)
    for z in TESLA_ZONES.values():
        within_r = dists <= z["radius"]
        if not np.any(within_r):
            continue
        from_rot = normalize_angle(z["from"] + ego_heading)
        to_rot = normalize_angle(z["to"] + ego_heading)
        if from_rot > to_rot:                       # zone wraps past 360
            ang_ok = (rel_angles >= from_rot) | (rel_angles <= to_rot)
        else:
            ang_ok = (rel_angles >= from_rot) & (rel_angles <= to_rot)
        in_any |= within_r & ang_ok
    return in_any


def tesla_neighbor_union(time_index, ego_x, ego_y, ego_h, ego_idx,
                         t_key, past_offsets, radius=MAX_SEARCH_RADIUS):
    """Set of trajectory indices that entered any Tesla zone over the past 1 s.

    Zones are evaluated against the ego's position at t_key but the neighbor
    positions at each past offset, matching the original implementation.
    """
    neighbor_set = set()
    for dt in past_offsets:
        snap_p = time_index.get(round(t_key - dt, 2))
        if snap_p is None:
            continue
        win = neighbors_in_x_window(snap_p, ego_x, radius)
        p_idx, p_x, p_y = win["idx"], win["x"], win["y"]
        keep = p_idx != ego_idx
        p_idx, p_x, p_y = p_idx[keep], p_x[keep], p_y[keep]
        if len(p_idx) == 0:
            continue
        pdx = p_x - ego_x
        pdy = p_y - ego_y
        pdist = np.sqrt(pdx * pdx + pdy * pdy)
        prel = np.mod(np.degrees(np.arctan2(pdy, pdx)), 360.0)
        for ni in p_idx[in_tesla_zone_vec(prel, pdist, ego_h)]:
            neighbor_set.add(int(ni))
    return neighbor_set
