#!/usr/bin/env python
"""
Rank candidate behavioral and context variables to select the modeled set.

Upstream:
  data/raw/<dataset>.json
Downstream:
  none. This justifies the six variables that scripts/00_preprocess computes.

Computes 13 behavioral and 9 context candidates on a sample of eligible egos and
ranks each on four criteria:

  within_context_spread      heterogeneity survives inside a single context bin
  between_context_shift      the distribution moves when the context moves
  temporal_evolution         the variable evolves within a trajectory, rather
                             than being either white noise or a constant
  cross_driver_consistency   drivers are separable from one another

A variable earns a place in the model by scoring on all four: it must vary
between drivers in the same situation (1), respond to the situation (2), carry
within-trajectory structure for the state recursion to track (3), and be stable
enough within a driver to be a driver property rather than noise (4).

Composite score is the arithmetic mean of the four. Pairwise correlations flag
redundancy.

The spacetime index holds every westbound vehicle. Only the SCORED egos are
sampled, so context variables see the true neighborhood density instead of a
thinned one.

Usage:
  python variable_discovery.py --json data/raw/i24.json --out data/analysis/discovery
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qdm.config import (
    ANALYSIS_DIR,
    FORWARD_LON_DIST,
    LANE_CENTERS,
    MAX_SEARCH_RADIUS,
    MIN_SPEED,
    OMNI_RADIUS,
    PAST_OFFSETS,
    TTC_CLIP,
    WIN_1S,
    WIN_5S,
)
from qdm.lanes import assign_lane, assign_lane_arr, lanes_to_scan
from qdm.signals import central_diff, log, shannon_entropy, windowed_stat
from qdm.spacetime import (
    build_spacetime_index,
    find_leader,
    load_trajectories,
    neighbors_in_x_window,
    tesla_neighbor_union,
)

N_SAMPLE = 10_000
SEED = 42

# Context bins for criteria 1 and 2. Quantile bins on a continuous context
# variable; C1 is integer-valued and degenerates to too few bins on its own.
N_REGIMES = 4
BINNING_PRIORITY = ["C3_fwd_mean_speed", "C5_speed_gap", "C1_omni_density"]

BEHAVIORAL = [
    "B1_speed", "B2_accel", "B3_jerk", "B4_lat_velocity", "B5_lat_accel",
    "B6_rel_speed", "B7_headway", "B8_ttc", "B9_headway_time",
    "B10_accel_std_1s", "B11_speed_std_5s", "B12_y_std_1s", "B13_lane_dev",
]
CONTEXT = [
    "C1_omni_density", "C2_fwd_density", "C3_fwd_mean_speed",
    "C4_fwd_min_speed", "C5_speed_gap", "C6_speed_entropy",
    "C7_lateral_entropy", "C8_neighbor_accel_entropy", "C9_lane",
]
VAR_NAMES = BEHAVIORAL + CONTEXT

# The set that survived ranking and redundancy screening.
SELECTED = [
    "B1_speed", "B7_headway", "B3_jerk",
    "C1_omni_density", "C6_speed_entropy", "C8_neighbor_accel_entropy",
]


# ============================================================
# Candidate computation
# ============================================================

def compute_behavioral(traj):
    """B1-B5 and B10-B13. B6-B9 need a leader and are filled in later.

    Computed for EVERY vehicle: neighbors need B1 for the index and B2 for C8.
    """
    t = np.asarray(traj["timestamp"], dtype=np.float64)
    x = np.asarray(traj["x_position"], dtype=np.float64)
    y = np.asarray(traj["y_position"], dtype=np.float64)
    n = len(t)
    out = {}

    speed = np.abs(central_diff(x, t))
    accel = central_diff(speed, t)
    y_dot = central_diff(y, t)

    out["B1_speed"] = speed
    out["B2_accel"] = accel
    out["B3_jerk"] = central_diff(accel, t)
    out["B4_lat_velocity"] = y_dot
    out["B5_lat_accel"] = central_diff(y_dot, t)

    # B13: absolute deviation from the assigned lane's center. NaN off-road.
    lane = assign_lane_arr(y)
    centers = np.concatenate([[np.nan], LANE_CENTERS])   # index 0 = no lane
    out["B13_lane_dev"] = np.abs(y - np.where(lane > 0, centers[lane], np.nan))

    out["B10_accel_std_1s"] = windowed_stat(accel, WIN_1S, np.nanstd)
    out["B11_speed_std_5s"] = windowed_stat(speed, WIN_5S, np.nanstd)
    out["B12_y_std_1s"] = windowed_stat(y, WIN_1S, np.nanstd)

    for c in ("B6_rel_speed", "B7_headway", "B8_ttc", "B9_headway_time"):
        out[c] = np.full(n, np.nan)

    return out


def fill_leader_vars(trajs_with_vars, time_index, targets):
    """B6-B9 for the sampled egos. Leader lookup uses the full index."""
    log(f"[vars] leader-dependent B6-B9 for {len(targets):,} egos")

    for ego_idx in targets:
        traj, vars_ = trajs_with_vars[ego_idx]
        ts, xs, ys = traj["timestamp"], traj["x_position"], traj["y_position"]
        ego_length = traj.get("length", 5.0)
        speed = vars_["B1_speed"]

        for k in range(len(ts)):
            snap = time_index.get(round(ts[k], 2))
            if snap is None:
                continue
            leader = find_leader(xs[k], ys[k], snap, ego_idx)
            if leader is None:
                continue
            _, v_leader, gap = leader

            h = gap - ego_length
            dv = speed[k] - v_leader          # positive means closing
            vars_["B7_headway"][k] = h
            vars_["B6_rel_speed"][k] = dv
            vars_["B8_ttc"][k] = min(h / dv, TTC_CLIP) if (dv > 0 and h > 0) else TTC_CLIP
            if speed[k] > MIN_SPEED:
                vars_["B9_headway_time"][k] = h / speed[k]


def compute_context(trajs_with_vars, time_index, traj_lookup, targets):
    """C1-C9 for the sampled egos. Neighbor lookup uses the full index."""
    log(f"[vars] context C1-C9 for {len(targets):,} egos")

    for ego_idx in targets:
        traj, vars_ = trajs_with_vars[ego_idx]
        n = len(traj["timestamp"])
        for c in CONTEXT[:-1]:
            vars_[c] = np.full(n, np.nan)

        # C9: the ego's own lane. No neighbors needed.
        lane = assign_lane_arr(np.asarray(traj["y_position"], dtype=np.float64))
        c9 = lane.astype(np.float64)
        c9[lane == 0] = np.nan
        vars_["C9_lane"] = c9

    every = max(1, len(targets) // 20)
    for n_done, ego_idx in enumerate(targets, start=1):
        traj, vars_ = trajs_with_vars[ego_idx]
        ts, xs, ys = traj["timestamp"], traj["x_position"], traj["y_position"]

        for k in range(len(ts)):
            t_key = round(ts[k], 2)
            snap = time_index.get(t_key)
            if snap is None or t_key not in traj_lookup[ego_idx]:
                continue

            ego_x, ego_y = xs[k], ys[k]
            ego_v = vars_["B1_speed"][k]
            ego_h = traj_lookup[ego_idx][t_key][3]

            win = neighbors_in_x_window(snap, ego_x, MAX_SEARCH_RADIUS)
            nidx, nx, ny, nv = win["idx"], win["x"], win["y"], win["v"]
            keep = nidx != ego_idx
            nidx, nx, ny, nv = nidx[keep], nx[keep], ny[keep], nv[keep]

            if len(nidx) == 0:
                vars_["C1_omni_density"][k] = 0
                vars_["C2_fwd_density"][k] = 0
                continue

            dxn = nx - ego_x
            dyn = ny - ego_y
            dist = np.sqrt(dxn * dxn + dyn * dyn)

            # -- C1: omni density --
            vars_["C1_omni_density"][k] = int(np.sum(dist <= OMNI_RADIUS))

            # -- C2-C5: forward zone, ego's 3-lane window, no heading cone --
            ego_lane = assign_lane(ego_y)
            if ego_lane is None:
                vars_["C2_fwd_density"][k] = 0
                fwd_idx = np.array([], dtype=np.int64)
            else:
                scan = lanes_to_scan(ego_lane)
                ahead = (nx < ego_x) & ((ego_x - nx) <= FORWARD_LON_DIST)
                fwd = ahead & np.isin(assign_lane_arr(ny), scan)
                fwd_v, fwd_idx = nv[fwd], nidx[fwd]

                vars_["C2_fwd_density"][k] = int(len(fwd_v))
                if len(fwd_v) > 0:
                    mean_v = float(fwd_v.mean())
                    vars_["C3_fwd_mean_speed"][k] = mean_v
                    vars_["C4_fwd_min_speed"][k] = float(fwd_v.min())
                    vars_["C5_speed_gap"][k] = mean_v - ego_v

            # -- C8: accel entropy of the forward zone --
            if len(fwd_idx) > 0:
                accels = []
                for ni in fwd_idx:
                    lk = traj_lookup.get(int(ni))
                    if lk is None or t_key not in lk:
                        continue
                    a_n = trajs_with_vars[int(ni)][1]["B2_accel"][lk[t_key][4]]
                    if not np.isnan(a_n):
                        accels.append(a_n)
                vars_["C8_neighbor_accel_entropy"][k] = shannon_entropy(np.asarray(accels))

            # -- C6, C7: Tesla-zone neighbors over the past 1 s --
            neighbor_set = tesla_neighbor_union(
                time_index, ego_x, ego_y, ego_h, ego_idx, t_key, PAST_OFFSETS
            )
            t_past = round(t_key - 1.0, 2)
            speed_changes, lateral_changes = [], []
            for ni in neighbor_set:
                lk = traj_lookup.get(ni)
                if lk is None or t_key not in lk or t_past not in lk:
                    continue
                _, y_now, v_now, _, _ = lk[t_key]
                _, y_past, v_past, _, _ = lk[t_past]
                speed_changes.append(v_now - v_past)
                lateral_changes.append(abs(y_now - y_past))

            vars_["C6_speed_entropy"][k] = shannon_entropy(np.asarray(speed_changes))
            vars_["C7_lateral_entropy"][k] = shannon_entropy(np.asarray(lateral_changes))

        if n_done % every == 0 or n_done == len(targets):
            log(f"[vars] context {n_done:,}/{len(targets):,} egos")


def assemble(trajs_with_vars, targets):
    """One row per (ego, frame), all 22 candidates as columns."""
    log("[df] assembling")
    frames = []
    for ego_idx in targets:
        traj, vars_ = trajs_with_vars[ego_idx]
        n = len(traj["timestamp"])
        d = {"traj_id": np.full(n, ego_idx),
             "t": np.asarray(traj["timestamp"], dtype=np.float64)}
        d.update({v: vars_[v] for v in VAR_NAMES})
        frames.append(pd.DataFrame(d))

    df = pd.concat(frames, ignore_index=True)
    log(f"[df] {len(df):,} observations")
    return df


# ============================================================
# Ranking criteria
# ============================================================

def pick_binning_context(df):
    for c in BINNING_PRIORITY:
        valid = df[c].dropna()
        if len(valid) >= 1000 and valid.nunique() >= N_REGIMES * 2:
            return c
    raise RuntimeError("no context variable has enough distinct values to bin")


def make_regimes(df, ctx_var):
    valid = df[ctx_var].dropna()
    edges = np.unique(np.quantile(valid, np.linspace(0, 1, N_REGIMES + 1)))
    if len(edges) < N_REGIMES + 1:
        edges = np.linspace(valid.min(), valid.max() + 1e-9, N_REGIMES + 1)
    df = df.copy()
    df["_regime"] = pd.cut(df[ctx_var], bins=edges, labels=False,
                           include_lowest=True, duplicates="drop")
    return df, edges


def crit_within_context_spread(df, var):
    """Mean within-regime std / overall std. High means the variable still
    spreads out among drivers who are all in the same situation."""
    sub = df[[var, "_regime"]].dropna()
    if len(sub) < 200:
        return np.nan, f"{len(sub)} valid rows"
    overall = sub[var].std()
    if not overall > 0:
        return np.nan, "overall std is 0"
    within = sub.groupby("_regime")[var].std()
    if within.isna().all():
        return np.nan, "all regime stds NaN"
    return float(within.mean() / overall), None


def crit_between_context_shift(df, var):
    """Std of regime means / overall std. High means the variable tracks the
    context rather than ignoring it."""
    sub = df[[var, "_regime"]].dropna()
    if len(sub) < 200:
        return np.nan, f"{len(sub)} valid rows"
    overall = sub[var].std()
    if not overall > 0:
        return np.nan, "overall std is 0"
    means = sub.groupby("_regime")[var].mean()
    if len(means) < 2:
        return np.nan, "fewer than 2 regimes"
    return float(means.std() / overall), None


def crit_temporal_evolution(df, var):
    """Rewards lag-1 correlation (not white noise), penalizes long-lag
    correlation (not a constant), and requires the short-window std to be a real
    fraction of the total (it actually moves)."""
    scores, skipped = [], 0
    for _, group in df.groupby("traj_id"):
        x = group[var].to_numpy()
        x = x[~np.isnan(x)]
        if len(x) < 100:
            skipped += 1
            continue
        lag_long = min(100, len(x) // 2)
        r_short = np.corrcoef(x[:-1], x[1:])[0, 1]
        r_long = np.corrcoef(x[:-lag_long], x[lag_long:])[0, 1]
        if np.isnan(r_short) or np.isnan(r_long):
            skipped += 1
            continue
        std_short = np.std(x[:WIN_1S * 2]) if len(x) >= WIN_1S * 2 else np.std(x)
        std_long = np.std(x)
        ratio = std_short / std_long if std_long > 0 else 0.0
        scores.append(max(0, r_short)
                      * max(0, 1 - r_long)
                      * max(0, min(1, ratio)))
    if not scores:
        return np.nan, f"{skipped} trajectories skipped"
    return float(np.mean(scores)), None


def crit_cross_driver_consistency(df, var):
    """Between-driver variance / total variance, an intraclass correlation.
    High means a driver's value is a property of that driver."""
    sub = df[[var, "traj_id"]].dropna()
    if len(sub) < 1000:
        return np.nan, f"{len(sub)} valid rows"
    grouped = sub.groupby("traj_id")[var]
    between = grouped.mean().var()
    stds = grouped.std()
    within = (stds ** 2).mean() if not stds.empty else 0.0
    total = between + within
    if not total > 0 or np.isnan(total):
        return np.nan, "total variance is 0 or NaN"
    return float(between / total), None


def composite(values):
    """Arithmetic mean of the criteria that returned a number. Needs at least 3
    of 4: a single NaN axis should not zero out an otherwise strong variable."""
    valid = [v for v in values if v is not None and not np.isnan(v)]
    return float(np.mean(valid)) if len(valid) >= 3 else np.nan


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default=str(ANALYSIS_DIR / "discovery"))
    ap.add_argument("--n-sample", type=int, default=N_SAMPLE,
                    help="Egos to score. The index always holds every vehicle.")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log("VARIABLE DISCOVERY")
    log("=" * 70)

    trajs, is_eligible = load_trajectories(args.json)

    eligible = np.nonzero(is_eligible)[0]
    rng = np.random.default_rng(args.seed)
    if len(eligible) <= args.n_sample:
        targets = sorted(eligible.tolist())
    else:
        targets = sorted(rng.choice(eligible, size=args.n_sample,
                                    replace=False).tolist())
    log(f"[sample] scoring {len(targets):,} of {len(eligible):,} eligible egos")

    log(f"[vars] behavioral B1-B5, B10-B13 for all {len(trajs):,} vehicles")
    trajs_with_vars = [(t, compute_behavioral(t)) for t in trajs]

    time_index, traj_lookup = build_spacetime_index(
        trajs_with_vars, speed_key="B1_speed"
    )

    fill_leader_vars(trajs_with_vars, time_index, targets)
    compute_context(trajs_with_vars, time_index, traj_lookup, targets)

    df = assemble(trajs_with_vars, targets)
    df.to_parquet(out_dir / "candidates.parquet", index=False)
    log(f"[save] {out_dir / 'candidates.parquet'}")

    ctx_var = pick_binning_context(df)
    df, edges = make_regimes(df, ctx_var)
    log(f"[bin] regimes from {ctx_var}, edges {np.round(edges, 2)}")

    log("[rank] scoring")
    rows = []
    for var in VAR_NAMES:
        c1, n1 = crit_within_context_spread(df, var)
        c2, n2 = crit_between_context_shift(df, var)
        c3, n3 = crit_temporal_evolution(df, var)
        c4, n4 = crit_cross_driver_consistency(df, var)
        rows.append({
            "variable": var,
            "type": "behavioral" if var.startswith("B") else "context",
            "within_context_spread": c1,
            "between_context_shift": c2,
            "temporal_evolution": c3,
            "cross_driver_consistency": c4,
            "score": composite([c1, c2, c3, c4]),
            "nan_rate": df[var].isna().mean(),
            "selected": var in SELECTED,
        })
        for note in (n for n in (n1, n2, n3, n4) if n):
            log(f"[rank] {var}: {note}")

    ranking = pd.DataFrame(rows).sort_values("score", ascending=False)
    ranking.to_csv(out_dir / "variable_ranking.csv", index=False)

    print("\n" + "=" * 100)
    print("RANKING")
    print("=" * 100)
    print(ranking.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Redundancy. Pairwise-complete, so a high-NaN variable such as headway does
    # not shrink the sample for every other pair.
    corr = df[VAR_NAMES].corr().abs()
    corr.to_csv(out_dir / "correlation.csv")

    print("\n" + "=" * 100)
    print("REDUNDANT PAIRS  |r| > 0.85")
    print("=" * 100)
    flagged = [
        (a, b, corr.loc[a, b])
        for i, a in enumerate(VAR_NAMES)
        for b in VAR_NAMES[i + 1:]
        if corr.loc[a, b] > 0.85
    ]
    for a, b, r in flagged:
        print(f"  {a:<28s} {b:<28s} |r| = {r:.3f}")
    if not flagged:
        print("  none")

    print("\n" + "=" * 100)
    print("SELECTED SET, pairwise |r|")
    print("=" * 100)
    print(corr.loc[SELECTED, SELECTED].to_string(
        float_format=lambda v: f"{v:.3f}"))

    log("[done]")


if __name__ == "__main__":
    main()
