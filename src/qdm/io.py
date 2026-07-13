"""Loading the processed parquet, imputation, and per-ego frame alignment."""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import BEHAVIORAL_COLS, CONTEXT_COLS
from .signals import log


def load_parquet(path, columns=None):
    """Read the processed parquet and impute the two columns that carry NaNs.

    headway       NaN means no same-lane leader within the search window, i.e.
                  open road. Imputed to the 99th percentile of observed
                  headways: a large but finite gap.
    accel_entropy NaN means the forward zone held no vehicles. No neighbors,
                  no disorder, so entropy is 0.

    speed, jerk, density and sp_entropy carry no NaNs by construction.

    The 99th percentile is computed over whatever rows are loaded, so always
    load the full file when the imputed value must match training.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/00_preprocess/ first."
        )

    log(f"[load] {path}")
    df = pd.read_parquet(path, columns=columns)
    log(f"[load] {len(df):,} rows, {df['trajectory_id'].nunique():,} egos")

    if "headway" in df.columns:
        h99 = df["headway"].quantile(0.99)
        n = int(df["headway"].isna().sum())
        df["headway"] = df["headway"].fillna(h99)
        log(f"[load] imputed {n:,} NaN headway -> 99th pct = {h99:.3f} ft")

    if "accel_entropy" in df.columns:
        n = int(df["accel_entropy"].isna().sum())
        df["accel_entropy"] = df["accel_entropy"].fillna(0.0)
        log(f"[load] imputed {n:,} NaN accel_entropy -> 0.0")

    return df


def behavioral_context_arrays(df):
    """(X_raw, C_raw, ids, times) in the parquet's row order."""
    return (
        df[BEHAVIORAL_COLS].to_numpy(dtype=np.float64),
        df[CONTEXT_COLS].to_numpy(dtype=np.float64),
        df["trajectory_id"].to_numpy(),
        df["time"].to_numpy(dtype=np.float64),
    )


def sample_egos(ids, times, n_egos, seed, arrays=()):
    """Subsample whole trajectories, then sort by (trajectory_id, time).

    Trajectories are kept whole, and frames are made contiguous, because the
    per-driver recursion in forward_chunk assumes it sees each ego's frames in
    time order. `arrays` are any extra per-row arrays to subset in lockstep.
    """
    unique = np.unique(ids)
    log(f"[sample] {len(unique):,} eligible egos in file")

    if len(unique) > n_egos:
        rng = np.random.default_rng(seed)
        chosen = set(rng.choice(unique, size=n_egos, replace=False).tolist())
        mask = np.fromiter((i in chosen for i in ids), dtype=bool, count=len(ids))
    else:
        log(f"[sample] file has <= {n_egos:,} egos; using all of them")
        mask = np.ones(len(ids), dtype=bool)

    ids, times = ids[mask], times[mask]
    order = np.lexsort((times, ids))
    arrays = tuple(a[mask][order] for a in arrays)
    ids, times = ids[order], times[order]

    log(f"[sample] {len(np.unique(ids)):,} egos, {len(ids):,} rows")
    return ids, times, arrays


def align_by_time(times_parquet, times_saved):
    """Map each saved per-frame value onto the parquet rows it came from.

    Inference writes one row per parquet frame, so the two normally match
    exactly. This resolves them by timestamp rather than by position, so a
    length mismatch surfaces as missing frames rather than a silent off-by-N
    shift. Returns (idx_parquet, idx_saved): parallel index arrays into the two
    inputs, covering only the timestamps present in both.
    """
    tp = np.round(np.asarray(times_parquet, dtype=np.float64), 2)
    ts = np.round(np.asarray(times_saved, dtype=np.float64), 2)

    order = np.argsort(ts, kind="mergesort")
    ts_sorted = ts[order]

    loc = np.searchsorted(ts_sorted, tp)
    ok = (loc < len(ts_sorted)) & (ts_sorted[loc.clip(max=len(ts_sorted) - 1)] == tp)

    idx_parquet = np.nonzero(ok)[0]
    idx_saved = order[loc[ok]]
    return idx_parquet, idx_saved
