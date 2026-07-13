"""Numeric helpers shared by discovery and preprocessing."""

import numpy as np

from .config import ENTROPY_BINS


def log(msg):
    """Flushed print, so progress is visible when piped to a file."""
    print(msg, flush=True)


def normalize_angle(a):
    return a % 360.0


def normalize_angle_arr(a):
    return np.mod(a, 360.0)


def central_diff(arr, t):
    """d(arr)/dt by central differences; one-sided at the endpoints."""
    arr = np.asarray(arr, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    n = len(arr)
    if n < 2:
        return np.full(n, np.nan)
    out = np.empty(n, dtype=np.float64)
    out[1:-1] = (arr[2:] - arr[:-2]) / (t[2:] - t[:-2])
    out[0] = (arr[1] - arr[0]) / (t[1] - t[0])
    out[-1] = (arr[-1] - arr[-2]) / (t[-1] - t[-2])
    return out


def windowed_stat(arr, half_window, stat_fn):
    """stat_fn over a centered +/- half_window frame window, shrinking at edges."""
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    out = np.empty(n, dtype=np.float64)
    for k in range(n):
        lo = max(0, k - half_window)
        hi = min(n, k + half_window + 1)
        chunk = arr[lo:hi]
        out[k] = np.nan if len(chunk) < 2 else stat_fn(chunk)
    return out


def shannon_entropy(values, n_bins=ENTROPY_BINS):
    """Entropy in nats of a histogram of `values`. Zero for < 2 valid samples."""
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return 0.0
    values = values[~np.isnan(values)]
    if values.size < 2:
        return 0.0
    counts, _ = np.histogram(values, bins=n_bins)
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs)))
