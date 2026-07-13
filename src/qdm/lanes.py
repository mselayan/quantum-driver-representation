"""Lane assignment from lateral position y (ft)."""

import numpy as np

from .config import LANE_EDGES


def assign_lane(y):
    """Scalar y -> lane 1..4, or None if outside the drivable bins."""
    if y < LANE_EDGES[0] or y >= LANE_EDGES[-1]:
        return None
    for ln in range(4):
        if LANE_EDGES[ln] <= y < LANE_EDGES[ln + 1]:
            return ln + 1
    return None


def assign_lane_arr(y):
    """Vectorized. Returns int array; 0 means no lane."""
    y = np.asarray(y, dtype=np.float64)
    lane = np.zeros(len(y), dtype=np.int64)
    for ln in range(4):
        m = (y >= LANE_EDGES[ln]) & (y < LANE_EDGES[ln + 1])
        lane[m] = ln + 1
    return lane


def lanes_to_scan(ego_lane):
    """3-lane forward window. Lanes 1,2 -> (1,2,3); lanes 3,4 -> (2,3,4)."""
    center = min(max(ego_lane, 2), 3)
    return (center - 1, center, center + 1)
