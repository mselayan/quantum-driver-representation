"""
Central constants for the Q-DM I-24 pipeline.

Every script imports from here. Values that a user may want to change per run
are exposed as CLI flags in the scripts; the values below are the defaults used
for the results in the paper.
"""

from pathlib import Path

# ------------------------------------------------------------------
# Paths (relative to repo root; override with --data-root on any script)
# ------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"

RAW_DIR       = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
MODEL_DIR     = DATA_ROOT / "models"
INFERENCE_DIR = DATA_ROOT / "inference"
ANALYSIS_DIR  = DATA_ROOT / "analysis"

PARQUET_NAME = "full_qdm.parquet"


# ------------------------------------------------------------------
# Dataset filters (I-24 MOTION westbound, full recording)
# ------------------------------------------------------------------

# The paper's results come from one day of the I-24 MOTION INCEPTION release:
#   22 Nov 2022 (Tue), 06:00, 4 hours
#   collection 637c399add50d54aa5af0cf4__post2
# See data/raw/README.md. The code does not depend on the filename.
SOURCE_COLLECTION = "637c399add50d54aa5af0cf4__post2"

WESTBOUND    = -1
HDV_CLASSES  = [0, 1, 2, 3]   # sedan, midsize, van, pickup
MIN_DURATION = 10.0           # s; ego eligibility only

# Warm-up: the first 26 frames (~1 s at 25 Hz) of each ego are dropped so that
# windowed and entropy quantities have history.
WARMUP_FRAMES = 26
WARMUP_SECS   = 1.0


# ------------------------------------------------------------------
# Geometry
# ------------------------------------------------------------------

FT_PER_M = 3.28084

# Lane bins. y in [9,24)->1, [24,36)->2, [36,48)->3, [48,63)->4, else no lane.
LANE_EDGES   = [9.0, 24.0, 36.0, 48.0, 63.0]
LANE_CENTERS = [18.0, 30.0, 42.0, 54.0]     # ft, lanes 1..4

# Westbound: "ahead" is LOWER x.
LEADER_WINDOW    = 500.0                # ft, leader search back-window
FORWARD_LON_DIST = 150.0 * FT_PER_M     # ft, forward zone depth (492 ft)
OMNI_RADIUS      = 150.0 * FT_PER_M     # ft, density radius (492 ft)

# Binary-search half-window around the ego. Note this truncates the largest
# Tesla zone (Narrow Fwd, 250 m = 820 ft); neighbors beyond 520 ft do not
# contribute to sp_entropy. This matches the configuration used for the paper.
MAX_SEARCH_RADIUS = 520.0


# ------------------------------------------------------------------
# Tesla perception zones (radii in m, converted to ft; angles in degrees)
# ------------------------------------------------------------------

TESLA_ZONES = {
    "Wide Fwd":    {"radius":  60 * FT_PER_M, "from": 300.0, "to":  60.0},
    "Main Fwd":    {"radius": 150 * FT_PER_M, "from": 337.5, "to":  22.5},
    "Narrow Fwd":  {"radius": 250 * FT_PER_M, "from": 342.5, "to":  17.5},
    "Side Fwd L":  {"radius":  80 * FT_PER_M, "from":  25.0, "to": 115.0},
    "Side Fwd R":  {"radius":  80 * FT_PER_M, "from": 245.0, "to": 335.0},
    "Rear":        {"radius":  50 * FT_PER_M, "from": 112.5, "to": 247.5},
    "Side Rear L": {"radius": 100 * FT_PER_M, "from": 107.5, "to": 182.5},
    "Side Rear R": {"radius": 100 * FT_PER_M, "from": 177.5, "to": 252.5},
}


# ------------------------------------------------------------------
# Signal processing
# ------------------------------------------------------------------

SAMPLE_HZ    = 25.0
SAMPLE_DT    = 1.0 / SAMPLE_HZ
ENTROPY_BINS = 5
WIN_1S       = 12    # +/- frames, ~1 s
WIN_5S       = 62    # +/- frames, ~5 s
MIN_SPEED    = 0.1   # ft/s, divide-by-zero guard
TTC_CLIP     = 100.0 # s

# Past-snapshot offsets for the 1 s Tesla-zone neighbor union.
PAST_OFFSETS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

BEHAVIORAL_COLS = ["speed", "headway", "jerk"]
CONTEXT_COLS    = ["density", "sp_entropy", "accel_entropy"]

FINAL_COLUMNS = [
    "trajectory_id", "time", "x", "y",
    "speed", "headway", "jerk",
    "density", "sp_entropy", "accel_entropy",
]

K          = 3       # number of behavioral profiles
D          = 100     # RFF feature dimension
RANK       = 10      # max rank of each profile
ALPHA      = 0.2     # FIXED state-evolution blend
ETA        = 0.1     # FIXED behavioral adaptation (see model.py: no gradient
                     # reaches eta, so it cannot be learned)
GAMMA      = 4.0     # von Neumann entropy penalty weight (multi-rank prior)
RFF_GAMMA  = 1.0     # RBF kernel bandwidth for the RFF map
RFF_SEED   = 42      # RBFSampler random_state

SEED       = 702
EPOCHS     = 5
CHUNK_SIZE = 5000
LR         = 0.005
N_TRAIN_EGOS = 100_000


# ------------------------------------------------------------------
# Fundamental diagram / hysteresis
# ------------------------------------------------------------------

FD_DX_FT  = 1000.0
FD_DT_SEC = 30.0

FT_PER_MILE = 5280.0
SEC_PER_HR  = 3600.0
MPH_PER_FPS = 0.681818
