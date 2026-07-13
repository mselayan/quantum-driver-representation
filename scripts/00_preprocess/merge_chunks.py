#!/usr/bin/env python
"""
Concatenate the preprocessing chunks into the final dataset.

Upstream:
  scripts/00_preprocess/preprocess.py   -> data/processed/qdm_chunk_*.parquet
Downstream:
  scripts/02_train/train_qdm.py, and every analysis script

Each chunk already holds final-schema rows for its own disjoint set of egos, so
the merge is a concatenate. No recomputation, no duplicate trajectory_ids.

Aborts if a chunk is missing, rather than silently writing a partial dataset.

Usage:
  python merge_chunks.py --chunks data/processed --n-chunks 32
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qdm.config import FINAL_COLUMNS, PARQUET_NAME, PROCESSED_DIR
from qdm.signals import log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=str(PROCESSED_DIR),
                    help="Directory holding qdm_chunk_*.parquet")
    ap.add_argument("--n-chunks", type=int, required=True)
    ap.add_argument("--out", default=None,
                    help=f"Output path (default: <chunks>/{PARQUET_NAME})")
    args = ap.parse_args()

    chunk_dir = Path(args.chunks)
    out_path = Path(args.out) if args.out else chunk_dir / PARQUET_NAME

    log("=" * 70)
    log(f"MERGE {args.n_chunks} CHUNKS")
    log("=" * 70)

    parts, missing = [], []
    for cid in range(args.n_chunks):
        path = chunk_dir / f"qdm_chunk_{cid:04d}.parquet"
        if not path.exists():
            missing.append(cid)
            continue
        d = pd.read_parquet(path)
        parts.append(d)
        log(f"[chunk {cid:>4d}] {len(d):>10,} rows, "
            f"{d['trajectory_id'].nunique():>7,} egos")

    if missing:
        log(f"\n[abort] {len(missing)} chunk(s) missing: {missing}")
        log("[abort] re-run them before merging, or the dataset is incomplete")
        sys.exit(1)

    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["trajectory_id", "time"]).reset_index(drop=True)
    df = df[FINAL_COLUMNS]
    df.to_parquet(out_path, index=False)

    log(f"\n[done] {len(df):,} rows, {df['trajectory_id'].nunique():,} egos")
    log(f"[done] -> {out_path}")

    log("\nNaN rate by column (headway and accel_entropy are imputed at load):")
    for c in FINAL_COLUMNS[4:]:
        log(f"  {c:<14s} {df[c].isna().mean() * 100:5.2f}%")

    log("\nSummary:")
    log(df[FINAL_COLUMNS[4:]].describe().to_string())


if __name__ == "__main__":
    main()
