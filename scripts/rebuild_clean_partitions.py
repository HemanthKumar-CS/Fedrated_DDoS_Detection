#!/usr/bin/env python3
"""Rebuild clean federated client partitions with:
1. Global train/test split first (stratified)
2. Row de-duplication before splitting
3. Leakage feature removal (perfect predictors / near-perfect correlation)
4. Balanced per-client stratified partitioning (no train/test overlap)

Usage:
  python scripts/rebuild_clean_partitions.py \
      --input data/optimized/balanced_dataset.csv \
      --output-dir data/optimized/clean_partitions \
      --num-clients 4 --test-size 0.2

After running, point clients to --data_dir data/optimized/clean_partitions
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json

TARGET_COL = "Binary_Label"
LABEL_COL = "Label"


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {path}")
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    # Deduplicate based on all feature columns (exclude possibly derived label col only?)
    df_dedup = df.drop_duplicates()
    removed = before - len(df_dedup)
    return df_dedup, removed


def find_leakage_features(df: pd.DataFrame, corr_threshold: float = 0.9999, max_unique_ratio: float = 0.9):
    """Identify obvious leakage features.
    Criteria:
      - Perfect mapping: each unique feature value maps to exactly one label (achieves 100% accuracy) AND at least 2 unique values.
      - Or absolute Pearson correlation > corr_threshold (for binary target) (only for numeric).
      - Skip features with extremely high cardinality (unique_ratio > max_unique_ratio) to avoid removing quasi-identifiers that just happen to separate.
    Returns list of feature names to drop.
    """
    leakage = []
    y = df[TARGET_COL].values
    n = len(df)
    feature_cols = [c for c in df.columns if c not in [TARGET_COL, LABEL_COL]]
    for col in feature_cols:
        series = df[col]
        # Skip if all values same
        if series.nunique() <= 1:
            continue
        unique_ratio = series.nunique() / n
        # Perfect mapping test
        mapping = {}
        perfect = True
        for xv, yv in zip(series.values, y):
            if xv in mapping and mapping[xv] != yv:
                perfect = False
                break
            mapping[xv] = yv
        if perfect and len(mapping) > 1 and unique_ratio <= max_unique_ratio:
            leakage.append(col)
            continue
        # Correlation test (numeric only)
        if np.issubdtype(series.dtype, np.number):
            try:
                corr = np.corrcoef(series.values, y)[0, 1]
                if not np.isnan(corr) and abs(corr) >= corr_threshold and unique_ratio <= max_unique_ratio:
                    leakage.append(col)
            except Exception:
                pass
    return sorted(list(set(leakage)))


def stratified_partition(df: pd.DataFrame, num_clients: int, target_col: str) -> list[pd.DataFrame]:
    """Split DataFrame into num_clients stratified by target distribution."""
    parts = [list() for _ in range(num_clients)]
    for label, group in df.groupby(target_col):
        idx = group.index.to_list()
        np.random.shuffle(idx)
        # Chunk indices roughly equally
        chunks = np.array_split(idx, num_clients)
        for cid, chunk in enumerate(chunks):
            if len(chunk) == 0:
                continue
            parts[cid].append(df.loc[chunk])
    return [pd.concat(p, axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True) if p else pd.DataFrame(columns=df.columns) for p in parts]


def save_client_partitions(train_parts, test_parts, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for cid, (tr, te) in enumerate(zip(train_parts, test_parts)):
        tr.to_csv(out_dir / f"client_{cid}_train.csv", index=False)
        te.to_csv(out_dir / f"client_{cid}_test.csv", index=False)
    # Save summary
    summary = {
        "train_sizes": {cid: int(len(tr)) for cid, tr in enumerate(train_parts)},
        "test_sizes": {cid: int(len(te)) for cid, te in enumerate(test_parts)},
    }
    with (out_dir / "partition_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Rebuild clean federated partitions with leakage mitigation")
    parser.add_argument("--input", type=Path, required=True, help="Input balanced dataset CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for new partitions")
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-unique-ratio", type=float, default=0.95, help="Skip removing features with unique ratio above this (likely identifiers)")
    parser.add_argument("--corr-threshold", type=float, default=0.9999)
    args = parser.parse_args()

    np.random.seed(args.seed)

    df = load_dataset(args.input)
    orig_rows = len(df)

    # De-duplicate
    df, removed_dups = deduplicate(df)

    # Identify leakage features
    leakage = find_leakage_features(df, corr_threshold=args.corr_threshold, max_unique_ratio=args.max_unique_ratio)
    df_clean = df.drop(columns=leakage) if leakage else df.copy()

    # Global stratified train/test split
    train_df, test_df = train_test_split(
        df_clean, test_size=args.test_size, random_state=args.seed, stratify=df_clean[TARGET_COL]
    )

    # Partition each split among clients
    train_parts = stratified_partition(train_df, args.num_clients, TARGET_COL)
    test_parts = stratified_partition(test_df, args.num_clients, TARGET_COL)

    # Assert no overlap
    train_hashes = set(train_df.apply(lambda r: hash(tuple(r.values.tolist())), axis=1))
    test_hashes = set(test_df.apply(lambda r: hash(tuple(r.values.tolist())), axis=1))
    overlap = len(train_hashes & test_hashes)

    # Save partitions
    save_client_partitions(train_parts, test_parts, args.output_dir)

    report = {
        "input_rows": orig_rows,
        "after_dedup_rows": int(len(df)),
        "duplicates_removed": int(removed_dups),
        "leakage_features_removed": leakage,
        "num_leakage_features_removed": len(leakage),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_test_overlap_rows": int(overlap),
        "train_label_dist": train_df[TARGET_COL].value_counts(normalize=True).to_dict(),
        "test_label_dist": test_df[TARGET_COL].value_counts(normalize=True).to_dict(),
    }
    with (args.output_dir / "clean_build_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== Clean Partition Build Report ===")
    for k, v in report.items():
        print(f"{k}: {v}")
    if leakage:
        print("Removed leakage features:")
        for feat in leakage:
            print(f"  - {feat}")
    else:
        print("No leakage features detected under provided criteria.")
    if overlap > 0:
        print("WARNING: Detected unexpected train/test overlap after splitting.")


if __name__ == "__main__":
    main()
