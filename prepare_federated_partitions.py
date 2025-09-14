#!/usr/bin/env python3
"""
Generate federated client partitions from the realistic dataset.

This script creates per-client train/test CSVs like:
  data/optimized/clean_partitions/client_0_train.csv
  data/optimized/clean_partitions/client_0_test.csv
  ...

Usage (PowerShell):
  .\venv\Scripts\python.exe prepare_federated_partitions.py --num_clients 4 --test_size 0.2

Inputs:
  - data/optimized/realistic_train.csv (default)
Outputs:
  - data/optimized/clean_partitions/client_{i}_train.csv
  - data/optimized/clean_partitions/client_{i}_test.csv
  - data/optimized/clean_partitions/train_summary.txt
  - data/optimized/clean_partitions/test_summary.txt
  - data/optimized/clean_partitions/partition_summary.json
  - data/optimized/clean_partitions/clean_build_report.json
"""

from __future__ import annotations
from src.data.federated_split import FederatedDataDistributor  # type: ignore
import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict

import pandas as pd

# Ensure we can import from src/
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = THIS_DIR
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def ensure_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Binary_Label is present; derive from Label if needed."""
    if "Binary_Label" not in df.columns:
        if "Label" not in df.columns:
            raise ValueError(
                "Input data must have 'Label' or 'Binary_Label' column")
        df = df.copy()
        df["Binary_Label"] = (df["Label"].astype(str) != "BENIGN").astype(int)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare federated client partitions")
    parser.add_argument("--input_csv", type=str,
                        default=os.path.join("data", "optimized", "realistic_train.csv"))
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join("data", "optimized", "clean_partitions"))
    parser.add_argument("--num_clients", type=int, default=4)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--strategy", type=str, choices=["iid", "non_iid_label", "non_iid_attack"], default="non_iid_label",
                        help="How to distribute data across clients")
    parser.add_argument("--concentration", type=float, default=0.7,
                        help="Label concentration for non_iid_label (higher => more skew)")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    df = ensure_binary_label(df)

    # Some datasets may include extraneous index columns like 'Unnamed: 0'
    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed:")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Configure distributor
    dist = FederatedDataDistributor(
        num_clients=args.num_clients, random_state=args.random_state)

    # Create distribution
    if args.strategy == "iid":
        client_data: Dict[int, pd.DataFrame] = dist.create_iid_distribution(df)
    elif args.strategy == "non_iid_label":
        client_data = dist.create_non_iid_by_label(
            df, concentration=args.concentration)
    else:
        client_data = dist.create_non_iid_by_attack_type(df)

    # Split into train/test per client
    train_data, test_data = dist.split_train_test(
        client_data, test_size=args.test_size)

    # Save partitions
    dist.save_federated_data(train_data, args.output_dir, data_type="train")
    dist.save_federated_data(test_data, args.output_dir, data_type="test")

    # Write JSON summaries
    partition_summary = {
        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_csv": os.path.relpath(args.input_csv, REPO_ROOT),
        "output_dir": os.path.relpath(args.output_dir, REPO_ROOT),
        "num_clients": args.num_clients,
        "test_size": args.test_size,
        "strategy": args.strategy,
        "concentration": args.concentration if args.strategy == "non_iid_label" else None,
        "random_state": args.random_state,
        "client_train_sizes": {cid: int(len(df)) for cid, df in train_data.items()},
        "client_test_sizes": {cid: int(len(df)) for cid, df in test_data.items()},
    }
    with open(os.path.join(args.output_dir, "partition_summary.json"), "w", encoding="utf-8") as f:
        json.dump(partition_summary, f, indent=2)

    clean_build_report = {
        "status": "success",
        "message": "Federated partitions generated successfully.",
        "details": partition_summary,
    }
    with open(os.path.join(args.output_dir, "clean_build_report.json"), "w", encoding="utf-8") as f:
        json.dump(clean_build_report, f, indent=2)

    print("✅ Federated client partitions created in:", args.output_dir)


if __name__ == "__main__":
    main()
