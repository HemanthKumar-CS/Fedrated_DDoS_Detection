#!/usr/bin/env python3
"""
Build a unified optimized 30-feature federated dataset by selecting a shared feature list
and rewriting per-client CSVs to contain only those features + Binary_Label (+ Label if present).

Usage (PowerShell):
  python scripts/prepare_optimized_federated_dataset.py --input-dir data/optimized/clean_partitions --output-dir data/optimized/clean_partitions --clients 4 --k 30 --label-col Binary_Label

If --output-dir == --input-dir, files will be overwritten in-place (safe; we only drop/append columns).
This script also writes selected_features.json and an OPTIMIZATION_SUMMARY-like log line.
"""
import argparse, json, os, glob
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold


def select_top_k_features(df: pd.DataFrame, label_col: str, k: int) -> list[str]:
    feature_cols = [c for c in df.columns if c not in [label_col, "Label", "Binary_Label"]]
    X = df[feature_cols].select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df[label_col].astype(int).values

    # 1) Remove zero-variance
    if X.shape[1] == 0:
        return []
    vt = VarianceThreshold(threshold=0.0)
    X_vt = vt.fit_transform(X)
    kept_vt = X.columns[vt.get_support()].tolist()

    # 2) Remove highly correlated (> 0.95)
    X_corr = pd.DataFrame(X_vt, columns=kept_vt)
    if X_corr.shape[1] == 0:
        return []
    corr = X_corr.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    kept_corr = [c for c in kept_vt if c not in to_drop]
    X_corr = X_corr[kept_corr]

    # 3) Mutual information ranking
    if X_corr.shape[1] == 0:
        return []
    mi = mutual_info_classif(X_corr.values, y, discrete_features=False, random_state=42)
    order = np.argsort(mi)[::-1]
    top = [kept_corr[i] for i in order[:k]]
    return top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="data/optimized/clean_partitions")
    ap.add_argument("--output-dir", default="data/optimized/clean_partitions")
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--label-col", default="Binary_Label")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_paths = sorted(glob.glob(os.path.join(args.input_dir, "client_*_train.csv")))
    if not train_paths:
        raise FileNotFoundError(f"No client_*_train.csv in {args.input_dir}")

    # Build combined train set for feature selection
    frames = []
    for p in train_paths:
        df = pd.read_csv(p)
        if args.label_col not in df.columns and "Label" in df.columns:
            df[args.label_col] = (df["Label"] != "BENIGN").astype(int)
        if args.label_col not in df.columns:
            raise ValueError(f"Missing {args.label_col} in {p}")
        frames.append(df)
    df_all = pd.concat(frames, ignore_index=True)

    selected = select_top_k_features(df_all, args.label_col, args.k)
    with open(os.path.join(args.output_dir, "selected_features.json"), "w", encoding="utf-8") as f:
        json.dump({"features": selected, "k": args.k}, f, indent=2)
    print(f"Selected {len(selected)} features -> {os.path.join(args.output_dir, 'selected_features.json')}")

    # Rewrite per-client files
    for cid in range(args.clients):
        for split in ["train", "test"]:
            ip = os.path.join(args.input_dir, f"client_{cid}_{split}.csv")
            if not os.path.exists(ip):
                print(f"Skip missing {ip}")
                continue
            df = pd.read_csv(ip)
            # Keep label aside
            y = None
            if args.label_col in df.columns:
                y = df[args.label_col].astype(int).values
            elif "Label" in df.columns:
                y = (df["Label"] != "BENIGN").astype(int).values
                df[args.label_col] = y
            else:
                raise ValueError(f"Missing label column in {ip}")
            # Drop labels from features
            X = df.drop(columns=[c for c in ["Label", args.label_col] if c in df.columns])
            # Add missing selected features
            for m in selected:
                if m not in X.columns:
                    X[m] = 0.0
            # Enforce order and drop extras
            X = X[[c for c in selected]]
            out = X.copy()
            out[args.label_col] = y
            # Preserve original string Label if you want (optional)
            if "Label" in df.columns:
                out["Label"] = df["Label"]
            op = os.path.join(args.output_dir, f"client_{cid}_{split}.csv")
            out.to_csv(op, index=False)
            print(f"Wrote {op} with {len(selected)} features + {args.label_col}")


if __name__ == "__main__":
    main()
