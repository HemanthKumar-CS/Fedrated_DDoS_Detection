import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score


def jaccard_rows(df_a: pd.DataFrame, df_b: pd.DataFrame) -> float:
    if df_a.empty or df_b.empty:
        return 0.0
    # Create hash for each row (excluding order-dependent index)
    a_hash = df_a.apply(lambda r: hash(tuple(r.values.tolist())), axis=1)
    b_hash = df_b.apply(lambda r: hash(tuple(r.values.tolist())), axis=1)
    set_a, set_b = set(a_hash), set(b_hash)
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def feature_label_correlations(df: pd.DataFrame, target_col: str = 'Binary_Label'):
    numeric = df.select_dtypes(include=[np.number])
    if target_col not in numeric.columns:
        return {}
    corrs = {}
    y = numeric[target_col]
    for col in numeric.columns:
        if col == target_col:
            continue
        x = numeric[col]
        if x.nunique() <= 1:
            continue
        corr = np.corrcoef(x, y)[0, 1]
        if np.isnan(corr):
            continue
        corrs[col] = corr
    return corrs


def perfect_predictors(df: pd.DataFrame, target_col: str = 'Binary_Label'):
    predictors = []
    y = df[target_col].values
    for col in df.columns:
        if col in [target_col, 'Label']:
            continue
        x = df[col].values
        if len(np.unique(x)) <= 1:
            continue
        # Simple heuristic: if each unique x maps to single y and predicts perfectly
        mapping = {}
        perfect = True
        for xv, yv in zip(x, y):
            if xv in mapping and mapping[xv] != yv:
                perfect = False
                break
            mapping[xv] = yv
        if perfect and len(mapping) > 1:
            # Check actual accuracy
            pred = np.array([mapping[val] for val in x])
            if (pred == y).all():
                predictors.append(col)
    return predictors


def analyze_client(data_dir: Path, cid: int):
    train_path = data_dir / f"client_{cid}_train.csv"
    test_path = data_dir / f"client_{cid}_test.csv"
    if not train_path.exists() or not test_path.exists():
        return {"client": cid, "error": "missing files"}

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Overlap (excluding labels) row-wise
    feature_cols = [c for c in train_df.columns if c not in ['Label', 'Binary_Label']]
    train_feat = train_df[feature_cols]
    test_feat = test_df[feature_cols]
    overlap_ratio = jaccard_rows(train_feat, test_feat)

    # Correlations on train
    corr_map = feature_label_correlations(train_df[feature_cols + ['Binary_Label']])
    top_corr = sorted(((abs(v), k, v) for k, v in corr_map.items()), reverse=True)[:10]

    # Perfect predictors
    perfect = perfect_predictors(train_df[feature_cols + ['Binary_Label']])

    # Per-feature separability (mean diff / pooled std)
    separability = {}
    for col in feature_cols:
        grp = train_df.groupby('Binary_Label')[col]
        if len(grp) == 2:
            try:
                m0, m1 = grp.mean().values
                s0, s1 = grp.std().values
                pooled = np.sqrt(((s0**2) + (s1**2)) / 2)
                if pooled > 0:
                    separability[col] = abs(m0 - m1) / pooled
            except Exception:
                pass
    top_sep = sorted(((v, k) for k, v in separability.items()), reverse=True)[:10]

    return {
        "client": cid,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "overlap_jaccard": overlap_ratio,
        "top_abs_corr": [(name, float(corr_val)) for _, name, corr_val in top_corr],
        "perfect_predictors": perfect,
        "top_separable_features": [(name, float(score)) for score, name in top_sep],
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose federated client splits for leakage and feature issues")
    parser.add_argument('--data_dir', type=Path, default=Path('data/optimized'))
    parser.add_argument('--clients', type=int, default=4)
    args = parser.parse_args()

    reports = []
    for cid in range(args.clients):
        reports.append(analyze_client(args.data_dir, cid))

    for r in reports:
        print("\n=== Client", r.get('client'), '===')
        if 'error' in r:
            print('Error:', r['error'])
            continue
        print(f"Train/Test sizes: {r['train_size']} / {r['test_size']}")
        print(f"Train/Test feature Jaccard overlap: {r['overlap_jaccard']:.6f}")
        print("Top |corr| features (name, corr):")
        for name, val in r['top_abs_corr']:
            print(f"  {name}: {val:.4f}")
        print("Perfect predictors:", r['perfect_predictors'] if r['perfect_predictors'] else 'None')
        print("Top separability (feature, d):")
        for name, score in r['top_separable_features']:
            print(f"  {name}: {score:.4f}")

    # Summary flags
    high_overlap = [r['client'] for r in reports if r.get('overlap_jaccard', 0) > 0.01]
    if high_overlap:
        print(f"\nWARNING: Clients with suspicious train/test overlap (>1% Jaccard): {high_overlap}")
    else:
        print("\nNo significant train/test row overlap detected (by Jaccard hash).")

    strong_corr = {}
    for r in reports:
        for name, val in r.get('top_abs_corr', []):
            if abs(val) > 0.95:
                strong_corr.setdefault(r['client'], []).append((name, val))
    if strong_corr:
        print("\nWARNING: Strong (>0.95) correlations detected:")
        for cid, feats in strong_corr.items():
            for name, val in feats:
                print(f"  Client {cid}: {name} corr={val:.4f}")
    else:
        print("No extreme (>0.95) feature-label correlations found in top list.")

    perfect = {r['client']: r['perfect_predictors'] for r in reports if r.get('perfect_predictors')}
    if perfect:
        print("\nWARNING: Perfect predictors found:")
        for cid, feats in perfect.items():
            print(f"  Client {cid}: {feats}")
    else:
        print("No perfect predictors detected.")

if __name__ == '__main__':
    main()
