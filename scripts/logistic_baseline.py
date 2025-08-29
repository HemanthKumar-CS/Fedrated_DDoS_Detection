import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import numpy as np


def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    # Expect 'Binary_Label' as target, 'Label' (string) can be dropped.
    target_col = 'Binary_Label'
    if target_col not in df.columns:
        raise ValueError(f"Expected column '{target_col}' in dataset")
    y = df[target_col].astype(int)
    # Drop label columns and any wholly non-numeric columns
    drop_cols = [c for c in ['Label', 'Binary_Label'] if c in df.columns]
    X = df.drop(columns=drop_cols)
    # Ensure numeric (coerce errors)
    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            X[col] = pd.to_numeric(X[col], errors='coerce')
    # Fill any NaNs produced by coercion with column median
    X = X.fillna(X.median(numeric_only=True))
    return X, y


def train_and_evaluate(X, y, test_size=0.2, seed=42, use_class_weight=False, cv: int | None = None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    lr = LogisticRegression(
        max_iter=2000,
        n_jobs=-1 if hasattr(LogisticRegression(), 'n_jobs') else None,
        class_weight='balanced' if use_class_weight else None,
        solver='lbfgs'
    )

    pipe = Pipeline([
        ('scaler', StandardScaler(with_mean=True, with_std=True)),
        ('clf', lr),
    ])

    if cv and cv > 1:
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=None)
    else:
        cv_scores = None

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'report': classification_report(y_test, y_pred, zero_division=0),
        'cv_mean_accuracy': float(np.mean(cv_scores)) if cv_scores is not None else None,
        'cv_std_accuracy': float(np.std(cv_scores)) if cv_scores is not None else None,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Logistic Regression baseline on balanced dataset.')
    parser.add_argument('--data', type=Path, default=Path('data/optimized/balanced_dataset.csv'))
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--class-weight', action='store_true', help='Use class_weight="balanced".')
    parser.add_argument('--cv', type=int, default=0, help='Optional k-fold CV on training split (k>1).')
    args = parser.parse_args()

    X, y = load_data(args.data)
    metrics = train_and_evaluate(X, y, test_size=args.test_size, seed=args.seed, use_class_weight=args.class_weight, cv=args.cv if args.cv > 1 else None)

    print('\n=== Logistic Regression Baseline Results ===')
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Accuracy:        {metrics['accuracy']:.6f}")
    print(f"Precision:       {metrics['precision']:.6f}")
    print(f"Recall:          {metrics['recall']:.6f}")
    print(f"F1-score:        {metrics['f1']:.6f}")
    print(f"ROC-AUC:         {metrics['roc_auc']:.6f}")
    if metrics['cv_mean_accuracy'] is not None:
        print(f"CV Accuracy:     {metrics['cv_mean_accuracy']:.6f} ± {metrics['cv_std_accuracy']:.6f} (k={args.cv})")
    print('Confusion Matrix (TN, FP / FN, TP):')
    cm = metrics['confusion_matrix']
    print(f"{cm[0]}\n{cm[1]}")
    print('\nClassification Report:\n')
    print(metrics['report'])
    if metrics['accuracy'] >= 0.99:
        print('NOTE: Accuracy >= 0.99. Dataset likely linearly separable / trivial for baseline.')
    else:
        print('Baseline below 0.99. CNN may still add value.')


if __name__ == '__main__':
    main()
