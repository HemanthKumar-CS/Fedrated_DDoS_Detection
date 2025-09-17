#!/usr/bin/env python3
"""Centralized baseline training using cleaned federated partitions.

Aggregates all client train partitions into one global training set and all
client test partitions into one global test set (no leakage due to global
first split in clean_partitions). Trains the existing CNN binary classifier
and reports metrics.

Usage:
  python train_centralized.py \
	  --data_dir data/optimized/clean_partitions \
	  --epochs 25 --batch 64 --lr 0.001

Outputs:
	- Saves model to results/centralized_model.keras (default)
	- Writes metrics JSON to results/centralized_training_results.json
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import logging
from src.models.trainer import ModelTrainer  # type: ignore

# Import visualization
try:
    from src.visualization.training_visualizer import generate_training_visualizations
    VISUALIZATION_AVAILABLE = True
except ImportError:
    print("Warning: Visualization module not available")
    VISUALIZATION_AVAILABLE = False

logging.basicConfig(level=logging.INFO,
                    format='[CENTRAL] %(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger("centralized_baseline")

TARGET_COL = 'Binary_Label'
LABEL_COL = 'Label'


def load_aggregated(data_dir: Path, num_clients: int):
    train_frames = []
    test_frames = []
    for cid in range(num_clients):
        tpath = data_dir / f"client_{cid}_train.csv"
        vpath = data_dir / f"client_{cid}_test.csv"
        if not tpath.exists() or not vpath.exists():
            raise FileNotFoundError(
                f"Missing partition for client {cid} in {data_dir}")
        train_frames.append(pd.read_csv(tpath))
        test_frames.append(pd.read_csv(vpath))
    train_df = pd.concat(train_frames, axis=0).reset_index(drop=True)
    test_df = pd.concat(test_frames, axis=0).reset_index(drop=True)
    logger.info(
        f"Aggregated train shape: {train_df.shape}; test shape: {test_df.shape}")
    return train_df, test_df


def split_features_labels(df: pd.DataFrame):
    if TARGET_COL not in df.columns:
        raise ValueError(f"Dataset missing {TARGET_COL}")
    X = df.drop(
        columns=[c for c in [TARGET_COL, LABEL_COL] if c in df.columns])
    y = df[TARGET_COL].astype(int).values
    # Keep only numeric
    X = X.select_dtypes(include=[np.number])
    return X.values, y


def normalize(train_X: np.ndarray, test_X: np.ndarray):
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0)
    std[std == 0] = 1.0
    return (train_X - mean) / std, (test_X - mean) / std


def main():
    ap = argparse.ArgumentParser(
        description="Centralized CNN baseline on cleaned federated partitions")
    ap.add_argument('--data_dir', type=Path,
                    default=Path('data/optimized/clean_partitions'))
    ap.add_argument('--num_clients', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--model_out', type=Path,
                    default=Path('results/centralized_model.keras'))
    ap.add_argument('--metrics_out', type=Path,
                    default=Path('results/centralized_training_results.json'))
    args = ap.parse_args()

    train_df, test_df = load_aggregated(args.data_dir, args.num_clients)
    X_train, y_train = split_features_labels(train_df)
    X_test, y_test = split_features_labels(test_df)

    X_train, X_test = normalize(X_train, X_test)

    input_features = X_train.shape[1]
    trainer = ModelTrainer(input_features=input_features, num_classes=1)
    trainer.create_model(learning_rate=args.lr)

    # Use a validation split from training data (e.g., 10%)
    val_size = int(0.1 * len(X_train))
    if val_size < 1:
        raise ValueError("Training set too small for validation split")
    X_val = X_train[:val_size]
    y_val = y_train[:val_size]
    X_tr = X_train[val_size:]
    y_tr = y_train[val_size:]

    history = trainer.model.get_model().fit(
        trainer.model.prepare_data(X_tr), y_tr,
        validation_data=(trainer.model.prepare_data(X_val), y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        verbose=1,
        callbacks=[
            # Early stopping logic
        ]
    )

    eval_results = trainer.evaluate_model(X_test, y_test)

    # Save model
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.model_out))

    # Persist metrics
    metrics_payload = {
        'test_accuracy': eval_results['test_accuracy'],
        'test_loss': eval_results['test_loss'],
        'confusion_matrix': eval_results['confusion_matrix'],
        'classification_report': eval_results['classification_report'],
        'train_size': int(len(X_train)),
        'test_size': int(len(X_test)),
        'epochs': args.epochs,
        'batch_size': args.batch,
        'learning_rate': args.lr,
        'input_features': input_features,
    }
    with args.metrics_out.open('w', encoding='utf-8') as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info(f"Saved metrics to {args.metrics_out}")

    # Generate comprehensive visualizations
    if VISUALIZATION_AVAILABLE:
        logger.info("🎨 Generating comprehensive training visualizations...")
        try:
            plots = generate_training_visualizations(
                model=trainer.model.get_model(),
                history=history,
                X_test=trainer.model.prepare_data(X_test),
                y_test=y_test,
                results_dir="results"
            )
            logger.info(
                f"✅ Generated {len(plots) if plots else 0} visualization plots")
        except Exception as e:
            logger.error(f"❌ Error generating visualizations: {e}")

    print(f"\n=== Centralized Baseline Results ===")
    print(f"Test Accuracy: {metrics_payload['test_accuracy']:.4f}")
    if VISUALIZATION_AVAILABLE:
        print(f"📊 Visualizations saved to: results/visualizations/")
    print(f"💾 Model saved to: {args.model_out}")
    print(f"📈 Metrics saved to: {args.metrics_out}")
    print(f"Test Loss: {metrics_payload['test_loss']:.4f}")
    print(f"Model saved: {args.model_out}")
    print(f"Metrics JSON: {args.metrics_out}")


if __name__ == '__main__':
    main()
