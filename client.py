#!/usr/bin/env python3
"""
Standalone Flower Federated Learning Client.
Run (after server is up):
  python client.py --cid 0
  python client.py --cid 1
  python client.py --cid 2
  python client.py --cid 3
"""
from __future__ import annotations
import os
import sys
import argparse
import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import warnings
import flwr as fl
import tensorflow as tf

# Local imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.models.trainer import ModelTrainer  # type: ignore
from src.models.cnn_model import create_ddos_cnn_model  # type: ignore

logging.basicConfig(level=logging.INFO, format='[CLIENT %(client_id)s] %(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger("federated_client")

# ---------------------------------------------------------------------------
# Data loading per client
# ---------------------------------------------------------------------------

def load_partition(data_dir: str, cid: int):
    train_path = os.path.join(data_dir, f"client_{cid}_train.csv")
    test_path = os.path.join(data_dir, f"client_{cid}_test.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing train partition: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test partition: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    label_col = 'Binary_Label'
    if label_col not in train_df.columns:
        raise ValueError(f"Label column '{label_col}' not found in training data (columns={list(train_df.columns)})")

    # Separate labels
    y_train = train_df[label_col].astype(int).values
    y_test = test_df[label_col].astype(int).values

    # Drop textual label column 'Label' if present along with Binary_Label
    drop_cols = [label_col]
    if 'Label' in train_df.columns:
        drop_cols.append('Label')
    X_train_df = train_df.drop(columns=drop_cols, errors='ignore')
    X_test_df = test_df.drop(columns=drop_cols, errors='ignore')

    # Handle non-numeric columns: factorize (stable per run) then scale
    for col in X_train_df.columns:
        if not np.issubdtype(X_train_df[col].dtype, np.number):
            # Factorize based on training set; unseen categories in test get -1
            categories, codes = np.unique(X_train_df[col].astype(str), return_inverse=True)
            mapping = {cat: i for i, cat in enumerate(categories)}
            X_train_df[col] = codes
            X_test_df[col] = X_test_df[col].astype(str).map(mapping).fillna(-1).astype(int)
            logger.info(f"Factorized non-numeric column '{col}' into {len(categories)} categories", extra={'client_id': cid})

    # Convert all to float32
    X_train = X_train_df.astype('float32').values
    X_test = X_test_df.astype('float32').values

    # Normalize (per-client) using train stats (avoid divide-by-zero)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    zero_std = std == 0
    if np.any(zero_std):
        # Replace zero std with 1 to avoid division by zero
        count_zero = int(zero_std.sum())
        warnings.warn(f"{count_zero} feature(s) had zero variance; left unscaled.")
        std[zero_std] = 1.0
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Reshape for CNN (samples, features, 1)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    return X_train, y_train, X_test, y_test

# ---------------------------------------------------------------------------
# Flower NumPyClient implementation
# ---------------------------------------------------------------------------
class DDoSClient(fl.client.NumPyClient):
    def __init__(self, cid: int, data_dir: str, epochs: int = 3, batch_size: int = 32):
        self.cid = cid
        self.data_dir = data_dir
        self.epochs = epochs
        self.batch_size = batch_size

        # Seed for reproducibility
        seed = 42 + cid
        np.random.seed(seed)
        tf.random.set_seed(seed)
        # Load partition first to know feature count
        self.X_train, self.y_train, self.X_test, self.y_test = load_partition(data_dir, cid)
        input_features = self.X_train.shape[1]

        # Build model with correct input size and binary output
        self.trainer = ModelTrainer(input_features=input_features, num_classes=1)
        self.trainer.create_model(learning_rate=0.001)
        logger.info(
            f"Client {self.cid} data: train={self.X_train.shape} test={self.X_test.shape}",
            extra={'client_id': self.cid}
        )

    # Flower interface
    def get_parameters(self, config: Dict[str, Any]):
        logger.info("get_parameters", extra={'client_id': self.cid})
        return self.trainer.model.get_model().get_weights()

    def set_parameters(self, parameters: List[np.ndarray]):
        self.trainer.model.get_model().set_weights(parameters)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]):  # type: ignore
        self.set_parameters(parameters)
        epochs = int(config.get("epochs", self.epochs))
        batch_size = int(config.get("batch_size", self.batch_size))
        logger.info(f"fit start epochs={epochs} batch={batch_size}", extra={'client_id': self.cid})
        history = self.trainer.model.get_model().fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        loss = history.history['loss'][-1]
        acc = history.history.get('accuracy', [None])[-1]
        logger.info(f"fit done loss={loss:.4f} acc={acc:.4f}", extra={'client_id': self.cid})
        return self.get_parameters({}), len(self.X_train), {"loss": float(loss), "accuracy": float(acc), "client_id": self.cid}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]):  # type: ignore
        self.set_parameters(parameters)
        eval_results = self.trainer.model.get_model().evaluate(self.X_test, self.y_test, verbose=0)
        # Keras returns [loss, acc, precision, recall] given current compile
        loss = float(eval_results[0])
        acc = float(eval_results[1]) if len(eval_results) > 1 else None
        precision = float(eval_results[2]) if len(eval_results) > 2 else None
        recall = float(eval_results[3]) if len(eval_results) > 3 else None
        metrics = {"client_id": self.cid}
        if acc is not None:
            metrics["accuracy"] = acc
        if precision is not None:
            metrics["precision"] = precision
        if recall is not None:
            metrics["recall"] = recall
        acc_str = f"{acc:.4f}" if acc is not None else "NA"
        logger.info(
            f"evaluate loss={loss:.4f} acc={acc_str}",
            extra={'client_id': self.cid}
        )
        return loss, len(self.X_test), metrics

# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Flower Federated Client")
    parser.add_argument('--cid', type=int, required=True, help='Client ID')
    # Use loopback for connecting (0.0.0.0 is only for binding on server side)
    parser.add_argument('--server', type=str, default='127.0.0.1:8080', help='Server address host:port (use localhost/127.0.0.1)')
    parser.add_argument('--data_dir', type=str, default='data/optimized', help='Directory with client_#_train.csv')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()

    client = DDoSClient(args.cid, args.data_dir, epochs=args.epochs, batch_size=args.batch)
    logger.info(f"Connecting to server {args.server}", extra={'client_id': args.cid})
    # New recommended Flower API (avoids deprecated start_numpy_client warnings)
    fl.client.start_client(server_address=args.server, client=client.to_client())


if __name__ == '__main__':
    main()
