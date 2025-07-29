#!/usr/bin/env python3
"""
Federated Learning Client Implementation using Flower Framework
Implements a federated learning client for DDoS detection
"""

from src.models.trainer import ModelTrainer
from src.models.cnn_model import create_ddos_cnn_model
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Tuple, Any
import flwr as fl
import logging
from collections import OrderedDict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DDoSFederatedClient(fl.client.NumPyClient):
    """
    Federated Learning Client for DDoS Detection
    Implements Flower NumPyClient interface
    """

    def __init__(self, client_id: int, train_data_path: str, test_data_path: str):
        """
        Initialize federated client

        Args:
            client_id: Unique identifier for this client
            train_data_path: Path to client's training data
            test_data_path: Path to client's test data
        """
        self.client_id = client_id
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path

        # Initialize model trainer
        self.trainer = ModelTrainer()
        self.trainer.create_model(learning_rate=0.001)

        # Load client data
        self._load_client_data()

        logger.info(f"✅ Federated client {client_id} initialized")

    def _load_client_data(self):
        """Load and preprocess client's local data"""
        logger.info(f"Loading data for client {self.client_id}")

        # Load training data
        self.X_train, self.y_train = self.trainer.load_data(
            self.train_data_path)

        # Load test data
        self.X_test, self.y_test = self.trainer.load_data(self.test_data_path)

        # Normalize features using training data statistics
        train_mean = self.X_train.mean(axis=0)
        train_std = self.X_train.std(axis=0)

        self.X_train = (self.X_train - train_mean) / (train_std + 1e-8)
        self.X_test = (self.X_test - train_mean) / (train_std + 1e-8)

        # Prepare for CNN
        self.X_train = self.trainer.model.prepare_data(self.X_train)
        self.X_test = self.trainer.model.prepare_data(self.X_test)

        logger.info(f"Client {self.client_id} data loaded:")
        logger.info(f"  Training samples: {len(self.X_train)}")
        logger.info(f"  Test samples: {len(self.X_test)}")

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """
        Get model parameters

        Args:
            config: Configuration dictionary

        Returns:
            List of model parameters
        """
        logger.info(f"Client {self.client_id}: Getting model parameters")
        return self.trainer.model.get_model().get_weights()

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters

        Args:
            parameters: List of model parameters to set
        """
        logger.info(f"Client {self.client_id}: Setting model parameters")
        self.trainer.model.get_model().set_weights(parameters)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Train model on client's local data

        Args:
            parameters: Global model parameters from server
            config: Training configuration

        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        logger.info(f"Client {self.client_id}: Starting local training")

        # Set global parameters
        self.set_parameters(parameters)

        # Get training configuration
        epochs = config.get("epochs", 5)
        batch_size = config.get("batch_size", 32)

        # Train model locally
        history = self.trainer.model.get_model().fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0  # Quiet training for federated setting
        )

        # Get updated parameters
        updated_parameters = self.get_parameters({})

        # Calculate metrics
        final_loss = history.history["loss"][-1]
        final_accuracy = history.history["accuracy"][-1]

        metrics = {
            "loss": final_loss,
            "accuracy": final_accuracy,
            "client_id": self.client_id
        }

        logger.info(
            f"Client {self.client_id}: Training completed - Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}")

        return updated_parameters, len(self.X_train), metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluate model on client's local test data

        Args:
            parameters: Model parameters to evaluate
            config: Evaluation configuration

        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        logger.info(f"Client {self.client_id}: Starting local evaluation")

        # Set parameters
        self.set_parameters(parameters)

        # Evaluate model
        loss, accuracy = self.trainer.model.get_model().evaluate(
            self.X_test, self.y_test, verbose=0
        )

        # Additional metrics
        y_pred_proba = self.trainer.model.get_model().predict(self.X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Calculate per-class accuracy
        per_class_accuracy = {}
        for class_id in np.unique(self.y_test):
            mask = self.y_test == class_id
            if np.sum(mask) > 0:
                class_accuracy = np.mean(y_pred[mask] == self.y_test[mask])
                per_class_accuracy[f"class_{class_id}_accuracy"] = class_accuracy

        metrics = {
            "accuracy": accuracy,
            "client_id": self.client_id,
            **per_class_accuracy
        }

        logger.info(
            f"Client {self.client_id}: Evaluation completed - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        return loss, len(self.X_test), metrics


def create_federated_client(client_id: int, data_directory: str = "../../data/optimized") -> DDoSFederatedClient:
    """
    Factory function to create a federated client

    Args:
        client_id: Unique identifier for the client
        data_directory: Directory containing client data files

    Returns:
        Configured DDoSFederatedClient instance
    """
    # Construct data file paths
    train_data_path = os.path.join(
        data_directory, f"client_{client_id}_train.csv")
    test_data_path = os.path.join(
        data_directory, f"client_{client_id}_test.csv")

    # Verify files exist
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Training data not found: {train_data_path}")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found: {test_data_path}")

    # Create client
    client = DDoSFederatedClient(client_id, train_data_path, test_data_path)

    return client


def start_federated_client(client_id: int, server_address: str = "localhost:8080", data_directory: str = "../../data/optimized"):
    """
    Start a federated learning client

    Args:
        client_id: Unique identifier for this client
        server_address: Address of the federated learning server
        data_directory: Directory containing client data
    """
    logger.info(f"🚀 Starting Federated Client {client_id}")
    logger.info(f"📡 Connecting to server at: {server_address}")

    try:
        # Create client
        client = create_federated_client(client_id, data_directory)

        # Start federated learning
        fl.client.start_numpy_client(
            server_address=server_address,
            client=client
        )

        logger.info(f"✅ Client {client_id} completed federated learning")

    except Exception as e:
        logger.error(f"❌ Error in client {client_id}: {e}")
        raise


def main():
    """
    Main function for testing client functionality
    """
    print("🔧 Testing Federated Learning Client")
    print("=" * 50)

    try:
        # Test client creation
        client_id = 0
        client = create_federated_client(client_id)

        print(f"✅ Client {client_id} created successfully")
        print(f"Training samples: {len(client.X_train)}")
        print(f"Test samples: {len(client.X_test)}")

        # Test parameter operations
        print("\n🧪 Testing parameter operations...")
        initial_params = client.get_parameters({})
        print(f"Number of parameter arrays: {len(initial_params)}")
        print(f"First layer shape: {initial_params[0].shape}")

        # Test local training (1 epoch)
        print("\n🏋️ Testing local training...")
        config = {"epochs": 1, "batch_size": 32}
        updated_params, num_examples, train_metrics = client.fit(
            initial_params, config)

        print(f"Training completed:")
        print(f"  Examples used: {num_examples}")
        print(f"  Final accuracy: {train_metrics['accuracy']:.4f}")

        # Test evaluation
        print("\n📊 Testing evaluation...")
        loss, num_examples, eval_metrics = client.evaluate(updated_params, {})

        print(f"Evaluation completed:")
        print(f"  Test loss: {loss:.4f}")
        print(f"  Test accuracy: {eval_metrics['accuracy']:.4f}")

        print("\n🎉 All client tests passed!")

    except Exception as e:
        print(f"❌ Client test failed: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Federated Learning Client for DDoS Detection")
    parser.add_argument("--client_id", type=int,
                        default=0, help="Client ID (0-3)")
    parser.add_argument("--server", type=str,
                        default="localhost:8080", help="Server address")
    parser.add_argument("--data_dir", type=str,
                        default="../../data/optimized", help="Data directory")
    parser.add_argument("--test", action="store_true", help="Run test mode")

    args = parser.parse_args()

    if args.test:
        main()
    else:
        start_federated_client(args.client_id, args.server, args.data_dir)
