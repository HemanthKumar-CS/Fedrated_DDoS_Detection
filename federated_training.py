#!/usr/bin/env python3
"""
Federated Learning Training Script
Demonstrates distributed training using the balanced dataset and CNN model
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import flwr as fl
from typing import Dict, List, Tuple
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.cnn_model import create_ddos_cnn_model
from src.data.preprocessing import prepare_data
from src.federated.flower_client import DDoSFederatedClient
from src.federated.flower_server import create_strategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_client_data(client_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load training and test data for a specific client"""
    
    # Load client-specific data
    train_path = f"data/optimized/client_{client_id}_train.csv"
    test_path = f"data/optimized/client_{client_id}_test.csv"
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Client {client_id} data not found")
    
    # Load and prepare training data
    train_df = pd.read_csv(train_path)
    X_train = train_df.drop('Binary_Label', axis=1).values
    y_train = train_df['Binary_Label'].values
    
    # Load and prepare test data
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop('Binary_Label', axis=1).values
    y_test = test_df['Binary_Label'].values
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Reshape for CNN (samples, features, 1)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    logger.info(f"Client {client_id} data loaded: Train={X_train.shape}, Test={X_test.shape}")
    
    return X_train, y_train, X_test, y_test

def create_client_fn(client_id: int):
    """Create a client function for Flower simulation"""
    
    def client_fn(cid: str) -> DDoSFederatedClient:
        # Load client data
        X_train, y_train, X_test, y_test = load_client_data(client_id)
        
        # Create model
        model = create_ddos_cnn_model(input_shape=(X_train.shape[1], 1))
        
        # Create client
        return DDoSFederatedClient(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            client_id=client_id
        )
    
    return client_fn

def run_federated_simulation(num_rounds: int = 10, num_clients: int = 4):
    """Run federated learning simulation"""
    
    logger.info("🚀 Starting Federated Learning Simulation")
    logger.info(f"Configuration: {num_rounds} rounds, {num_clients} clients")
    
    # Create strategy
    strategy = create_strategy()
    
    # Create client functions
    client_fns = {}
    for i in range(num_clients):
        client_fns[str(i)] = create_client_fn(i)
    
    # Start simulation
    start_time = datetime.now()
    
    history = fl.simulation.start_simulation(
        client_fn=lambda cid: client_fns[cid](),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.25},  # Adjust based on your hardware
    )
    
    end_time = datetime.now()
    training_time = end_time - start_time
    
    logger.info(f"✅ Federated training completed in: {training_time}")
    
    # Save results
    results = {
        "federated_training_time": str(training_time),
        "num_rounds": num_rounds,
        "num_clients": num_clients,
        "final_accuracy": history.metrics_distributed.get("accuracy", [])[-1][1] if history.metrics_distributed.get("accuracy") else "N/A",
        "training_history": {
            "losses_distributed": history.losses_distributed,
            "metrics_distributed": history.metrics_distributed,
            "losses_centralized": history.losses_centralized,
            "metrics_centralized": history.metrics_centralized
        }
    }
    
    # Save to file
    with open("results/federated_training_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("📊 Federated learning results saved to results/federated_training_results.json")
    
    return history

def compare_centralized_vs_federated():
    """Compare centralized and federated learning results"""
    
    logger.info("📈 Comparing Centralized vs Federated Learning Results")
    
    # Load centralized results
    try:
        with open("results/balanced_training_results.json", "r") as f:
            centralized_results = json.load(f)
        
        logger.info("Centralized Model Results:")
        logger.info(f"  - Test Accuracy: {centralized_results.get('test_accuracy', 'N/A'):.4f}")
        logger.info(f"  - Test F1-Score: {centralized_results.get('test_f1', 'N/A'):.4f}")
        logger.info(f"  - Training Time: {centralized_results.get('training_time', 'N/A')}")
        
    except FileNotFoundError:
        logger.warning("Centralized results not found")
    
    # Load federated results
    try:
        with open("results/federated_training_results.json", "r") as f:
            federated_results = json.load(f)
        
        logger.info("Federated Model Results:")
        logger.info(f"  - Final Accuracy: {federated_results.get('final_accuracy', 'N/A')}")
        logger.info(f"  - Training Time: {federated_results.get('federated_training_time', 'N/A')}")
        logger.info(f"  - Number of Rounds: {federated_results.get('num_rounds', 'N/A')}")
        
    except FileNotFoundError:
        logger.warning("Federated results not found")

if __name__ == "__main__":
    try:
        # Check if client data exists
        for i in range(4):
            train_path = f"data/optimized/client_{i}_train.csv"
            test_path = f"data/optimized/client_{i}_test.csv"
            if not os.path.exists(train_path) or not os.path.exists(test_path):
                logger.error(f"Client {i} data not found. Please run federated data preparation first.")
                sys.exit(1)
        
        # Run federated learning
        logger.info("=" * 70)
        logger.info("🌐 FEDERATED DDOS DETECTION TRAINING")
        logger.info("=" * 70)
        
        history = run_federated_simulation(num_rounds=5, num_clients=4)
        
        # Compare results
        compare_centralized_vs_federated()
        
        logger.info("=" * 70)
        logger.info("🎉 Federated Learning Demonstration Complete!")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"❌ Error during federated training: {str(e)}")
        sys.exit(1)
