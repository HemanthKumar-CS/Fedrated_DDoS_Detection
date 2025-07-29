#!/usr/bin/env python3
"""
Federated Learning Server Implementation using Flower Framework
Implements a federated learning server for DDoS detection
"""

from src.models.cnn_model import create_ddos_cnn_model
import os
import sys
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Any
import flwr as fl
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.strategy import FedAvg
import logging
from datetime import datetime
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DDoSFederatedStrategy(FedAvg):
    """
    Custom Federated Averaging Strategy for DDoS Detection
    Extends Flower's FedAvg with custom evaluation and logging
    """

    def __init__(self,
                 min_fit_clients: int = 2,
                 min_evaluate_clients: int = 2,
                 min_available_clients: int = 2,
                 evaluate_fn: Optional[callable] = None,
                 on_fit_config_fn: Optional[callable] = None,
                 on_evaluate_config_fn: Optional[callable] = None,
                 accept_failures: bool = True,
                 initial_parameters: Optional[Parameters] = None,
                 fit_metrics_aggregation_fn: Optional[callable] = None,
                 evaluate_metrics_aggregation_fn: Optional[callable] = None):

        super().__init__(
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn
        )

        # Initialize tracking
        self.round_number = 0
        self.training_history = []
        self.evaluation_history = []

        logger.info("✅ DDoS Federated Strategy initialized")

    def aggregate_fit(self, server_round: int, results: List[Tuple[fl.client.ClientProxy, FitRes]], failures: List[BaseException]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate fit results from all clients

        Args:
            server_round: Current round number
            results: List of fit results from clients
            failures: List of failures from clients

        Returns:
            Tuple of aggregated parameters and metrics
        """
        logger.info(
            f"🔄 Round {server_round}: Aggregating fit results from {len(results)} clients")

        # Extract metrics from each client
        client_metrics = []
        for client_proxy, fit_res in results:
            if fit_res.metrics:
                client_metrics.append(fit_res.metrics)

        # Log individual client performance
        for i, metrics in enumerate(client_metrics):
            logger.info(
                f"  Client {metrics.get('client_id', i)}: Loss={metrics.get('loss', 'N/A'):.4f}, Accuracy={metrics.get('accuracy', 'N/A'):.4f}")

        # Aggregate using parent class
        aggregated_parameters, aggregated_metrics = super(
        ).aggregate_fit(server_round, results, failures)

        # Calculate average metrics across clients
        if client_metrics:
            avg_loss = np.mean([m.get('loss', 0) for m in client_metrics])
            avg_accuracy = np.mean([m.get('accuracy', 0)
                                   for m in client_metrics])

            aggregated_metrics["round"] = server_round
            aggregated_metrics["avg_train_loss"] = avg_loss
            aggregated_metrics["avg_train_accuracy"] = avg_accuracy
            aggregated_metrics["num_clients"] = len(client_metrics)

            # Store round history
            round_info = {
                "round": server_round,
                "num_clients": len(client_metrics),
                "avg_train_loss": float(avg_loss),
                "avg_train_accuracy": float(avg_accuracy),
                "client_metrics": client_metrics
            }
            self.training_history.append(round_info)

            logger.info(f"✅ Round {server_round} aggregation complete:")
            logger.info(f"  Average Loss: {avg_loss:.4f}")
            logger.info(f"  Average Accuracy: {avg_accuracy:.4f}")

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[fl.client.ClientProxy, fl.common.EvaluateRes]], failures: List[BaseException]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluate results from all clients

        Args:
            server_round: Current round number
            results: List of evaluate results from clients
            failures: List of failures from clients

        Returns:
            Tuple of aggregated loss and metrics
        """
        logger.info(
            f"📊 Round {server_round}: Aggregating evaluation results from {len(results)} clients")

        if not results:
            return None, {}

        # Extract losses and metrics
        losses = []
        client_metrics = []
        total_examples = 0

        for client_proxy, evaluate_res in results:
            losses.append(evaluate_res.loss)
            total_examples += evaluate_res.num_examples
            if evaluate_res.metrics:
                client_metrics.append(evaluate_res.metrics)

        # Weighted average loss
        weighted_loss = sum(losses) / len(losses)  # Simple average for now

        # Calculate average metrics
        aggregated_metrics = {"round": server_round}
        if client_metrics:
            avg_accuracy = np.mean([m.get('accuracy', 0)
                                   for m in client_metrics])
            aggregated_metrics["avg_test_accuracy"] = avg_accuracy
            aggregated_metrics["avg_test_loss"] = weighted_loss
            aggregated_metrics["total_test_examples"] = total_examples

            # Store evaluation history
            eval_info = {
                "round": server_round,
                "avg_test_loss": float(weighted_loss),
                "avg_test_accuracy": float(avg_accuracy),
                "total_examples": total_examples,
                "client_metrics": client_metrics
            }
            self.evaluation_history.append(eval_info)

            logger.info(f"✅ Round {server_round} evaluation complete:")
            logger.info(f"  Average Test Loss: {weighted_loss:.4f}")
            logger.info(f"  Average Test Accuracy: {avg_accuracy:.4f}")

        return weighted_loss, aggregated_metrics

    def save_history(self, filepath: str):
        """Save training and evaluation history to file"""
        history = {
            "training_history": self.training_history,
            "evaluation_history": self.evaluation_history,
            "timestamp": datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)

        logger.info(f"📝 History saved to {filepath}")


def get_initial_parameters() -> Parameters:
    """
    Create initial global model parameters

    Returns:
        Initial model parameters
    """
    logger.info("🏗️ Creating initial global model parameters")

    # Create a model to get initial parameters
    model = create_ddos_cnn_model()
    model.build()

    # Get initial weights
    initial_weights = model.get_model().get_weights()

    # Convert to Flower parameters format
    parameters = fl.common.ndarrays_to_parameters(initial_weights)

    logger.info(
        f"✅ Initial parameters created with {len(initial_weights)} arrays")

    return parameters


def create_fit_config(server_round: int) -> Dict[str, Any]:
    """
    Create configuration for training rounds

    Args:
        server_round: Current server round

    Returns:
        Configuration dictionary for training
    """
    # Adaptive training configuration
    if server_round <= 3:
        epochs = 5  # More epochs in early rounds
        batch_size = 32
    else:
        epochs = 3  # Fewer epochs in later rounds
        batch_size = 64

    config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "server_round": server_round
    }

    logger.info(f"📋 Round {server_round} training config: {config}")
    return config


def create_evaluate_config(server_round: int) -> Dict[str, Any]:
    """
    Create configuration for evaluation rounds

    Args:
        server_round: Current server round

    Returns:
        Configuration dictionary for evaluation
    """
    config = {
        "server_round": server_round
    }

    return config


def weighted_average(metrics: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Aggregate metrics using weighted average

    Args:
        metrics: List of (num_examples, metrics_dict) tuples

    Returns:
        Aggregated metrics dictionary
    """
    if not metrics:
        return {}

    # Calculate total examples
    total_examples = sum(num_examples for num_examples, _ in metrics)

    if total_examples == 0:
        return {}

    # Initialize aggregated metrics
    aggregated = {}

    # Aggregate each metric
    for num_examples, metric_dict in metrics:
        weight = num_examples / total_examples

        for key, value in metric_dict.items():
            if isinstance(value, (int, float)):
                if key not in aggregated:
                    aggregated[key] = 0.0
                aggregated[key] += weight * value

    return aggregated


def start_federated_server(num_clients: int = 4,
                           num_rounds: int = 10,
                           server_address: str = "localhost:8080",
                           save_history: bool = True):
    """
    Start the federated learning server

    Args:
        num_clients: Number of clients to wait for
        num_rounds: Number of federated learning rounds
        server_address: Server address to bind to
        save_history: Whether to save training history
    """
    logger.info("🚀 Starting Federated Learning Server")
    logger.info(f"📊 Configuration:")
    logger.info(f"  Clients: {num_clients}")
    logger.info(f"  Rounds: {num_rounds}")
    logger.info(f"  Address: {server_address}")

    try:
        # Create strategy
        strategy = DDoSFederatedStrategy(
            min_fit_clients=max(2, min(num_clients, 4)),  # At least 2, max 4
            min_evaluate_clients=max(2, min(num_clients, 4)),
            min_available_clients=num_clients,
            initial_parameters=get_initial_parameters(),
            on_fit_config_fn=create_fit_config,
            on_evaluate_config_fn=create_evaluate_config,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average
        )

        # Start server
        logger.info(f"🌐 Starting server on {server_address}")

        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy
        )

        logger.info("✅ Federated learning completed successfully!")

        # Save history if requested
        if save_history:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            history_file = f"../../results/federated_history_{timestamp}.json"
            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            strategy.save_history(history_file)

    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        raise


def main():
    """
    Main function for testing server functionality
    """
    print("🔧 Testing Federated Learning Server")
    print("=" * 50)

    try:
        # Test initial parameters creation
        print("🏗️ Testing initial parameters...")
        initial_params = get_initial_parameters()
        print(f"✅ Initial parameters created")

        # Test strategy creation
        print("\n🧪 Testing strategy creation...")
        strategy = DDoSFederatedStrategy(
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
            initial_parameters=initial_params
        )
        print("✅ Strategy created successfully")

        # Test configuration functions
        print("\n📋 Testing configuration functions...")
        fit_config = create_fit_config(1)
        eval_config = create_evaluate_config(1)
        print(f"Fit config: {fit_config}")
        print(f"Eval config: {eval_config}")

        print("\n🎉 All server tests passed!")
        print("\n💡 To start actual federated learning:")
        print("   Server: python flower_server.py --rounds 10 --clients 4")
        print("   Client: python flower_client.py --client_id 0")

    except Exception as e:
        print(f"❌ Server test failed: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Federated Learning Server for DDoS Detection")
    parser.add_argument("--clients", type=int, default=4,
                        help="Number of clients to wait for")
    parser.add_argument("--rounds", type=int, default=10,
                        help="Number of federated rounds")
    parser.add_argument("--address", type=str,
                        default="localhost:8080", help="Server address")
    parser.add_argument("--no_save", action="store_true",
                        help="Don't save training history")
    parser.add_argument("--test", action="store_true", help="Run test mode")

    args = parser.parse_args()

    if args.test:
        main()
    else:
        start_federated_server(
            num_clients=args.clients,
            num_rounds=args.rounds,
            server_address=args.address,
            save_history=not args.no_save
        )
