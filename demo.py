#!/usr/bin/env python3
"""
Complete Federated DDoS Detection Demo
Demonstrates the full pipeline from data loading to federated learning
"""

from src.models.cnn_model import create_ddos_cnn_model
from src.models.trainer import ModelTrainer
import os
import sys
import time
import threading
import subprocess
import signal
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FederatedDemoOrchestrator:
    """
    Orchestrates the complete federated learning demo
    """

    def __init__(self, base_dir: str = None):
        """
        Initialize the demo orchestrator

        Args:
            base_dir: Base directory for the project
        """
        self.base_dir = base_dir or os.path.dirname(__file__)
        self.data_dir = os.path.join(self.base_dir, "data", "optimized")
        self.results_dir = os.path.join(self.base_dir, "results")
        self.src_dir = os.path.join(self.base_dir, "src")

        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)

        # Process management
        self.server_process = None
        self.client_processes = []

        logger.info("🚀 Federated Demo Orchestrator initialized")

    def verify_environment(self) -> bool:
        """
        Verify that all required files and dependencies exist

        Returns:
            True if environment is ready, False otherwise
        """
        logger.info("🔍 Verifying environment...")

        # Check required directories
        required_dirs = [self.data_dir, self.src_dir]
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                logger.error(f"❌ Required directory not found: {dir_path}")
                return False

        # Check required data files
        required_files = [
            os.path.join(self.data_dir, f"client_{i}_train.csv") for i in range(4)
        ] + [
            os.path.join(self.data_dir, f"client_{i}_test.csv") for i in range(4)
        ]

        for file_path in required_files:
            if not os.path.exists(file_path):
                logger.error(f"❌ Required data file not found: {file_path}")
                return False

        # Check source files
        source_files = [
            os.path.join(self.src_dir, "models", "cnn_model.py"),
            os.path.join(self.src_dir, "models", "trainer.py"),
            os.path.join(self.src_dir, "federated", "flower_client.py"),
            os.path.join(self.src_dir, "federated", "flower_server.py")
        ]

        for file_path in source_files:
            if not os.path.exists(file_path):
                logger.error(f"❌ Required source file not found: {file_path}")
                return False

        logger.info("✅ Environment verification passed")
        return True

    def test_model_standalone(self) -> bool:
        """
        Test the CNN model in standalone mode

        Returns:
            True if test passes, False otherwise
        """
        logger.info("🧪 Testing CNN model in standalone mode...")

        try:
            # Initialize trainer
            trainer = ModelTrainer()
            trainer.create_model(learning_rate=0.001)

            # Load a small subset of data for testing
            test_data_path = os.path.join(self.data_dir, "client_0_train.csv")
            X, y = trainer.load_data(test_data_path)

            # Take only first 100 samples for quick test
            X_test = X[:100]
            y_test = y[:100]

            # Reshape for CNN input (batch_size, features, channels)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            # Quick training test (1 epoch)
            logger.info("  Training model for 1 epoch...")
            history = trainer.model.model.fit(
                X_test, y_test,
                epochs=1,
                batch_size=32,
                verbose=0
            )

            # Evaluation test
            results = trainer.model.model.evaluate(X_test, y_test, verbose=0)
            loss = results[0] if isinstance(results, list) else results
            accuracy = results[1] if isinstance(
                results, list) and len(results) > 1 else 0.0

            logger.info(
                f"✅ Standalone test passed - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            return True

        except Exception as e:
            logger.error(f"❌ Standalone test failed: {e}")
            return False

    def run_centralized_baseline(self, epochs: int = 10) -> dict:
        """
        Run centralized training as baseline

        Args:
            epochs: Number of training epochs

        Returns:
            Training results dictionary
        """
        logger.info("📊 Running centralized baseline...")

        try:
            # Initialize trainer
            trainer = ModelTrainer()
            trainer.create_model(learning_rate=0.001)

            # Load all data
            all_X_train, all_y_train = [], []
            all_X_test, all_y_test = [], []

            for client_id in range(4):
                # Load training data
                train_path = os.path.join(
                    self.data_dir, f"client_{client_id}_train.csv")
                X_train, y_train = trainer.load_data(train_path)
                all_X_train.append(X_train)
                all_y_train.append(y_train)

                # Load test data
                test_path = os.path.join(
                    self.data_dir, f"client_{client_id}_test.csv")
                X_test, y_test = trainer.load_data(test_path)
                all_X_test.append(X_test)
                all_y_test.append(y_test)

            # Combine all data
            import numpy as np
            X_train_combined = np.vstack(all_X_train)
            y_train_combined = np.hstack(all_y_train)
            X_test_combined = np.vstack(all_X_test)
            y_test_combined = np.hstack(all_y_test)

            # Train centralized model
            logger.info(
                f"  Training centralized model with {len(X_train_combined)} samples...")

            # Split training data for validation
            from sklearn.model_selection import train_test_split
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train_combined, y_train_combined,
                test_size=0.2, random_state=42, stratify=y_train_combined
            )

            results = trainer.train_model(
                X_train_split, y_train_split,
                X_val_split, y_val_split,
                epochs=epochs,
                batch_size=64,
                verbose=1
            )

            # Evaluate on test set
            test_results = trainer.evaluate_model(
                X_test_combined, y_test_combined)
            results["evaluation"] = test_results

            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(
                self.results_dir, f"centralized_baseline_{timestamp}.json")

            import json
            with open(results_file, 'w') as f:
                json.dump({
                    "type": "centralized_baseline",
                    "timestamp": timestamp,
                    "epochs": epochs,
                    "training_samples": len(X_train_combined),
                    "test_samples": len(X_test_combined),
                    "final_metrics": results["evaluation"]
                }, f, indent=2)

            logger.info(f"✅ Centralized baseline completed")
            logger.info(
                f"  Final Test Accuracy: {results['evaluation']['accuracy']:.4f}")
            logger.info(f"  Results saved to: {results_file}")

            return results

        except Exception as e:
            logger.error(f"❌ Centralized baseline failed: {e}")
            return {}

    def start_federated_server(self, num_rounds: int = 10, num_clients: int = 4) -> bool:
        """
        Start the federated learning server

        Args:
            num_rounds: Number of federated rounds
            num_clients: Number of clients to wait for

        Returns:
            True if server started successfully
        """
        logger.info("🌐 Starting federated learning server...")

        try:
            server_script = os.path.join(
                self.src_dir, "federated", "flower_server.py")

            # Start server in subprocess
            cmd = [
                sys.executable, server_script,
                "--rounds", str(num_rounds),
                "--clients", str(num_clients),
                "--address", "localhost:8080"
            ]

            # Set environment for subprocess
            env = os.environ.copy()
            env['PYTHONPATH'] = self.base_dir

            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.base_dir,
                env=env
            )

            # Give server time to start
            time.sleep(3)

            # Check if server is still running
            if self.server_process.poll() is None:
                logger.info("✅ Federated server started successfully")
                return True
            else:
                stdout, stderr = self.server_process.communicate()
                logger.error(f"❌ Server failed to start")
                logger.error(f"STDOUT: {stdout}")
                logger.error(f"STDERR: {stderr}")
                return False

        except Exception as e:
            logger.error(f"❌ Failed to start server: {e}")
            return False

    def start_federated_clients(self, num_clients: int = 4, delay: float = 2.0) -> bool:
        """
        Start federated learning clients

        Args:
            num_clients: Number of clients to start
            delay: Delay between starting clients

        Returns:
            True if all clients started successfully
        """
        logger.info(f"👥 Starting {num_clients} federated clients...")

        try:
            client_script = os.path.join(
                self.src_dir, "federated", "flower_client.py")

            for client_id in range(num_clients):
                logger.info(f"  Starting client {client_id}...")

                cmd = [
                    sys.executable, client_script,
                    "--client_id", str(client_id),
                    "--server", "localhost:8080",
                    "--data_dir", self.data_dir
                ]

                # Set environment for subprocess
                env = os.environ.copy()
                env['PYTHONPATH'] = self.base_dir

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=self.base_dir,
                    env=env
                )

                self.client_processes.append(process)

                # Delay between client starts
                if client_id < num_clients - 1:
                    time.sleep(delay)

            logger.info(f"✅ All {num_clients} clients started")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to start clients: {e}")
            return False

    def wait_for_completion(self, timeout: int = 300) -> bool:  # Reduced timeout
        """
        Wait for federated learning to complete

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if completed successfully
        """
        logger.info("⏳ Waiting for federated learning to complete...")

        start_time = time.time()

        try:
            # Wait for server to complete
            if self.server_process:
                self.server_process.wait(timeout=timeout)
                logger.info("✅ Server completed")

            # Give clients a short time to finish after server completes
            client_timeout = 30  # 30 seconds max for clients to finish
            logger.info(f"⏳ Giving clients {client_timeout}s to complete...")

            for i, process in enumerate(self.client_processes):
                try:
                    process.wait(timeout=client_timeout)
                    logger.info(f"✅ Client {i} completed")
                except subprocess.TimeoutExpired:
                    logger.warning(f"⚠️ Client {i} timed out, will terminate")

            # Force cleanup of any remaining processes
            self.cleanup_processes()

            logger.info("🎉 Federated learning completed successfully!")
            return True

        except subprocess.TimeoutExpired:
            logger.error("⏰ Federated learning timed out")
            self.cleanup_processes()
            return False
        except Exception as e:
            logger.error(f"❌ Error waiting for completion: {e}")
            self.cleanup_processes()
            return False

    def cleanup_processes(self):
        """Clean up any running processes"""
        logger.info("🧹 Cleaning up processes...")

        # Terminate server
        if self.server_process and self.server_process.poll() is None:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()

        # Terminate clients
        for process in self.client_processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

        logger.info("✅ Cleanup completed")

    def run_complete_demo(self,
                          run_centralized: bool = True,
                          run_federated: bool = True,
                          centralized_epochs: int = 10,
                          federated_rounds: int = 10,
                          num_clients: int = 4) -> dict:
        """
        Run the complete federated learning demo

        Args:
            run_centralized: Whether to run centralized baseline
            run_federated: Whether to run federated learning
            centralized_epochs: Epochs for centralized training
            federated_rounds: Rounds for federated learning
            num_clients: Number of federated clients

        Returns:
            Demo results dictionary
        """
        logger.info("🚀 Starting Complete Federated DDoS Detection Demo")
        logger.info("=" * 60)

        results = {
            "start_time": datetime.now().isoformat(),
            "centralized_results": {},
            "federated_results": {},
            "success": False
        }

        try:
            # Verify environment
            if not self.verify_environment():
                logger.error("❌ Environment verification failed")
                return results

            # Test model standalone
            if not self.test_model_standalone():
                logger.error("❌ Standalone model test failed")
                return results

            # Run centralized baseline
            if run_centralized:
                logger.info("\n" + "="*30 + " CENTRALIZED BASELINE " + "="*30)
                centralized_results = self.run_centralized_baseline(
                    centralized_epochs)
                results["centralized_results"] = centralized_results

            # Run federated learning
            if run_federated:
                logger.info("\n" + "="*30 + " FEDERATED LEARNING " + "="*30)

                # Start server
                if not self.start_federated_server(federated_rounds, num_clients):
                    logger.error("❌ Failed to start federated server")
                    return results

                # Start clients
                if not self.start_federated_clients(num_clients):
                    logger.error("❌ Failed to start federated clients")
                    return results

                # Wait for completion
                if not self.wait_for_completion():
                    logger.error(
                        "❌ Federated learning did not complete successfully")
                    return results

                results["federated_results"] = {"completed": True}

            # Mark as successful
            results["success"] = True
            results["end_time"] = datetime.now().isoformat()

            logger.info("\n" + "="*60)
            logger.info("🎉 DEMO COMPLETED SUCCESSFULLY! 🎉")
            logger.info("=" * 60)

            return results

        except KeyboardInterrupt:
            logger.info("\n⚠️ Demo interrupted by user")
            self.cleanup_processes()
            return results
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            self.cleanup_processes()
            return results
        finally:
            self.cleanup_processes()


def main():
    """Main function for running the demo"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Federated DDoS Detection Demo")
    parser.add_argument("--no_centralized", action="store_true",
                        help="Skip centralized baseline")
    parser.add_argument("--no_federated", action="store_true",
                        help="Skip federated learning")
    parser.add_argument("--centralized_epochs", type=int,
                        default=10, help="Epochs for centralized training")
    parser.add_argument("--federated_rounds", type=int,
                        default=10, help="Rounds for federated learning")
    parser.add_argument("--clients", type=int, default=4,
                        help="Number of federated clients")
    parser.add_argument("--base_dir", type=str,
                        help="Base directory for the project")

    args = parser.parse_args()

    # Create orchestrator
    orchestrator = FederatedDemoOrchestrator(args.base_dir)

    # Run demo
    results = orchestrator.run_complete_demo(
        run_centralized=not args.no_centralized,
        run_federated=not args.no_federated,
        centralized_epochs=args.centralized_epochs,
        federated_rounds=args.federated_rounds,
        num_clients=args.clients
    )

    # Print final status
    if results["success"]:
        print("\n✅ Demo completed successfully!")
    else:
        print("\n❌ Demo failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
