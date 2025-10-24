#!/usr/bin/env python3
"""
Standalone Federated Learning Server (Flower) with Multi-Krum style robust aggregation
Run: python server.py --rounds 3 --address 0.0.0.0:8080 --f 1 --log

Notes:
- Starts BEFORE any clients.
- Implements a custom strategy extending FedAvg overriding aggregate_fit with Multi-Krum selection.
- Falls back to FedAvg if too few clients for Multi-Krum (n < 2f + 3).
"""
from __future__ import annotations
from src.models.cnn_model import create_ddos_cnn_model  # type: ignore
import os
import sys
import argparse
import logging
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import json
import pandas as pd
import tensorflow as tf  # type: ignore
import numpy as np
import flwr as fl
from flwr.common import Parameters, FitRes, Scalar
from flwr.server.strategy import FedAvg

# Ensure src import path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Optional visualization support
VISUALIZATION_AVAILABLE = False
try:
    from src.visualization.training_visualizer import (generate_training_visualizations,
                                                       generate_federated_analysis_visualizations)
    VISUALIZATION_AVAILABLE = True
except Exception:
    # Keep VISUALIZATION_AVAILABLE as False; plotting will be skipped gracefully
    try:
        generate_training_visualizations  # type: ignore[name-defined]
        generate_federated_analysis_visualizations  # type: ignore[name-defined]
    except Exception:
        # Ensure the symbol exists to avoid NameError if referenced accidentally
        generate_training_visualizations = None  # type: ignore[assignment]
        generate_federated_analysis_visualizations = None  # type: ignore[assignment]
    print("Warning: Visualization module not available")

logging.basicConfig(level=logging.INFO,
                    format='[SERVER] %(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger("federated_server")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_initial_parameters(initial_model_path: Optional[str] = None) -> Optional[Parameters]:
    """Return initial global model parameters if a compatible model is provided.

    If a saved Keras model path is provided and exists, load its weights.
    If no path is given, return None so the server will use client-provided
    initial parameters (avoids input feature mismatch across environments).
    """
    if initial_model_path and Path(initial_model_path).is_file():
        try:
            logger.info(f"Loading initial weights from {initial_model_path}")
            loaded = tf.keras.models.load_model(
                initial_model_path, compile=False)
            weights = loaded.get_weights()
            return fl.common.ndarrays_to_parameters(weights)
        except Exception as e:
            logger.error(
                f"Failed to load model at {initial_model_path}: {e}; proceeding without initial parameters")
            return None
    # No initial model provided: let clients provide initial weights
    return None


def flatten_weights(ndarrays: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([w.ravel() for w in ndarrays])


def reconstruct_weights(flat: np.ndarray, template: List[np.ndarray]) -> List[np.ndarray]:
    rebuilt: List[np.ndarray] = []
    offset = 0
    for arr in template:
        size = arr.size
        rebuilt.append(flat[offset:offset+size].reshape(arr.shape))
        offset += size
    return rebuilt


def determine_input_features(default_if_missing: int = 78) -> int:
    """Determine input feature count for saving/rebuilding the global model.

    Prefers reading selected_features.json to get the length (optimized mode = 30).
    Falls back to default_if_missing when not found.
    """
    candidates = [
        os.path.join("data", "optimized", "clean_partitions", "selected_features.json"),
        os.path.join("data", "optimized", "selected_features.json"),
    ]
    for sp in candidates:
        if os.path.exists(sp):
            try:
                with open(sp, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                    features = obj.get("features", obj if isinstance(obj, list) else None)
                if features:
                    return int(len(features))
            except Exception as e:
                logger.warning(f"Could not read {sp} to determine input features: {e}")
    return default_if_missing


def load_global_test_data():
    """Load global test dataset for federated analysis"""
    try:
        # Load the realistic test dataset
        test_file = "data/optimized/realistic_test.csv"
        if not os.path.exists(test_file):
            logger.warning(f"Global test file {test_file} not found")
            return None, None
        
        logger.info(f"Loading global test data from {test_file}")
        test_data = pd.read_csv(test_file)
        
        # Use the same preprocessing as clients
        label_col = 'Binary_Label'
        if label_col not in test_data.columns:
            logger.error(f"Label column '{label_col}' not found in test data")
            return None, None
        
        # Separate labels
        y_test = test_data[label_col].astype(int).values
        
        # Drop label columns
        drop_cols = [label_col]
        if 'Label' in test_data.columns:
            drop_cols.append('Label')
        X_test_df = test_data.drop(columns=drop_cols, errors='ignore')

        # If selected_features.json exists, enforce the same 30-feature schema/order
        selected_paths = [
            os.path.join("data", "optimized", "selected_features.json"),
            os.path.join("data", "optimized", "clean_partitions", "selected_features.json"),
        ]
        selected_features = None
        for sp in selected_paths:
            if os.path.exists(sp):
                try:
                    with open(sp, "r", encoding="utf-8") as f:
                        obj = json.load(f)
                        selected_features = obj.get("features", obj if isinstance(obj, list) else None)
                    logger.info(f"Using optimized feature list for global evaluation from {sp} ({len(selected_features or [])} features)")
                except Exception as e:
                    logger.warning(f"Failed to read selected_features.json at {sp}: {e}")
                break
        if selected_features:
            # add missing columns as zeros and drop extra columns
            missing = [c for c in selected_features if c not in X_test_df.columns]
            for m in missing:
                X_test_df[m] = 0.0
            X_test_df = X_test_df[[c for c in selected_features]]
        
        # Handle non-numeric columns by factorizing
        for col in X_test_df.columns:
            if not np.issubdtype(X_test_df[col].dtype, np.number):
                # Factorize categorical columns
                X_test_df[col] = pd.Categorical(X_test_df[col]).codes
        
        # Convert to float32
        X_test = X_test_df.astype('float32').values
        
        # Basic normalization (using dataset statistics)
        mean = X_test.mean(axis=0)
        std = X_test.std(axis=0)
        zero_std = std == 0
        if np.any(zero_std):
            std[zero_std] = 1.0
        X_test = (X_test - mean) / std
        
        # Reshape for CNN (samples, features, 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        logger.info(f"Loaded global test data: {X_test.shape} samples")
        return X_test, y_test
        
    except Exception as e:
        logger.error(f"Error loading global test data: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# ---------------------------------------------------------------------------
# Multi-Krum FedAvg Strategy
# ---------------------------------------------------------------------------


class MultiKrumFedAvg(FedAvg):
    """FedAvg variant performing Multi-Krum selection before averaging.

    Parameters:
        f: Assumed max number of Byzantine (malicious) clients.
        m: Number of selected updates to average after Krum scores (if None, m = n - f - 2)
    """

    def __init__(self, f: int = 1, m: Optional[int] = None, history_path: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.f = f
        self.m = m
        # Track separate train/test accuracies per round
        self.history: Dict[str, List[float]] = {
            "train_accuracy": [], "test_accuracy": []}
        self.history_path = Path(history_path) if history_path else Path(
            "results/federated_metrics_history.json")
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

        # Enhanced Multi-Krum parameters
        self.aggregation_log: List[Dict[str, Any]] = []
        self.client_performance_history: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Helper to persist history after updates
    # ------------------------------------------------------------------
    def _persist_history(self) -> None:
        try:
            # Enhanced history with aggregation logs
            enhanced_history = {
                "train_accuracy": self.history["train_accuracy"],
                "test_accuracy": self.history["test_accuracy"],
                "aggregation_log": self.aggregation_log,
                "client_performance_history": self.client_performance_history
            }
            with self.history_path.open("w", encoding="utf-8") as f:
                json.dump(enhanced_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist history: {e}")

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.client.ClientProxy, FitRes]],
        failures: List[BaseException]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # Extract (weights list, num_examples, metrics)
        weights_results: List[List[np.ndarray]] = []
        num_examples: List[int] = []
        client_ids: List[str] = []
        metrics_list: List[Dict[str, Scalar]] = []
        for client_proxy, fit_res in results:
            nds = fl.common.parameters_to_ndarrays(fit_res.parameters)
            weights_results.append(nds)
            num_examples.append(fit_res.num_examples)
            client_ids.append(client_proxy.cid)
            metrics_list.append(fit_res.metrics or {})

        n = len(weights_results)
        template = weights_results[0]

        # Convert to update vectors (current weights themselves since FedAvg style) relative to global previous? Use absolute weights.
        flat_updates = [flatten_weights(w) for w in weights_results]

        # Multi-Krum requirement: n >= 2f + 3. If not, fallback to FedAvg.
        if n < (2 * self.f + 3):
            logger.info(
                f"[Round {server_round}] Too few clients for Multi-Krum (n={n} < 2f+3={2*self.f+3}); fallback to FedAvg")
            return super().aggregate_fit(server_round, results, failures)

        # Determine m (number of selected updates)
        m = self.m if self.m is not None else max(1, n - self.f - 2)
        m = min(m, n)  # safety

        # Compute pairwise squared distances
        dist_matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(flat_updates[i] - flat_updates[j]) ** 2
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # Enhanced Multi-Krum scoring with client performance history
        retained = n - self.f - 2
        retained = max(1, retained)
        scores: List[Tuple[float, int]] = []

        for i in range(n):
            # distances to others
            row = np.sort(dist_matrix[i, np.arange(n) != i])
            krum_score = np.sum(row[:retained])

            # Optional: Weight by historical client performance
            client_id = client_ids[i]
            if client_id in self.client_performance_history:
                avg_performance = np.mean(
                    self.client_performance_history[client_id])
                # Adjust score based on historical performance (lower score = better)
                performance_weight = max(
                    0.1, 1.0 - avg_performance) if avg_performance > 0 else 1.0
                adjusted_score = krum_score * performance_weight
            else:
                adjusted_score = krum_score

            scores.append((adjusted_score, i))

        scores.sort(key=lambda x: x[0])

        selected_indices = [idx for (_, idx) in scores[:m]]

        # Log aggregation details
        aggregation_info = {
            "round": server_round,
            "total_clients": n,
            "selected_clients": selected_indices,
            "krum_scores": [(client_ids[idx], score) for score, idx in scores],
            "aggregation_method": "Multi-Krum" if n >= (2 * self.f + 3) else "FedAvg"
        }
        self.aggregation_log.append(aggregation_info)

        logger.info(
            f"[Round {server_round}] Enhanced Multi-Krum selected clients: {[client_ids[i] for i in selected_indices]} "
            f"(from {client_ids}); top scores: {[(client_ids[idx], f'{score:.3f}') for score, idx in scores[:m]]}"
        )

        # Weighted average only over selected
        total_examples = sum(num_examples[idx] for idx in selected_indices)
        agg_flat = None
        for idx in selected_indices:
            weight = num_examples[idx] / \
                total_examples if total_examples > 0 else 1.0 / m
            vec = flat_updates[idx]
            agg_flat = vec * weight if agg_flat is None else agg_flat + vec * weight

        aggregated_weights = reconstruct_weights(agg_flat, template)
        aggregated_parameters = fl.common.ndarrays_to_parameters(
            aggregated_weights)

        # Aggregate metrics (simple average)
        aggregated_metrics: Dict[str, Scalar] = {
            "round": server_round,
            "selected_clients": str(selected_indices),
            "multi_krum_m": m,
            "multi_krum_f": self.f,
        }
        if metrics_list:
            # Average training (fit) accuracy - rename to explicitly reflect it's train-side
            train_accs = [m.get("accuracy")
                          for m in metrics_list if "accuracy" in m]
            if train_accs:
                avg_train_acc = float(np.mean(train_accs))
                aggregated_metrics["avg_client_train_accuracy"] = avg_train_acc
                self.history["train_accuracy"].append(avg_train_acc)

                # Update client performance history
                for i, (client_id, train_acc) in enumerate(zip(client_ids, train_accs)):
                    if client_id not in self.client_performance_history:
                        self.client_performance_history[client_id] = []
                    if train_acc is not None:
                        self.client_performance_history[client_id].append(
                            train_acc)

                # Keep length consistency if test not yet appended this round
                if len(self.history["test_accuracy"]) < len(self.history["train_accuracy"]):
                    # placeholder until evaluate phase
                    self.history["test_accuracy"].append(None)
                self._persist_history()

        # Save the global model for visualization analysis
        try:
            # Create a model with the aggregated weights
            os.makedirs("results", exist_ok=True)
            # Determine input features dynamically (30 in optimized mode)
            input_features = determine_input_features(default_if_missing=78)
            temp_model_wrapper = create_ddos_cnn_model(input_features=input_features)
            temp_model = temp_model_wrapper.model  # Get the actual Keras model
            temp_model.set_weights(aggregated_weights)
            model_path = "results/federated_global_model.keras"
            temp_model.save(model_path)
            logger.info(f"[Round {server_round}] Saved global model to {model_path}")
        except Exception as e:
            logger.warning(f"Could not save global model: {e}")
            import traceback
            traceback.print_exc()

        return aggregated_parameters, aggregated_metrics

    # ------------------------------------------------------------------
    # Override aggregate_evaluate to collect test accuracies separately
    # ------------------------------------------------------------------
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.client.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        losses: List[float] = []
        accs: List[float] = []
        for _, eval_res in results:
            losses.append(eval_res.loss)
            if eval_res.metrics and "accuracy" in eval_res.metrics:
                try:
                    accs.append(float(eval_res.metrics["accuracy"]))
                except (TypeError, ValueError):
                    pass

        aggregated_loss = float(np.mean(losses)) if losses else None
        metrics: Dict[str, Scalar] = {"round": server_round}
        if accs:
            avg_test_acc = float(np.mean(accs))
            metrics["avg_client_test_accuracy"] = avg_test_acc
            # Align placeholder None inserted earlier (replace last None) or append
            if self.history["test_accuracy"] and self.history["test_accuracy"][-1] is None:
                self.history["test_accuracy"][-1] = avg_test_acc
            else:
                # If for some reason train placeholder missing, append to maintain order
                self.history["test_accuracy"].append(avg_test_acc)
            # Ensure both lists same length
            if len(self.history["train_accuracy"]) < len(self.history["test_accuracy"]):
                self.history["train_accuracy"].append(
                    None)  # unusual but guard
            self._persist_history()

        if aggregated_loss is not None:
            metrics["avg_client_test_loss"] = aggregated_loss
        return aggregated_loss, metrics

# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Federated Server (Flower) with Multi-Krum")
    parser.add_argument('--address', type=str,
                        default='0.0.0.0:8080', help='Server host:port')
    parser.add_argument('--rounds', type=int, default=3,
                        help='Number of FL rounds')
    parser.add_argument('--f', type=int, default=0,
                        help='Assumed max Byzantine clients (set 0 for 4 clients)')
    parser.add_argument('--m', type=int, default=-1,
                        help='Number of selected updates (<= n - f - 2). -1 = auto')
    parser.add_argument('--min_fit', type=int, default=4,
                        help='Min fit clients')
    parser.add_argument('--min_eval', type=int, default=4,
                        help='Min eval clients')
    parser.add_argument('--min_available', type=int,
                        default=4, help='Min available clients')
    parser.add_argument('--initial_model', type=str, default='',
                        help='Path to pre-trained centralized model (.h5/.keras) to initialize global weights')
    args = parser.parse_args()

    # Only load initial parameters if a model path was explicitly provided.
    initial_parameters = get_initial_parameters(
        args.initial_model) if args.initial_model else None

    # Strategy
    strategy = MultiKrumFedAvg(
        f=args.f,
        m=None if args.m < 0 else args.m,
        min_fit_clients=args.min_fit,
        min_evaluate_clients=args.min_eval,
        min_available_clients=args.min_available,
        initial_parameters=initial_parameters,
        history_path="results/federated_metrics_history.json",
    )

    logger.info(
        f"Starting server on {args.address} rounds={args.rounds} f={args.f}")

    # Start server
    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    # Generate enhanced visualizations after training completion
    if VISUALIZATION_AVAILABLE:
        logger.info(
            "🎨 Generating federated analysis visualizations...")
        try:
            # Load global test data for confusion matrix and other analyses
            X_test, y_test = load_global_test_data()
            
            # Load the final global model to generate predictions
            global_model = None
            model_path = "results/federated_global_model.keras"
            if os.path.exists(model_path):
                try:
                    global_model = tf.keras.models.load_model(model_path)
                    logger.info(f"Loaded global model from {model_path}")
                except Exception as e:
                    logger.warning(f"Could not load global model: {e}")
            else:
                logger.warning(f"Global model not found at {model_path}")
            
            # Generate the 5 specific federated analysis visualizations
            plots = generate_federated_analysis_visualizations(
                federated_history_path="results/federated_metrics_history.json",
                global_model=global_model,
                X_test=X_test,
                y_test=y_test,
                results_dir="results/federated_analysis"
            )
            
            logger.info(
                "✅ Federated analysis visualization generation completed successfully!")
            if plots:
                file_paths = [p for p in plots.values() if isinstance(p, str)]
                logger.info(
                    f"📊 Generated {len(file_paths)} essential analysis plots (saved under results/federated_analysis):")
                for key, path in plots.items():
                    if isinstance(path, str):
                        logger.info(f"   - {key}: {Path(path).name}")
                logger.info("🔍 Essential analysis includes:")
                logger.info("   📊 Client Performance Metrics (Training/Testing Accuracy & Loss)")
                logger.info("   🎯 Client Confusion Matrices (CNN-based per client)")
                logger.info("   📈 Client ROC Curves (Client-based CNN performance)")
        except Exception as e:
            logger.error(f"❌ Error generating federated analysis visualizations: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.warning("⚠️ Federated analysis visualization module not available")


if __name__ == '__main__':
    main()
