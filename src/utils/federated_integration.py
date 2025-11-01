"""
Integration utilities for connecting enhanced visualizations with existing federated training pipeline
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

def extract_federated_history_from_flower(flower_history) -> Dict[str, List[float]]:
    """
    Extract federated training history from Flower simulation results
    
    Args:
        flower_history: Flower simulation history object
        
    Returns:
        Dict containing formatted training history
    """
    try:
        federated_history = {
            'train_accuracy': [],
            'test_accuracy': [],
            'train_loss': [],
            'test_loss': []
        }
        
        # Extract distributed metrics (client-side)
        if hasattr(flower_history, 'metrics_distributed') and flower_history.metrics_distributed:
            # Extract accuracy from distributed metrics
            if 'accuracy' in flower_history.metrics_distributed:
                for round_num, (_, accuracy) in enumerate(flower_history.metrics_distributed['accuracy']):
                    federated_history['test_accuracy'].append(accuracy)
        
        # Extract distributed losses
        if hasattr(flower_history, 'losses_distributed') and flower_history.losses_distributed:
            for round_num, (_, loss) in enumerate(flower_history.losses_distributed):
                federated_history['test_loss'].append(loss)
        
        # Extract centralized metrics (server-side evaluation)
        if hasattr(flower_history, 'metrics_centralized') and flower_history.metrics_centralized:
            if 'accuracy' in flower_history.metrics_centralized:
                for round_num, (_, accuracy) in enumerate(flower_history.metrics_centralized['accuracy']):
                    federated_history['train_accuracy'].append(accuracy)
        
        # Extract centralized losses
        if hasattr(flower_history, 'losses_centralized') and flower_history.losses_centralized:
            for round_num, (_, loss) in enumerate(flower_history.losses_centralized):
                federated_history['train_loss'].append(loss)
        
        # Fill missing data with reasonable defaults if needed
        max_rounds = max(
            len(federated_history['train_accuracy']),
            len(federated_history['test_accuracy']),
            len(federated_history['train_loss']),
            len(federated_history['test_loss'])
        )
        
        for key in federated_history:
            while len(federated_history[key]) < max_rounds:
                if 'accuracy' in key:
                    federated_history[key].append(0.5)  # Default accuracy
                else:
                    federated_history[key].append(1.0)  # Default loss
        
        logger.info(f"Extracted federated history: {max_rounds} rounds")
        return federated_history
        
    except Exception as e:
        logger.error(f"Error extracting federated history: {e}")
        return {
            'train_accuracy': [0.5],
            'test_accuracy': [0.5],
            'train_loss': [1.0],
            'test_loss': [1.0]
        }

def load_client_data_for_evaluation(client_id: int, data_dir: str = "data/optimized") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load client data for evaluation (compatible with existing federated_training.py)
    
    Args:
        client_id: Client identifier
        data_dir: Directory containing client data files
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    try:
        # Load client-specific data
        train_path = os.path.join(data_dir, f"client_{client_id}_train.csv")
        test_path = os.path.join(data_dir, f"client_{client_id}_test.csv")
        
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
        
        logger.debug(f"Client {client_id} data loaded: Train={X_train.shape}, Test={X_test.shape}")
        
        return X_train, y_train, X_test, y_test
        
    except Exception as e:
        logger.error(f"Error loading client {client_id} data: {e}")
        raise

def extract_client_predictions_from_model(model, num_clients: int = 4, 
                                        data_dir: str = "data/optimized") -> Dict[str, Dict]:
    """
    Extract predictions from trained model for all clients
    
    Args:
        model: Trained federated model
        num_clients: Number of clients
        data_dir: Directory containing client data
        
    Returns:
        Dict mapping client_id to prediction data
    """
    try:
        client_predictions = {}
        
        for client_id in range(num_clients):
            # Load client test data
            _, _, X_test, y_test = load_client_data_for_evaluation(client_id, data_dir)
            
            # Get model predictions
            y_pred_proba = model.predict(X_test, verbose=0)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            
            # Handle different probability formats
            if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] == 1:
                y_pred_proba = y_pred_proba.flatten()
            
            client_predictions[f"client_{client_id}"] = {
                'y_true': y_test.astype(int),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            logger.debug(f"Client {client_id} predictions extracted: {len(y_test)} samples")
        
        logger.info(f"Extracted predictions for {num_clients} clients")
        return client_predictions
        
    except Exception as e:
        logger.error(f"Error extracting client predictions: {e}")
        return {}

def create_aggregated_predictions(client_predictions: Dict[str, Dict]) -> Dict[str, np.ndarray]:
    """
    Create aggregated predictions from all clients
    
    Args:
        client_predictions: Dict mapping client_id to prediction data
        
    Returns:
        Dict containing aggregated prediction data
    """
    try:
        if not client_predictions:
            return {}
        
        all_y_true = []
        all_y_pred = []
        all_y_pred_proba = []
        
        for client_id, pred_data in client_predictions.items():
            all_y_true.extend(pred_data['y_true'])
            all_y_pred.extend(pred_data['y_pred'])
            all_y_pred_proba.extend(pred_data['y_pred_proba'])
        
        aggregated_predictions = {
            'y_true': np.array(all_y_true, dtype=int),
            'y_pred': np.array(all_y_pred, dtype=int),
            'y_pred_proba': np.array(all_y_pred_proba)
        }
        
        logger.info(f"Created aggregated predictions: {len(all_y_true)} total samples")
        return aggregated_predictions
        
    except Exception as e:
        logger.error(f"Error creating aggregated predictions: {e}")
        return {}

def integrate_enhanced_visualizations_with_federated_training(flower_history, 
                                                           final_model,
                                                           num_clients: int = 4,
                                                           output_dir: str = "results/validations") -> Dict[str, str]:
    """
    Complete integration function for adding enhanced visualizations to federated training
    
    Args:
        flower_history: Flower simulation history object
        final_model: Final trained federated model
        num_clients: Number of federated clients
        output_dir: Output directory for visualizations
        
    Returns:
        Dict mapping visualization types to file paths
    """
    try:
        logger.info("🎨 Integrating enhanced visualizations with federated training")
        
        # 1. Extract federated training history
        federated_history = extract_federated_history_from_flower(flower_history)
        
        # 2. Extract client predictions
        client_predictions = extract_client_predictions_from_model(final_model, num_clients)
        
        # 3. Create aggregated predictions
        aggregated_predictions = create_aggregated_predictions(client_predictions)
        
        # 4. Generate enhanced visualizations
        from ..visualization.enhanced_federated_visualizer import generate_enhanced_federated_visualizations
        
        generated_files = generate_enhanced_federated_visualizations(
            federated_history=federated_history,
            client_predictions=client_predictions,
            global_predictions=aggregated_predictions,
            output_dir=output_dir
        )
        
        # 5. Generate comprehensive reports
        from ..evaluation.report_generator import ReportGenerator
        from ..evaluation.metrics_calculator import MetricsCalculator
        
        metrics_calc = MetricsCalculator()
        report_gen = ReportGenerator()
        
        # Calculate metrics
        client_metrics = {}
        for client_id, pred_data in client_predictions.items():
            metrics = metrics_calc.calculate_classification_metrics(
                pred_data['y_true'], pred_data['y_pred'], pred_data['y_pred_proba'])
            client_metrics[client_id] = metrics
        
        cross_client_metrics = metrics_calc.aggregate_client_metrics(client_metrics)
        
        # Generate reports
        if generated_files:
            sample_path = list(generated_files.values())[0]
            output_base = os.path.dirname(os.path.dirname(sample_path))
            
            report_path = report_gen.save_comprehensive_report(
                output_base, cross_client_metrics, client_metrics, generated_files)
            
            organized_files = report_gen.organize_output_files(output_base, generated_files)
            
            generated_files.update(organized_files)
            generated_files['comprehensive_report'] = report_path
        
        logger.info(f"✅ Enhanced visualization integration complete: {len(generated_files)} files generated")
        return generated_files
        
    except Exception as e:
        logger.error(f"❌ Error in enhanced visualization integration: {e}")
        return {}

def get_final_model_from_flower_simulation(flower_history, strategy) -> tf.keras.Model:
    """
    Extract the final global model from Flower simulation
    
    Args:
        flower_history: Flower simulation history
        strategy: Flower strategy object
        
    Returns:
        Final trained model
    """
    try:
        # This is a placeholder - actual implementation depends on your Flower setup
        # You would typically get the final parameters from the strategy and 
        # reconstruct the model
        
        logger.warning("get_final_model_from_flower_simulation is a placeholder - implement based on your Flower setup")
        
        # For now, try to load the most recent saved model
        model_paths = [
            "results/best_enhanced_model.keras",
            "results/best_model.h5",
            "models/federated_model.h5"
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                logger.info(f"Loading model from {path}")
                return tf.keras.models.load_model(path)
        
        # If no saved model found, create a basic model structure
        logger.warning("No saved model found, creating basic model structure")
        from ..models.cnn_model import create_ddos_cnn_model
        
        # Determine input shape from client data
        try:
            _, _, X_test, _ = load_client_data_for_evaluation(0)
            input_shape = (X_test.shape[1], 1)
        except:
            input_shape = (77, 1)  # Default shape
        
        return create_ddos_cnn_model(input_shape)
        
    except Exception as e:
        logger.error(f"Error getting final model: {e}")
        raise