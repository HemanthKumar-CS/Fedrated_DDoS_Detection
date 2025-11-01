"""
Enhanced Federated Visualizer - Main class integrating all visualization components
"""

import os
import numpy as np
from typing import Dict, Any
import logging

# Import our components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .evaluation_visualizer import EvaluationVisualizer
from .federated_visualizer import FederatedVisualizer
from ..evaluation.metrics_calculator import MetricsCalculator
from ..utils.visualization_utils import create_output_directory

logger = logging.getLogger(__name__)

class EnhancedFederatedVisualizer:
    """Main class integrating all visualization components for federated learning"""
    
    def __init__(self):
        self.eval_visualizer = EvaluationVisualizer()
        self.fed_visualizer = FederatedVisualizer()
        self.metrics_calculator = MetricsCalculator()
        
        logger.info("Enhanced Federated Visualizer initialized")
    
    def generate_all_visualizations(self, client_predictions: Dict[str, Dict], 
                                  aggregated_predictions: Dict,
                                  federated_history: Dict,
                                  output_dir: str = "results/validations") -> Dict[str, str]:
        """
        Main entry point for generating all enhanced federated visualizations
        
        Args:
            client_predictions: Dict mapping client_id to prediction data
            aggregated_predictions: Aggregated prediction data
            federated_history: Federated training history
            output_dir: Base output directory
            
        Returns:
            Dict mapping visualization type to file paths
        """
        try:
            # Create organized output directory
            output_base = create_output_directory(output_dir, timestamp=True)
            
            generated_files = {}
            
            # 1. Generate confusion matrices
            confusion_files = self.generate_confusion_matrices(
                client_predictions, aggregated_predictions, output_base)
            generated_files.update(confusion_files)
            
            # 2. Generate classification reports
            report_files = self.generate_classification_reports(
                client_predictions, aggregated_predictions, output_base)
            generated_files.update(report_files)
            
            # 3. Generate ROC curves
            roc_files = self.generate_roc_curves(
                client_predictions, aggregated_predictions, output_base)
            generated_files.update(roc_files)
            
            # 4. Generate precision-recall curves
            pr_files = self.generate_precision_recall_curves(
                client_predictions, aggregated_predictions, output_base)
            generated_files.update(pr_files)
            
            # 5. Generate training progress plots
            training_files = self.generate_training_progress_plots(
                federated_history, output_base, client_predictions)
            generated_files.update(training_files)
            
            logger.info(f"Generated {len(generated_files)} visualization files in {output_base}")
            return generated_files
            
        except Exception as e:
            logger.error(f"Error generating all visualizations: {e}")
            return {}
    
    def generate_confusion_matrices(self, client_predictions: Dict[str, Dict], 
                                  aggregated_predictions: Dict, output_base: str) -> Dict[str, str]:
        """Generate confusion matrices for all clients and aggregated results"""
        files = {}
        
        try:
            # Generate per-client confusion matrices
            client_cms = []
            for client_id, pred_data in client_predictions.items():
                if self._validate_prediction_data(pred_data):
                    output_path = os.path.join(output_base, "confusion_matrices", 
                                             f"{client_id}_confusion_matrix.png")
                    
                    saved_path = self.eval_visualizer.plot_confusion_matrix(
                        pred_data['y_true'], pred_data['y_pred'],
                        f"Confusion Matrix - {client_id}", output_path)
                    
                    if saved_path:
                        files[f"confusion_matrix_{client_id}"] = saved_path
                        
                        # Calculate confusion matrix for aggregation
                        from sklearn.metrics import confusion_matrix
                        cm = confusion_matrix(pred_data['y_true'], pred_data['y_pred'])
                        client_cms.append(cm)
            
            # Generate aggregated confusion matrix
            if client_cms and aggregated_predictions and self._validate_prediction_data(aggregated_predictions):
                output_path = os.path.join(output_base, "confusion_matrices", 
                                         "aggregated_confusion_matrix.png")
                
                saved_path = self.eval_visualizer.plot_aggregated_confusion_matrix(
                    client_cms, output_path, list(client_predictions.keys()))
                
                if saved_path:
                    files["confusion_matrix_aggregated"] = saved_path
            
            logger.info(f"Generated {len(files)} confusion matrix files")
            return files
            
        except Exception as e:
            logger.error(f"Error generating confusion matrices: {e}")
            return {}
    
    def generate_classification_reports(self, client_predictions: Dict[str, Dict], 
                                      aggregated_predictions: Dict, output_base: str) -> Dict[str, str]:
        """Generate classification reports for all clients and aggregated results"""
        files = {}
        
        try:
            # Generate per-client classification reports
            for client_id, pred_data in client_predictions.items():
                if self._validate_prediction_data(pred_data):
                    # Print to console
                    self.eval_visualizer.print_classification_report(
                        pred_data['y_true'], pred_data['y_pred'], 
                        title=f"Classification Report - {client_id}")
                    
                    # Save to file
                    output_path = os.path.join(output_base, "classification_reports", 
                                             f"{client_id}_classification_report.txt")
                    
                    saved_path = self.eval_visualizer.save_classification_report(
                        pred_data['y_true'], pred_data['y_pred'], output_path,
                        title=f"Classification Report - {client_id}")
                    
                    if saved_path:
                        files[f"classification_report_{client_id}"] = saved_path
            
            # Generate aggregated classification report
            if aggregated_predictions and self._validate_prediction_data(aggregated_predictions):
                # Print to console
                self.eval_visualizer.print_classification_report(
                    aggregated_predictions['y_true'], aggregated_predictions['y_pred'],
                    title="Aggregated Classification Report")
                
                # Save to file
                output_path = os.path.join(output_base, "classification_reports", 
                                         "aggregated_classification_report.txt")
                
                saved_path = self.eval_visualizer.save_classification_report(
                    aggregated_predictions['y_true'], aggregated_predictions['y_pred'], 
                    output_path, title="Aggregated Classification Report")
                
                if saved_path:
                    files["classification_report_aggregated"] = saved_path
            
            logger.info(f"Generated {len(files)} classification report files")
            return files
            
        except Exception as e:
            logger.error(f"Error generating classification reports: {e}")
            return {}
    
    def generate_roc_curves(self, client_predictions: Dict[str, Dict], 
                          aggregated_predictions: Dict, output_base: str) -> Dict[str, str]:
        """Generate ROC-AUC curves for all clients and aggregated results"""
        files = {}
        
        try:
            # Generate per-client ROC curves
            for client_id, pred_data in client_predictions.items():
                if self._validate_prediction_data(pred_data, require_proba=True):
                    output_path = os.path.join(output_base, "roc_curves", 
                                             f"{client_id}_roc_curve.png")
                    
                    saved_path = self.eval_visualizer.plot_roc_curve(
                        pred_data['y_true'], pred_data['y_pred_proba'],
                        f"ROC Curve - {client_id}", output_path,
                        pred_data.get('roc_auc'))
                    
                    if saved_path:
                        files[f"roc_curve_{client_id}"] = saved_path
            
            # Generate aggregated ROC curve
            if aggregated_predictions and self._validate_prediction_data(aggregated_predictions, require_proba=True):
                output_path = os.path.join(output_base, "roc_curves", 
                                         "aggregated_roc_curve.png")
                
                saved_path = self.eval_visualizer.plot_roc_curve(
                    aggregated_predictions['y_true'], aggregated_predictions['y_pred_proba'],
                    "Aggregated ROC Curve", output_path,
                    aggregated_predictions.get('roc_auc'))
                
                if saved_path:
                    files["roc_curve_aggregated"] = saved_path
            
            logger.info(f"Generated {len(files)} ROC curve files")
            return files
            
        except Exception as e:
            logger.error(f"Error generating ROC curves: {e}")
            return {}
    
    def generate_precision_recall_curves(self, client_predictions: Dict[str, Dict], 
                                       aggregated_predictions: Dict, output_base: str) -> Dict[str, str]:
        """Generate precision-recall curves for all clients and aggregated results"""
        files = {}
        
        try:
            # Generate per-client precision-recall curves
            for client_id, pred_data in client_predictions.items():
                if self._validate_prediction_data(pred_data, require_proba=True):
                    output_path = os.path.join(output_base, "precision_recall_curves", 
                                             f"{client_id}_pr_curve.png")
                    
                    saved_path = self.eval_visualizer.plot_precision_recall_curve(
                        pred_data['y_true'], pred_data['y_pred_proba'],
                        f"Precision-Recall Curve - {client_id}", output_path,
                        pred_data.get('average_precision'))
                    
                    if saved_path:
                        files[f"pr_curve_{client_id}"] = saved_path
            
            # Generate aggregated precision-recall curve
            if aggregated_predictions and self._validate_prediction_data(aggregated_predictions, require_proba=True):
                output_path = os.path.join(output_base, "precision_recall_curves", 
                                         "aggregated_pr_curve.png")
                
                saved_path = self.eval_visualizer.plot_precision_recall_curve(
                    aggregated_predictions['y_true'], aggregated_predictions['y_pred_proba'],
                    "Aggregated Precision-Recall Curve", output_path,
                    aggregated_predictions.get('average_precision'))
                
                if saved_path:
                    files["pr_curve_aggregated"] = saved_path
            
            logger.info(f"Generated {len(files)} precision-recall curve files")
            return files
            
        except Exception as e:
            logger.error(f"Error generating precision-recall curves: {e}")
            return {}
    
    def generate_training_progress_plots(self, federated_history: Dict, output_base: str,
                                       client_predictions: Dict = None) -> Dict[str, str]:
        """Generate training progress visualizations"""
        files = {}
        
        try:
            # Extract client histories if available
            client_histories = {}
            if client_predictions:
                for client_id in client_predictions.keys():
                    # Mock client histories for now - in real implementation, 
                    # these would come from actual training logs
                    client_histories[client_id] = {
                        'train_accuracy': federated_history.get('train_accuracy', []),
                        'test_accuracy': federated_history.get('test_accuracy', []),
                        'train_loss': federated_history.get('train_loss', []),
                        'test_loss': federated_history.get('test_loss', [])
                    }
            
            # Generate training vs test accuracy plot
            output_path = os.path.join(output_base, "training_progress", 
                                     "training_vs_test_accuracy.png")
            
            saved_path = self.fed_visualizer.plot_training_vs_test_accuracy(
                federated_history, output_path, client_histories)
            
            if saved_path:
                files["training_vs_test_accuracy"] = saved_path
            
            # Generate training vs test loss plot
            output_path = os.path.join(output_base, "training_progress", 
                                     "training_vs_test_loss.png")
            
            saved_path = self.fed_visualizer.plot_training_vs_test_loss(
                federated_history, output_path, client_histories)
            
            if saved_path:
                files["training_vs_test_loss"] = saved_path
            
            # Generate client performance comparison
            if client_predictions:
                client_metrics = {}
                for client_id, pred_data in client_predictions.items():
                    if self._validate_prediction_data(pred_data):
                        client_metrics[client_id] = self.metrics_calculator.calculate_classification_metrics(
                            pred_data['y_true'], pred_data['y_pred'], 
                            pred_data.get('y_pred_proba'))
                
                if client_metrics:
                    output_path = os.path.join(output_base, "training_progress", 
                                             "client_performance_comparison.png")
                    
                    saved_path = self.fed_visualizer.plot_client_performance_comparison(
                        client_metrics, output_path)
                    
                    if saved_path:
                        files["client_performance_comparison"] = saved_path
            
            # Generate convergence analysis
            output_path = os.path.join(output_base, "training_progress", 
                                     "convergence_analysis.png")
            
            saved_path = self.fed_visualizer.plot_convergence_analysis(
                federated_history, output_path)
            
            if saved_path:
                files["convergence_analysis"] = saved_path
            
            logger.info(f"Generated {len(files)} training progress files")
            return files
            
        except Exception as e:
            logger.error(f"Error generating training progress plots: {e}")
            return {}
    
    def _validate_prediction_data(self, pred_data: Dict, require_proba: bool = False) -> bool:
        """Validate prediction data structure and content"""
        try:
            if not pred_data:
                return False
            
            required_keys = ['y_true', 'y_pred']
            if require_proba:
                required_keys.append('y_pred_proba')
            
            for key in required_keys:
                if key not in pred_data or pred_data[key] is None:
                    logger.warning(f"Missing required key: {key}")
                    return False
            
            # Validate with metrics calculator
            is_valid, error_msg = self.metrics_calculator.validate_prediction_data(
                pred_data['y_true'], pred_data['y_pred'], 
                pred_data.get('y_pred_proba'))
            
            if not is_valid:
                logger.warning(f"Prediction data validation failed: {error_msg}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating prediction data: {e}")
            return False


# Convenience function for easy access
def generate_enhanced_federated_visualizations(
    federated_history: Dict,
    client_predictions: Dict[str, Dict],
    global_predictions: Dict,
    output_dir: str = "results/validations"
) -> Dict[str, str]:
    """
    Generate all enhanced federated visualizations
    
    Args:
        federated_history: Dictionary containing federated training history
        client_predictions: Dictionary mapping client_id to prediction data
        global_predictions: Global/aggregated prediction data
        output_dir: Directory to save results
        
    Returns:
        Dictionary with generated file paths
    """
    visualizer = EnhancedFederatedVisualizer()
    return visualizer.generate_all_visualizations(
        client_predictions, global_predictions, federated_history, output_dir)