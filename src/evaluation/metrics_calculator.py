"""
Metrics calculation engine for federated learning evaluation
Provides centralized calculation of all evaluation metrics
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix, classification_report
)
import logging

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Centralized calculation of all evaluation metrics"""
    
    def __init__(self):
        self.class_labels = ['Benign', 'Attack']
    
    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                       y_pred_proba: np.ndarray = None) -> Dict[str, Any]:
        """
        Calculate comprehensive classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dict containing all classification metrics
        """
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
                'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
                'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
                'f1_score_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
                'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
                'classification_report': classification_report(y_true, y_pred, 
                                                             target_names=self.class_labels, 
                                                             output_dict=True, zero_division=0)
            }
            
            # Add ROC-AUC if probabilities are provided
            if y_pred_proba is not None:
                roc_metrics = self.calculate_roc_metrics(y_true, y_pred_proba)
                metrics.update(roc_metrics)
                
                pr_metrics = self.calculate_precision_recall_metrics(y_true, y_pred_proba)
                metrics.update(pr_metrics)
            
            logger.debug(f"Calculated classification metrics: accuracy={metrics['accuracy']:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating classification metrics: {e}")
            return self._get_empty_metrics()
    
    def calculate_roc_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        Calculate ROC-AUC metrics for binary threat detection
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dict containing ROC metrics
        """
        try:
            # Handle different probability formats
            if y_pred_proba.ndim > 1:
                if y_pred_proba.shape[1] == 2:
                    # Binary classification with 2 columns
                    y_pred_proba = y_pred_proba[:, 1]
                else:
                    # Single column probabilities
                    y_pred_proba = y_pred_proba.flatten()
            
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            
            return {
                'roc_auc': roc_auc,
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'roc_thresholds': thresholds.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error calculating ROC metrics: {e}")
            return {
                'roc_auc': 0.0,
                'fpr': [],
                'tpr': [],
                'roc_thresholds': []
            }
    
    def calculate_precision_recall_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        Calculate precision-recall metrics for imbalanced datasets
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dict containing precision-recall metrics
        """
        try:
            # Handle different probability formats
            if y_pred_proba.ndim > 1:
                if y_pred_proba.shape[1] == 2:
                    y_pred_proba = y_pred_proba[:, 1]
                else:
                    y_pred_proba = y_pred_proba.flatten()
            
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
            avg_precision = average_precision_score(y_true, y_pred_proba)
            
            return {
                'average_precision': avg_precision,
                'precision_curve': precision.tolist(),
                'recall_curve': recall.tolist(),
                'pr_thresholds': thresholds.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error calculating precision-recall metrics: {e}")
            return {
                'average_precision': 0.0,
                'precision_curve': [],
                'recall_curve': [],
                'pr_thresholds': []
            }
    
    def aggregate_client_metrics(self, client_metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Aggregate metrics from multiple clients
        
        Args:
            client_metrics: Dictionary mapping client_id to metrics dict
            
        Returns:
            Dict containing aggregated metrics
        """
        if not client_metrics:
            return self._get_empty_metrics()
        
        try:
            # Collect all metrics
            all_accuracies = []
            all_precisions = []
            all_recalls = []
            all_f1_scores = []
            all_roc_aucs = []
            all_avg_precisions = []
            
            # Aggregate confusion matrices
            total_cm = None
            
            for client_id, metrics in client_metrics.items():
                if metrics:
                    all_accuracies.append(metrics.get('accuracy', 0))
                    all_precisions.append(metrics.get('precision', 0))
                    all_recalls.append(metrics.get('recall', 0))
                    all_f1_scores.append(metrics.get('f1_score', 0))
                    all_roc_aucs.append(metrics.get('roc_auc', 0))
                    all_avg_precisions.append(metrics.get('average_precision', 0))
                    
                    # Sum confusion matrices
                    if 'confusion_matrix' in metrics:
                        cm = np.array(metrics['confusion_matrix'])
                        if total_cm is None:
                            total_cm = cm
                        else:
                            total_cm += cm
            
            # Calculate aggregated metrics
            aggregated = {
                'num_clients': len(client_metrics),
                'accuracy_mean': np.mean(all_accuracies) if all_accuracies else 0,
                'accuracy_std': np.std(all_accuracies) if all_accuracies else 0,
                'precision_mean': np.mean(all_precisions) if all_precisions else 0,
                'precision_std': np.std(all_precisions) if all_precisions else 0,
                'recall_mean': np.mean(all_recalls) if all_recalls else 0,
                'recall_std': np.std(all_recalls) if all_recalls else 0,
                'f1_score_mean': np.mean(all_f1_scores) if all_f1_scores else 0,
                'f1_score_std': np.std(all_f1_scores) if all_f1_scores else 0,
                'roc_auc_mean': np.mean(all_roc_aucs) if all_roc_aucs else 0,
                'roc_auc_std': np.std(all_roc_aucs) if all_roc_aucs else 0,
                'avg_precision_mean': np.mean(all_avg_precisions) if all_avg_precisions else 0,
                'avg_precision_std': np.std(all_avg_precisions) if all_avg_precisions else 0,
                'aggregated_confusion_matrix': total_cm.tolist() if total_cm is not None else [[0, 0], [0, 0]]
            }
            
            logger.info(f"Aggregated metrics from {len(client_metrics)} clients")
            return aggregated
            
        except Exception as e:
            logger.error(f"Error aggregating client metrics: {e}")
            return self._get_empty_metrics()
    
    def validate_prediction_data(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               y_pred_proba: np.ndarray = None) -> Tuple[bool, str]:
        """
        Validate prediction data shapes and ranges
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check shapes
            if len(y_true) != len(y_pred):
                return False, f"Shape mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
            
            if y_pred_proba is not None:
                if len(y_true) != len(y_pred_proba):
                    return False, f"Shape mismatch: y_true={len(y_true)}, y_pred_proba={len(y_pred_proba)}"
            
            # Check value ranges
            unique_true = np.unique(y_true)
            unique_pred = np.unique(y_pred)
            
            if not all(val in [0, 1] for val in unique_true):
                return False, f"Invalid y_true values: {unique_true}. Expected [0, 1]"
            
            if not all(val in [0, 1] for val in unique_pred):
                return False, f"Invalid y_pred values: {unique_pred}. Expected [0, 1]"
            
            if y_pred_proba is not None:
                if y_pred_proba.ndim > 1:
                    proba_flat = y_pred_proba.flatten()
                else:
                    proba_flat = y_pred_proba
                
                if np.any(proba_flat < 0) or np.any(proba_flat > 1):
                    return False, "Probabilities must be in range [0, 1]"
            
            return True, "Validation passed"
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics dictionary for error cases"""
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'roc_auc': 0.0,
            'average_precision': 0.0,
            'confusion_matrix': [[0, 0], [0, 0]],
            'classification_report': {}
        }