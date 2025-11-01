"""
Evaluation visualizer for model evaluation visualizations
Handles confusion matrices, ROC curves, and precision-recall curves
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Dict, Any, Optional
import logging

# Import our utilities
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.visualization_utils import (
    apply_consistent_styling, save_plot_with_metadata, 
    format_classification_table, get_class_labels, ensure_directory_exists
)

logger = logging.getLogger(__name__)

class EvaluationVisualizer:
    """Handle model evaluation visualizations with focus on classification metrics"""
    
    def __init__(self):
        self.class_labels = get_class_labels()
        apply_consistent_styling()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            title: str, output_path: str, 
                            labels: List[str] = None) -> str:
        """
        Generate confusion matrix visualization for individual clients or aggregated results
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Title for the plot
            output_path: Path to save the plot
            labels: Class labels (optional)
            
        Returns:
            str: Path where plot was saved
        """
        try:
            if labels is None:
                labels = self.class_labels
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels,
                       cbar_kws={'label': 'Count'}, ax=ax)
            
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            
            # Add percentage annotations
            cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j+0.5, i+0.7, f'({cm_percentage[i, j]:.1f}%)', 
                           ha='center', va='center', fontsize=10, color='darkred')
            
            plt.tight_layout()
            
            # Save with metadata
            metadata = {
                'title': title,
                'confusion_matrix': cm.tolist(),
                'class_labels': labels,
                'total_samples': int(cm.sum())
            }
            
            ensure_directory_exists(output_path)
            saved_path = save_plot_with_metadata(fig, output_path, metadata)
            
            logger.info(f"Confusion matrix saved: {saved_path}")
            return saved_path
            
        except Exception as e:
            logger.error(f"Error creating confusion matrix: {e}")
            return ""
    
    def plot_aggregated_confusion_matrix(self, client_confusion_matrices: List[np.ndarray],
                                       output_path: str, client_ids: List[str] = None) -> str:
        """
        Generate aggregated confusion matrix combining all client results
        
        Args:
            client_confusion_matrices: List of confusion matrices from clients
            output_path: Path to save the plot
            client_ids: List of client identifiers
            
        Returns:
            str: Path where plot was saved
        """
        try:
            # Sum all confusion matrices
            total_cm = np.sum(client_confusion_matrices, axis=0)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Create heatmap
            sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_labels, yticklabels=self.class_labels,
                       cbar_kws={'label': 'Count'}, ax=ax)
            
            num_clients = len(client_confusion_matrices)
            title = f'Aggregated Confusion Matrix ({num_clients} Clients)'
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            
            # Add percentage annotations
            cm_percentage = total_cm.astype('float') / total_cm.sum(axis=1)[:, np.newaxis] * 100
            for i in range(total_cm.shape[0]):
                for j in range(total_cm.shape[1]):
                    ax.text(j+0.5, i+0.7, f'({cm_percentage[i, j]:.1f}%)', 
                           ha='center', va='center', fontsize=10, color='darkred')
            
            plt.tight_layout()
            
            # Save with metadata
            metadata = {
                'title': title,
                'aggregated_confusion_matrix': total_cm.tolist(),
                'num_clients': num_clients,
                'client_ids': client_ids or [f'Client_{i}' for i in range(num_clients)],
                'total_samples': int(total_cm.sum())
            }
            
            ensure_directory_exists(output_path)
            saved_path = save_plot_with_metadata(fig, output_path, metadata)
            
            logger.info(f"Aggregated confusion matrix saved: {saved_path}")
            return saved_path
            
        except Exception as e:
            logger.error(f"Error creating aggregated confusion matrix: {e}")
            return ""
    
    def print_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  labels: List[str] = None, title: str = "Classification Report") -> str:
        """
        Generate and print classification report in tabular format
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels (optional)
            title: Title for the report
            
        Returns:
            str: Formatted classification report string
        """
        try:
            if labels is None:
                labels = self.class_labels
            
            # Generate classification report
            report_dict = classification_report(y_true, y_pred, target_names=labels, 
                                              output_dict=True, zero_division=0)
            
            # Format as table
            formatted_report = format_classification_table(report_dict)
            
            # Add title
            title_line = f"\n{title}\n"
            full_report = title_line + formatted_report
            
            # Print to console
            print(full_report)
            
            logger.info(f"Classification report generated: {title}")
            return full_report
            
        except Exception as e:
            logger.error(f"Error generating classification report: {e}")
            return f"Error generating classification report: {e}"
    
    def save_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 output_path: str, labels: List[str] = None, 
                                 title: str = "Classification Report") -> str:
        """
        Save classification report to text file
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_path: Path to save the report
            labels: Class labels (optional)
            title: Title for the report
            
        Returns:
            str: Path where report was saved
        """
        try:
            # Generate formatted report
            formatted_report = self.print_classification_report(y_true, y_pred, labels, title)
            
            # Save to file
            ensure_directory_exists(output_path)
            with open(output_path, 'w') as f:
                f.write(formatted_report)
            
            logger.info(f"Classification report saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving classification report: {e}")
            return ""
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                      title: str, output_path: str, auc_score: float = None) -> str:
        """
        Generate ROC-AUC curve for binary threat detection
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            title: Title for the plot
            output_path: Path to save the plot
            auc_score: Pre-calculated AUC score (optional)
            
        Returns:
            str: Path where plot was saved
        """
        try:
            from sklearn.metrics import roc_curve, auc
            
            # Handle different probability formats
            if y_pred_proba.ndim > 1:
                if y_pred_proba.shape[1] == 2:
                    y_pred_proba = y_pred_proba[:, 1]
                else:
                    y_pred_proba = y_pred_proba.flatten()
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            if auc_score is None:
                auc_score = auc(fpr, tpr)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot ROC curve
            ax.plot(fpr, tpr, color='darkorange', lw=2,
                   label=f'ROC curve (AUC = {auc_score:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                   label='Random Classifier')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save with metadata
            metadata = {
                'title': title,
                'auc_score': float(auc_score),
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            }
            
            ensure_directory_exists(output_path)
            saved_path = save_plot_with_metadata(fig, output_path, metadata)
            
            logger.info(f"ROC curve saved: {saved_path}")
            return saved_path
            
        except Exception as e:
            logger.error(f"Error creating ROC curve: {e}")
            return ""
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                  title: str, output_path: str, 
                                  avg_precision: float = None) -> str:
        """
        Generate precision-recall curve for imbalanced threat datasets
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            title: Title for the plot
            output_path: Path to save the plot
            avg_precision: Pre-calculated average precision (optional)
            
        Returns:
            str: Path where plot was saved
        """
        try:
            from sklearn.metrics import precision_recall_curve, average_precision_score
            
            # Handle different probability formats
            if y_pred_proba.ndim > 1:
                if y_pred_proba.shape[1] == 2:
                    y_pred_proba = y_pred_proba[:, 1]
                else:
                    y_pred_proba = y_pred_proba.flatten()
            
            # Calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            if avg_precision is None:
                avg_precision = average_precision_score(y_true, y_pred_proba)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot precision-recall curve
            ax.plot(recall, precision, color='blue', lw=2,
                   label=f'PR curve (AP = {avg_precision:.3f})')
            
            # Add baseline (random classifier for imbalanced data)
            baseline = np.sum(y_true) / len(y_true)
            ax.axhline(y=baseline, color='red', linestyle='--', alpha=0.7,
                      label=f'Random Classifier (AP = {baseline:.3f})')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Recall', fontsize=12)
            ax.set_ylabel('Precision', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc="lower left")
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save with metadata
            metadata = {
                'title': title,
                'average_precision': float(avg_precision),
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'baseline': float(baseline)
            }
            
            ensure_directory_exists(output_path)
            saved_path = save_plot_with_metadata(fig, output_path, metadata)
            
            logger.info(f"Precision-recall curve saved: {saved_path}")
            return saved_path
            
        except Exception as e:
            logger.error(f"Error creating precision-recall curve: {e}")
            return ""