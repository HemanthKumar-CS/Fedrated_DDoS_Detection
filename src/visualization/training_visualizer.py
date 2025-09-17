import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import pandas as pd
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Any


class FederatedTrainingVisualizer:
    """Comprehensive visualization suite for federated DDoS detection training."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.viz_dir = self.results_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def generate_all_plots(self,
                           model=None,
                           history=None,
                           X_test=None,
                           y_test=None,
                           federated_history_path: Optional[str] = None):
        """Generate all visualization plots for comprehensive reporting."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plots_generated = []

        try:
            # 1. Training History Plots
            if history is not None:
                plots_generated.extend(
                    self._plot_training_history(history, timestamp))

            # 2. Model Performance Plots
            if model is not None and X_test is not None and y_test is not None:
                plots_generated.extend(self._plot_model_performance(
                    model, X_test, y_test, timestamp))

            # 3. Federated Learning Progress
            if federated_history_path and Path(federated_history_path).exists():
                plots_generated.extend(self._plot_federated_progress(
                    federated_history_path, timestamp))

            # 4. Comprehensive Summary Plot
            plots_generated.append(self._create_summary_dashboard(timestamp))

            print(
                f"✅ Generated {len(plots_generated)} visualization plots in {self.viz_dir}")
            return plots_generated

        except Exception as e:
            print(f"❌ Error generating plots: {e}")
            return []

    def _plot_training_history(self, history, timestamp: str) -> List[str]:
        """Plot training/validation curves."""
        plots = []

        # Training/Validation Loss and Accuracy
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curves
        ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in history.history:
            ax1.plot(history.history['val_loss'],
                     label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy curves
        ax2.plot(history.history['accuracy'],
                 label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in history.history:
            ax2.plot(history.history['val_accuracy'],
                     label='Validation Accuracy', linewidth=2)
        ax2.set_title('Model Accuracy Over Epochs',
                      fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Learning rate (if available)
        if hasattr(history, 'lr') or 'lr' in history.history:
            lr_data = history.history.get('lr', [])
            if lr_data:
                ax3.plot(lr_data, label='Learning Rate',
                         color='red', linewidth=2)
                ax3.set_title('Learning Rate Schedule',
                              fontsize=14, fontweight='bold')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Learning Rate')
                ax3.set_yscale('log')
                ax3.grid(True, alpha=0.3)

        # Precision/Recall if available
        if 'precision' in history.history and 'recall' in history.history:
            ax4.plot(history.history['precision'],
                     label='Precision', linewidth=2)
            ax4.plot(history.history['recall'], label='Recall', linewidth=2)
            ax4.set_title('Precision/Recall Over Epochs',
                          fontsize=14, fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Score')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f"training_history_{timestamp}.png"
        filepath = self.viz_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        plots.append(str(filepath))

        return plots

    def _plot_model_performance(self, model, X_test, y_test, timestamp: str) -> List[str]:
        """Plot comprehensive model performance metrics."""
        plots = []

        # Get predictions
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Create performance dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                    xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
        ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, linewidth=3,
                 label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax2.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7)
        ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        ax3.plot(recall, precision, linewidth=3,
                 label=f'PR Curve (AUC = {pr_auc:.3f})')
        ax3.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Prediction Distribution
        ax4.hist(y_pred_proba[y_test == 0], bins=50,
                 alpha=0.7, label='Benign', density=True)
        ax4.hist(y_pred_proba[y_test == 1], bins=50,
                 alpha=0.7, label='Attack', density=True)
        ax4.axvline(x=0.5, color='red', linestyle='--',
                    linewidth=2, label='Threshold')
        ax4.set_title('Prediction Score Distribution',
                      fontsize=14, fontweight='bold')
        ax4.set_xlabel('Prediction Score')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f"model_performance_{timestamp}.png"
        filepath = self.viz_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        plots.append(str(filepath))

        return plots

    def _plot_federated_progress(self, history_path: str, timestamp: str) -> List[str]:
        """Plot federated learning progress over rounds."""
        plots = []

        try:
            with open(history_path, 'r') as f:
                fed_history = json.load(f)

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

            rounds = list(range(1, len(fed_history['train_accuracy']) + 1))

            # 1. Training Accuracy Progress
            ax1.plot(rounds, fed_history['train_accuracy'], 'o-',
                     linewidth=3, markersize=8, label='Train Accuracy')
            if 'test_accuracy' in fed_history:
                # Filter out None values
                test_acc = [
                    acc for acc in fed_history['test_accuracy'] if acc is not None]
                test_rounds = rounds[:len(test_acc)]
                ax1.plot(test_rounds, test_acc, 's-', linewidth=3,
                         markersize=8, label='Test Accuracy')

            ax1.set_title('Federated Learning Progress',
                          fontsize=14, fontweight='bold')
            ax1.set_xlabel('Federated Round')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)

            # 2. Accuracy Gap Analysis
            if 'test_accuracy' in fed_history:
                test_acc_clean = [
                    acc for acc in fed_history['test_accuracy'] if acc is not None]
                if len(test_acc_clean) > 0:
                    gap = [train - test for train, test in zip(
                        fed_history['train_accuracy'][:len(test_acc_clean)], test_acc_clean)]
                    ax2.plot(test_rounds, gap, 'r-',
                             linewidth=3, label='Train-Test Gap')
                    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    ax2.set_title('Generalization Gap',
                                  fontsize=14, fontweight='bold')
                    ax2.set_xlabel('Federated Round')
                    ax2.set_ylabel('Accuracy Gap')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)

            # 3. Convergence Analysis
            train_smooth = np.convolve(fed_history['train_accuracy'], np.ones(
                min(5, len(rounds)))/min(5, len(rounds)), mode='valid')
            ax3.plot(rounds, fed_history['train_accuracy'],
                     alpha=0.5, label='Raw Train Accuracy')
            ax3.plot(rounds[:len(train_smooth)], train_smooth,
                     linewidth=3, label='Smoothed Trend')
            ax3.set_title('Convergence Analysis',
                          fontsize=14, fontweight='bold')
            ax3.set_xlabel('Federated Round')
            ax3.set_ylabel('Accuracy')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            filename = f"federated_progress_{timestamp}.png"
            filepath = self.viz_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            plots.append(str(filepath))

        except Exception as e:
            print(f"Warning: Could not plot federated progress: {e}")

        return plots

    def _create_summary_dashboard(self, timestamp: str) -> str:
        """Create a comprehensive summary dashboard."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Create a text summary
        summary_text = f"""
        Federated DDoS Detection Training Summary
        Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        📊 Training Completed Successfully
        🎯 Model: 1D CNN for Network Traffic Classification
        🔒 Privacy: Federated Learning with Multi-Krum Aggregation
        📈 Visualization: Comprehensive Performance Analysis
        
        Key Features:
        • Binary Classification (Benign vs Attack)
        • Robust Aggregation Strategy
        • Client Data Privacy Preservation
        • Real-time Performance Monitoring
        
        For detailed analysis, see generated plots:
        • Training History & Convergence
        • Model Performance Metrics
        • Federated Learning Progress
        • ROC/PR Curves & Confusion Matrix
        """

        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Federated DDoS Detection - Training Summary',
                     fontsize=16, fontweight='bold', pad=20)

        filename = f"training_summary_{timestamp}.png"
        filepath = self.viz_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return str(filepath)


def generate_training_visualizations(model=None, history=None, X_test=None, y_test=None,
                                     federated_history_path=None, results_dir="results"):
    """Convenience function to generate all training visualizations."""
    visualizer = FederatedTrainingVisualizer(results_dir)
    return visualizer.generate_all_plots(model, history, X_test, y_test, federated_history_path)
