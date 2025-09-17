import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, auc,
                             precision_recall_curve, precision_score, recall_score, f1_score,
                             accuracy_score, average_precision_score)
from sklearn.preprocessing import label_binarize
import pandas as pd
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class FederatedTrainingVisualizer:
    """Advanced visualization suite for federated DDoS detection training with comprehensive analysis."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.viz_dir = self.results_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Set enhanced style
        plt.style.use('default')
        sns.set_style("whitegrid")
        sns.set_palette("Set2")

        # Configure matplotlib for better plots
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 10

    def generate_all_plots(self,
                           model=None,
                           history=None,
                           X_test=None,
                           y_test=None,
                           federated_history_path: Optional[str] = None,
                           model_name: str = "DDoS_Detection_Model"):
        """Generate comprehensive visualization plots with detailed analysis."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plots_generated = []

        print(f"🎨 Generating advanced visualizations...")

        try:
            # 1. Enhanced Training History Plots
            if history is not None:
                plots_generated.extend(
                    self._plot_enhanced_training_history(history, timestamp))

            # 2. Comprehensive Model Performance Analysis
            if model is not None and X_test is not None and y_test is not None:
                plots_generated.extend(self._plot_comprehensive_performance(
                    model, X_test, y_test, timestamp, model_name))

            # 3. Advanced Federated Learning Analysis
            if federated_history_path and Path(federated_history_path).exists():
                plots_generated.extend(self._plot_advanced_federated_analysis(
                    federated_history_path, timestamp))

            # 4. Detailed Summary Dashboard
            summary_stats = self._calculate_comprehensive_stats(
                model, X_test, y_test, federated_history_path)
            plots_generated.append(self._create_detailed_summary_dashboard(
                summary_stats, timestamp, model_name))

            # 5. Performance Report
            if model is not None and X_test is not None and y_test is not None:
                self._generate_performance_report(
                    model, X_test, y_test, timestamp, model_name)

            print(
                f"✅ Generated {len(plots_generated)} high-quality visualization plots")
            print(f"📁 Saved to: {self.viz_dir}")
            for plot in plots_generated:
                print(f"   - {Path(plot).name}")

            return plots_generated

        except Exception as e:
            print(f"❌ Error generating plots: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _plot_enhanced_training_history(self, history, timestamp: str) -> List[str]:
        """Plot enhanced training/validation curves with detailed analysis."""
        plots = []

        # Create comprehensive training analysis
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # Loss curves with trend analysis
        ax1 = fig.add_subplot(gs[0, 0])
        epochs = range(1, len(history.history['loss']) + 1)
        ax1.plot(epochs, history.history['loss'], 'b-', linewidth=2,
                 label='Training Loss', marker='o', markersize=4)
        if 'val_loss' in history.history:
            ax1.plot(epochs, history.history['val_loss'], 'r-', linewidth=2,
                     label='Validation Loss', marker='s', markersize=4)

            # Add overfitting indicator
            min_val_epoch = np.argmin(history.history['val_loss']) + 1
            ax1.axvline(x=min_val_epoch, color='green', linestyle='--', alpha=0.7,
                        label=f'Best Model (Epoch {min_val_epoch})')

        ax1.set_title('Training Loss Analysis', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy curves with performance zones
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, history.history['accuracy'], 'b-', linewidth=2,
                 label='Training Accuracy', marker='o', markersize=4)
        if 'val_accuracy' in history.history:
            ax2.plot(epochs, history.history['val_accuracy'], 'r-', linewidth=2,
                     label='Validation Accuracy', marker='s', markersize=4)

            # Performance zones
            ax2.axhline(y=0.9, color='green', linestyle=':',
                        alpha=0.6, label='Excellent (>90%)')
            ax2.axhline(y=0.8, color='orange', linestyle=':',
                        alpha=0.6, label='Good (>80%)')
            ax2.axhline(y=0.7, color='red', linestyle=':',
                        alpha=0.6, label='Needs Improvement (<70%)')

        ax2.set_title('Accuracy Progress', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        # Learning rate schedule (if available)
        ax3 = fig.add_subplot(gs[0, 2])
        if 'lr' in history.history:
            lr_data = history.history['lr']
            ax3.plot(epochs, lr_data, 'g-', linewidth=2,
                     marker='d', markersize=4)
            ax3.set_title('Learning Rate Schedule', fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Learning Rate\nData Not Available',
                     ha='center', va='center', transform=ax3.transAxes,
                     fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax3.set_title('Learning Rate Schedule', fontweight='bold')

        # Precision/Recall analysis
        ax4 = fig.add_subplot(gs[0, 3])
        if 'precision' in history.history and 'recall' in history.history:
            ax4.plot(epochs, history.history['precision'], 'purple', linewidth=2,
                     label='Precision', marker='^', markersize=4)
            ax4.plot(epochs, history.history['recall'], 'orange', linewidth=2,
                     label='Recall', marker='v', markersize=4)

            # Calculate F1 score
            precision = np.array(history.history['precision'])
            recall = np.array(history.history['recall'])
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            ax4.plot(epochs, f1, 'brown', linewidth=2,
                     label='F1-Score', marker='*', markersize=4)

            ax4.set_title('Precision/Recall/F1 Analysis', fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Score')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 1)
        else:
            ax4.text(0.5, 0.5, 'Precision/Recall\nData Not Available',
                     ha='center', va='center', transform=ax4.transAxes,
                     fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax4.set_title('Precision/Recall Analysis', fontweight='bold')

        # Training efficiency analysis
        ax5 = fig.add_subplot(gs[1, :2])
        if 'val_loss' in history.history and 'val_accuracy' in history.history:
            # Generalization gap
            train_acc = np.array(history.history['accuracy'])
            val_acc = np.array(history.history['val_accuracy'])
            gap = train_acc - val_acc

            ax5.plot(epochs, gap, 'red', linewidth=2, marker='o', markersize=4)
            ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax5.axhline(y=0.05, color='orange', linestyle='--',
                        alpha=0.7, label='Acceptable Gap (5%)')
            ax5.axhline(y=0.1, color='red', linestyle='--',
                        alpha=0.7, label='Overfitting Warning (10%)')

            ax5.set_title(
                'Generalization Gap Analysis (Train - Validation)', fontweight='bold')
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('Accuracy Gap')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # Training summary statistics
        ax6 = fig.add_subplot(gs[1, 2:])
        stats_text = self._generate_training_stats_summary(history)
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        ax6.set_title('Training Statistics Summary', fontweight='bold')

        # Model convergence analysis
        ax7 = fig.add_subplot(gs[2, :])
        if 'val_loss' in history.history:
            # Loss improvement rate
            val_loss = np.array(history.history['val_loss'])
            loss_improvement = np.diff(val_loss)
            smoothed_improvement = np.convolve(
                loss_improvement, np.ones(5)/5, mode='valid')

            ax7.plot(epochs[1:len(loss_improvement)+1], loss_improvement, 'lightblue',
                     alpha=0.6, label='Raw Loss Change')
            ax7.plot(epochs[3:len(smoothed_improvement)+3], smoothed_improvement, 'blue',
                     linewidth=3, label='Smoothed Trend')
            ax7.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax7.set_title(
                'Training Convergence Analysis (Loss Improvement Rate)', fontweight='bold')
            ax7.set_xlabel('Epoch')
            ax7.set_ylabel('Loss Change')
            ax7.legend()
            ax7.grid(True, alpha=0.3)

        plt.suptitle('Enhanced Training History Analysis',
                     fontsize=16, fontweight='bold', y=0.98)
        filename = f"enhanced_training_history_{timestamp}.png"
        filepath = self.viz_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plots.append(str(filepath))

        return plots

    def _generate_training_stats_summary(self, history) -> str:
        """Generate comprehensive training statistics summary."""
        stats = []

        # Basic training info
        total_epochs = len(history.history['loss'])
        stats.append(f"📊 TRAINING STATISTICS")
        stats.append(f"{'='*30}")
        stats.append(f"Total Epochs: {total_epochs}")

        # Final metrics
        final_loss = history.history['loss'][-1]
        final_acc = history.history['accuracy'][-1]
        stats.append(f"Final Training Loss: {final_loss:.4f}")
        stats.append(f"Final Training Accuracy: {final_acc:.4f}")

        if 'val_loss' in history.history:
            final_val_loss = history.history['val_loss'][-1]
            final_val_acc = history.history['val_accuracy'][-1]
            stats.append(f"Final Validation Loss: {final_val_loss:.4f}")
            stats.append(f"Final Validation Accuracy: {final_val_acc:.4f}")

            # Best validation performance
            best_val_acc = max(history.history['val_accuracy'])
            best_val_epoch = np.argmax(history.history['val_accuracy']) + 1
            stats.append(f"Best Validation Accuracy: {best_val_acc:.4f}")
            stats.append(f"Best Performance at Epoch: {best_val_epoch}")

            # Overfitting analysis
            gap = final_acc - final_val_acc
            stats.append(f"Generalization Gap: {gap:.4f}")
            if gap > 0.1:
                stats.append("⚠️ WARNING: Possible overfitting")
            elif gap < 0.05:
                stats.append("✅ Good generalization")

        # Learning efficiency
        if len(history.history['accuracy']) > 1:
            acc_improvement = history.history['accuracy'][-1] - \
                history.history['accuracy'][0]
            stats.append(f"Total Accuracy Improvement: {acc_improvement:.4f}")

        # Precision/Recall summary
        if 'precision' in history.history and 'recall' in history.history:
            final_precision = history.history['precision'][-1]
            final_recall = history.history['recall'][-1]
            final_f1 = 2 * (final_precision * final_recall) / \
                (final_precision + final_recall + 1e-10)
            stats.append(f"Final Precision: {final_precision:.4f}")
            stats.append(f"Final Recall: {final_recall:.4f}")
            stats.append(f"Final F1-Score: {final_f1:.4f}")

        return '\n'.join(stats)

    def _plot_comprehensive_performance(self, model, X_test, y_test, timestamp: str, model_name: str) -> List[str]:
        """Plot comprehensive model performance analysis with detailed metrics."""
        plots = []

        # Get predictions
        print("🔮 Generating predictions for analysis...")
        y_pred_proba = model.predict(X_test, verbose=0)
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            # Take positive class probability
            y_pred_proba = y_pred_proba[:, 1]
        else:
            y_pred_proba = y_pred_proba.flatten()

        y_pred = (y_pred_proba > 0.5).astype(int)

        # Calculate comprehensive metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Create comprehensive performance dashboard
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        # 1. Enhanced Confusion Matrix with detailed analysis
        ax1 = fig.add_subplot(gs[0, 0])
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                    xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'],
                    cbar_kws={'label': 'Count'})

        # Add percentage annotations
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax1.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1%})',
                         ha='center', va='center', fontsize=10, color='red', fontweight='bold')

        ax1.set_title(
            'Detailed Confusion Matrix\n(Count and Percentage)', fontweight='bold')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')

        # 2. ROC Curve with detailed analysis
        ax2 = fig.add_subplot(gs[0, 1])
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        ax2.plot(fpr, tpr, linewidth=3,
                 label=f'ROC Curve (AUC = {roc_auc:.4f})', color='darkorange')
        ax2.plot([0, 1], [0, 1], 'k--', linewidth=2,
                 alpha=0.7, label='Random Classifier')

        # Mark optimal threshold
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = roc_thresholds[optimal_idx]
        ax2.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10,
                 label=f'Optimal Point (θ={optimal_threshold:.3f})')

        ax2.set_title('ROC Curve Analysis', fontweight='bold')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Precision-Recall Curve
        ax3 = fig.add_subplot(gs[0, 2])
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
            y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)

        ax3.plot(recall_curve, precision_curve, linewidth=3,
                 label=f'PR Curve (AP = {pr_auc:.4f})', color='darkgreen')

        # Mark current operating point
        ax3.plot(recall, precision, 'ro', markersize=10,
                 label=f'Current Point (θ=0.5)')

        # Add baseline (random classifier)
        baseline = np.sum(y_test) / len(y_test)
        ax3.axhline(y=baseline, color='red', linestyle='--', alpha=0.7,
                    label=f'Random Baseline ({baseline:.3f})')

        ax3.set_title('Precision-Recall Analysis', fontweight='bold')
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Prediction Score Distribution Analysis
        ax4 = fig.add_subplot(gs[0, 3])
        benign_scores = y_pred_proba[y_test == 0]
        attack_scores = y_pred_proba[y_test == 1]

        ax4.hist(benign_scores, bins=50, alpha=0.7, label=f'Benign (n={len(benign_scores)})',
                 density=True, color='blue')
        ax4.hist(attack_scores, bins=50, alpha=0.7, label=f'Attack (n={len(attack_scores)})',
                 density=True, color='red')
        ax4.axvline(x=0.5, color='black', linestyle='--',
                    linewidth=2, label='Decision Threshold')

        # Add mean lines
        ax4.axvline(x=np.mean(benign_scores), color='blue', linestyle=':', alpha=0.8,
                    label=f'Benign Mean ({np.mean(benign_scores):.3f})')
        ax4.axvline(x=np.mean(attack_scores), color='red', linestyle=':', alpha=0.8,
                    label=f'Attack Mean ({np.mean(attack_scores):.3f})')

        ax4.set_title('Prediction Score Distribution', fontweight='bold')
        ax4.set_xlabel('Prediction Score')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Threshold Analysis
        ax5 = fig.add_subplot(gs[1, :2])
        thresholds = np.linspace(0, 1, 100)
        precisions, recalls, f1_scores, accuracies = [], [], [], []

        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            if len(np.unique(y_pred_thresh)) > 1:
                precisions.append(precision_score(
                    y_test, y_pred_thresh, zero_division=0))
                recalls.append(recall_score(
                    y_test, y_pred_thresh, zero_division=0))
                f1_scores.append(
                    f1_score(y_test, y_pred_thresh, zero_division=0))
                accuracies.append(accuracy_score(y_test, y_pred_thresh))
            else:
                precisions.append(0)
                recalls.append(0)
                f1_scores.append(0)
                accuracies.append(0)

        ax5.plot(thresholds, precisions, label='Precision',
                 linewidth=2, marker='o', markersize=3)
        ax5.plot(thresholds, recalls, label='Recall',
                 linewidth=2, marker='s', markersize=3)
        ax5.plot(thresholds, f1_scores, label='F1-Score',
                 linewidth=2, marker='^', markersize=3)
        ax5.plot(thresholds, accuracies, label='Accuracy',
                 linewidth=2, marker='d', markersize=3)

        # Mark current threshold
        ax5.axvline(x=0.5, color='black', linestyle='--',
                    alpha=0.7, label='Current Threshold (0.5)')

        # Mark optimal F1 threshold
        optimal_f1_idx = np.argmax(f1_scores)
        optimal_f1_threshold = thresholds[optimal_f1_idx]
        ax5.axvline(x=optimal_f1_threshold, color='green', linestyle=':', alpha=0.7,
                    label=f'Optimal F1 Threshold ({optimal_f1_threshold:.3f})')

        ax5.set_title(
            'Threshold Analysis - Performance vs Decision Threshold', fontweight='bold')
        ax5.set_xlabel('Decision Threshold')
        ax5.set_ylabel('Score')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0, 1)

        # 6. Detailed Performance Metrics
        ax6 = fig.add_subplot(gs[1, 2:])
        # Calculate detailed metrics
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        # Negative Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        # Positive Predictive Value
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0

        metrics_text = f"""
🎯 COMPREHENSIVE PERFORMANCE METRICS
{'='*50}

BASIC METRICS:
• Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
• Precision (PPV): {precision:.4f} ({precision*100:.2f}%)
• Recall (Sensitivity): {recall:.4f} ({recall*100:.2f}%)
• F1-Score: {f1:.4f}
• Specificity: {specificity:.4f} ({specificity*100:.2f}%)

CONFUSION MATRIX BREAKDOWN:
• True Positives (TP): {tp:,} 
• True Negatives (TN): {tn:,}
• False Positives (FP): {fp:,} 
• False Negatives (FN): {fn:,}

RATES & RATIOS:
• False Positive Rate: {fp/(fp+tn)*100:.2f}%
• False Negative Rate: {fn/(fn+tp)*100:.2f}%
• Positive Predictive Value: {ppv:.4f}
• Negative Predictive Value: {npv:.4f}

AREA UNDER CURVES:
• ROC-AUC: {roc_auc:.4f}
• PR-AUC: {pr_auc:.4f}

THRESHOLD ANALYSIS:
• Current Threshold: 0.500
• Optimal F1 Threshold: {optimal_f1_threshold:.3f}
• Max F1-Score Achievable: {max(f1_scores):.4f}

SECURITY FOCUS:
• Attack Detection Rate: {recall*100:.2f}%
• False Alarm Rate: {fp/(fp+tn)*100:.2f}%
• Attack Precision: {precision*100:.2f}%
        """

        ax6.text(0.05, 0.95, metrics_text, transform=ax6.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        ax6.set_title('Detailed Performance Metrics', fontweight='bold')

        # 7. Class-wise Performance Analysis
        ax7 = fig.add_subplot(gs[2, :2])
        class_report = classification_report(y_test, y_pred, output_dict=True)

        classes = ['Benign', 'Attack']
        metrics_names = ['precision', 'recall', 'f1-score']
        class_metrics = np.array([[class_report['0'][metric] for metric in metrics_names],
                                 [class_report['1'][metric] for metric in metrics_names]])

        x = np.arange(len(classes))
        width = 0.25

        for i, metric in enumerate(metrics_names):
            ax7.bar(x + i*width,
                    class_metrics[:, i], width, label=metric.title())

        ax7.set_title('Class-wise Performance Comparison', fontweight='bold')
        ax7.set_xlabel('Traffic Class')
        ax7.set_ylabel('Score')
        ax7.set_xticks(x + width)
        ax7.set_xticklabels(classes)
        ax7.legend()
        ax7.grid(True, alpha=0.3, axis='y')
        ax7.set_ylim(0, 1)

        # 8. Model Confidence Analysis
        ax8 = fig.add_subplot(gs[2, 2:])

        # Confidence bins analysis
        confidence_bins = np.linspace(0, 1, 11)
        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2

        bin_accuracies = []
        bin_counts = []

        for i in range(len(confidence_bins)-1):
            mask = (y_pred_proba >= confidence_bins[i]) & (
                y_pred_proba < confidence_bins[i+1])
            if np.sum(mask) > 0:
                bin_acc = accuracy_score(y_test[mask], y_pred[mask])
                bin_accuracies.append(bin_acc)
                bin_counts.append(np.sum(mask))
            else:
                bin_accuracies.append(0)
                bin_counts.append(0)

        # Create bars colored by accuracy
        colors = ['red' if acc < 0.7 else 'orange' if acc <
                  0.9 else 'green' for acc in bin_accuracies]
        bars = ax8.bar(bin_centers, bin_counts,
                       width=0.08, alpha=0.7, color=colors)

        # Add accuracy labels on bars
        for i, (bar, acc, count) in enumerate(zip(bars, bin_accuracies, bin_counts)):
            if count > 0:
                ax8.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(bin_counts)*0.01,
                         f'{acc:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax8.set_title(
            'Model Confidence vs Accuracy by Prediction Score Bins', fontweight='bold')
        ax8.set_xlabel('Prediction Score Bin')
        ax8.set_ylabel('Number of Samples')
        ax8.grid(True, alpha=0.3, axis='y')

        # 9. Error Analysis
        ax9 = fig.add_subplot(gs[3, :])

        # Identify misclassifications
        false_positives = (y_test == 0) & (y_pred == 1)
        false_negatives = (y_test == 1) & (y_pred == 0)

        fp_scores = y_pred_proba[false_positives]
        fn_scores = y_pred_proba[false_negatives]

        ax9.hist(fp_scores, bins=30, alpha=0.7, label=f'False Positives (n={len(fp_scores)})',
                 color='orange', density=True)
        ax9.hist(fn_scores, bins=30, alpha=0.7, label=f'False Negatives (n={len(fn_scores)})',
                 color='red', density=True)
        ax9.axvline(x=0.5, color='black', linestyle='--',
                    linewidth=2, label='Decision Threshold')

        ax9.set_title(
            'Error Analysis - Distribution of Misclassified Samples', fontweight='bold')
        ax9.set_xlabel('Prediction Score')
        ax9.set_ylabel('Density')
        ax9.legend()
        ax9.grid(True, alpha=0.3)

        plt.suptitle(f'{model_name} - Comprehensive Performance Analysis',
                     fontsize=18, fontweight='bold', y=0.98)

        filename = f"comprehensive_performance_{timestamp}.png"
        filepath = self.viz_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plots.append(str(filepath))

        return plots

    def _plot_advanced_federated_analysis(self, history_path: str, timestamp: str) -> List[str]:
        """Plot advanced federated learning analysis with detailed insights."""
        plots = []

        try:
            with open(history_path, 'r') as f:
                fed_history = json.load(f)

            # Create comprehensive federated analysis
            fig = plt.figure(figsize=(20, 14))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

            rounds = list(range(1, len(fed_history['train_accuracy']) + 1))

            # 1. Training Progress with Trend Analysis
            ax1 = fig.add_subplot(gs[0, 0])
            train_acc = fed_history['train_accuracy']
            ax1.plot(rounds, train_acc, 'o-', linewidth=3, markersize=8,
                     color='blue', label='Train Accuracy')

            if 'test_accuracy' in fed_history:
                test_acc = [
                    acc for acc in fed_history['test_accuracy'] if acc is not None]
                test_rounds = rounds[:len(test_acc)]
                ax1.plot(test_rounds, test_acc, 's-', linewidth=3, markersize=8,
                         color='red', label='Test Accuracy')

                # Add trend lines
                if len(test_acc) > 3:
                    train_trend = np.polyfit(rounds, train_acc, 1)
                    test_trend = np.polyfit(test_rounds, test_acc, 1)
                    ax1.plot(rounds, np.poly1d(train_trend)(rounds), '--',
                             color='blue', alpha=0.7, label=f'Train Trend (slope: {train_trend[0]:.4f})')
                    ax1.plot(test_rounds, np.poly1d(test_trend)(test_rounds), '--',
                             color='red', alpha=0.7, label=f'Test Trend (slope: {test_trend[0]:.4f})')

            ax1.set_title(
                'Federated Learning Progress with Trends', fontweight='bold')
            ax1.set_xlabel('Round')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)

            # 2. Convergence Analysis
            ax2 = fig.add_subplot(gs[0, 1])
            if len(train_acc) > 1:
                # Calculate improvement rates
                improvement_rates = np.diff(train_acc)
                smoothed_rates = np.convolve(
                    improvement_rates, np.ones(3)/3, mode='valid')

                ax2.plot(rounds[1:], improvement_rates, 'o-', alpha=0.6,
                         label='Round-to-Round Improvement')
                if len(smoothed_rates) > 0:
                    ax2.plot(rounds[2:len(smoothed_rates)+2], smoothed_rates,
                             linewidth=3, label='Smoothed Trend')
                ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax2.set_title('Convergence Analysis', fontweight='bold')
                ax2.set_xlabel('Round')
                ax2.set_ylabel('Accuracy Improvement')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            # 3. Generalization Gap Analysis
            ax3 = fig.add_subplot(gs[0, 2])
            if 'test_accuracy' in fed_history:
                test_acc_clean = [
                    acc for acc in fed_history['test_accuracy'] if acc is not None]
                if len(test_acc_clean) > 0:
                    gap = [train - test for train, test in zip(
                        fed_history['train_accuracy'][:len(test_acc_clean)], test_acc_clean)]
                    gap_rounds = rounds[:len(gap)]

                    ax3.plot(gap_rounds, gap, 'r-', linewidth=3,
                             marker='o', markersize=6)
                    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                    ax3.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7,
                                label='Acceptable Gap (5%)')
                    ax3.axhline(y=0.1, color='red', linestyle='--', alpha=0.7,
                                label='Overfitting Warning (10%)')

                    # Add average gap
                    avg_gap = np.mean(gap)
                    ax3.axhline(y=avg_gap, color='purple', linestyle=':', alpha=0.8,
                                label=f'Average Gap ({avg_gap:.3f})')

                    ax3.set_title('Generalization Gap Analysis',
                                  fontweight='bold')
                    ax3.set_xlabel('Round')
                    ax3.set_ylabel('Train - Test Accuracy')
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)

            # 4. Round-by-Round Performance Heatmap
            ax4 = fig.add_subplot(gs[1, :])
            if 'test_accuracy' in fed_history:
                # Create performance matrix
                metrics_data = []
                metrics_labels = []

                if train_acc:
                    metrics_data.append(train_acc)
                    metrics_labels.append('Train Accuracy')

                test_acc_clean = [
                    acc if acc is not None else 0 for acc in fed_history.get('test_accuracy', [])]
                if test_acc_clean and any(x > 0 for x in test_acc_clean):
                    metrics_data.append(test_acc_clean[:len(train_acc)])
                    metrics_labels.append('Test Accuracy')

                if len(metrics_data) > 0:
                    performance_matrix = np.array(metrics_data)
                    sns.heatmap(performance_matrix,
                                xticklabels=[f'Round {i}' for i in rounds],
                                yticklabels=metrics_labels,
                                annot=True, fmt='.3f', cmap='RdYlGn',
                                cbar_kws={'label': 'Accuracy'}, ax=ax4)
                    ax4.set_title(
                        'Round-by-Round Performance Heatmap', fontweight='bold')

            # 5. Statistical Summary
            ax5 = fig.add_subplot(gs[2, 0])
            stats_text = self._generate_federated_stats_summary(fed_history)
            ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, fontsize=10,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
            ax5.set_xlim(0, 1)
            ax5.set_ylim(0, 1)
            ax5.axis('off')
            ax5.set_title('Federated Training Statistics', fontweight='bold')

            # 6. Performance Distribution
            ax6 = fig.add_subplot(gs[2, 1])
            all_accuracies = train_acc.copy()
            if 'test_accuracy' in fed_history:
                test_acc_clean = [
                    acc for acc in fed_history['test_accuracy'] if acc is not None]
                all_accuracies.extend(test_acc_clean)

            ax6.hist(train_acc, bins=10, alpha=0.7, label=f'Train (μ={np.mean(train_acc):.3f})',
                     color='blue', density=True)
            if 'test_accuracy' in fed_history and test_acc_clean:
                ax6.hist(test_acc_clean, bins=10, alpha=0.7,
                         label=f'Test (μ={np.mean(test_acc_clean):.3f})',
                         color='red', density=True)

            ax6.set_title('Performance Distribution', fontweight='bold')
            ax6.set_xlabel('Accuracy')
            ax6.set_ylabel('Density')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

            # 7. Learning Stability Analysis
            ax7 = fig.add_subplot(gs[2, 2])
            if len(train_acc) > 3:
                # Calculate rolling standard deviation
                window_size = min(3, len(train_acc) // 2)
                rolling_std = []
                for i in range(window_size, len(train_acc)):
                    rolling_std.append(np.std(train_acc[i-window_size:i+1]))

                if rolling_std:
                    ax7.plot(rounds[window_size:], rolling_std, 'g-', linewidth=3,
                             marker='o', markersize=6, label='Training Stability')
                    ax7.set_title(
                        'Learning Stability (Rolling Std)', fontweight='bold')
                    ax7.set_xlabel('Round')
                    ax7.set_ylabel('Standard Deviation')
                    ax7.legend()
                    ax7.grid(True, alpha=0.3)

                    # Add stability zones
                    ax7.axhline(y=0.01, color='green', linestyle='--', alpha=0.7,
                                label='Very Stable (<0.01)')
                    ax7.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7,
                                label='Stable (<0.05)')

            plt.suptitle('Advanced Federated Learning Analysis',
                         fontsize=16, fontweight='bold', y=0.98)

            filename = f"advanced_federated_analysis_{timestamp}.png"
            filepath = self.viz_dir / filename
            plt.savefig(filepath, dpi=300,
                        bbox_inches='tight', facecolor='white')
            plt.close()
            plots.append(str(filepath))

        except Exception as e:
            print(f"Warning: Could not plot advanced federated analysis: {e}")
            import traceback
            traceback.print_exc()

        return plots

    def _generate_federated_stats_summary(self, fed_history) -> str:
        """Generate comprehensive federated training statistics."""
        stats = []

        train_acc = fed_history['train_accuracy']
        total_rounds = len(train_acc)

        stats.append(f"🌐 FEDERATED LEARNING STATS")
        stats.append(f"{'='*35}")
        stats.append(f"Total Rounds: {total_rounds}")

        # Training accuracy statistics
        stats.append(f"")
        stats.append(f"TRAINING ACCURACY:")
        stats.append(f"• Final: {train_acc[-1]:.4f}")
        stats.append(f"• Best: {max(train_acc):.4f}")
        stats.append(f"• Average: {np.mean(train_acc):.4f}")
        stats.append(f"• Std Dev: {np.std(train_acc):.4f}")
        stats.append(f"• Improvement: {train_acc[-1] - train_acc[0]:.4f}")

        # Test accuracy statistics
        if 'test_accuracy' in fed_history:
            test_acc = [acc for acc in fed_history['test_accuracy']
                        if acc is not None]
            if test_acc:
                stats.append(f"")
                stats.append(f"TEST ACCURACY:")
                stats.append(f"• Final: {test_acc[-1]:.4f}")
                stats.append(f"• Best: {max(test_acc):.4f}")
                stats.append(f"• Average: {np.mean(test_acc):.4f}")
                stats.append(f"• Std Dev: {np.std(test_acc):.4f}")

                # Generalization analysis
                if len(test_acc) > 0:
                    final_gap = train_acc[len(test_acc)-1] - test_acc[-1]
                    avg_gap = np.mean([train_acc[i] - test_acc[i]
                                      for i in range(len(test_acc))])
                    stats.append(f"")
                    stats.append(f"GENERALIZATION:")
                    stats.append(f"• Final Gap: {final_gap:.4f}")
                    stats.append(f"• Average Gap: {avg_gap:.4f}")

                    if avg_gap > 0.1:
                        stats.append(f"• Status: ⚠️ Overfitting Risk")
                    elif avg_gap < 0.05:
                        stats.append(f"• Status: ✅ Good Generalization")
                    else:
                        stats.append(f"• Status: 📊 Acceptable")

        # Convergence analysis
        if len(train_acc) > 1:
            improvements = np.diff(train_acc)
            positive_improvements = sum(1 for x in improvements if x > 0)
            stats.append(f"")
            stats.append(f"CONVERGENCE:")
            stats.append(
                f"• Improving Rounds: {positive_improvements}/{len(improvements)}")
            stats.append(f"• Avg Improvement: {np.mean(improvements):.4f}")

            # Convergence status
            recent_improvements = improvements[-3:] if len(
                improvements) >= 3 else improvements
            if all(x <= 0.001 for x in recent_improvements):
                stats.append(f"• Status: ✅ Converged")
            elif np.mean(recent_improvements) > 0:
                stats.append(f"• Status: 📈 Still Improving")
            else:
                stats.append(f"• Status: 📉 May Need Adjustment")

        return '\n'.join(stats)

    def _calculate_comprehensive_stats(self, model, X_test, y_test, federated_history_path):
        """Calculate comprehensive statistics for summary dashboard."""
        stats = {}

        # Model performance stats
        if model is not None and X_test is not None and y_test is not None:
            y_pred_proba = model.predict(X_test, verbose=0)
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                y_pred_proba = y_pred_proba[:, 1]
            else:
                y_pred_proba = y_pred_proba.flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)

            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()

            stats['model'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': auc(*roc_curve(y_test, y_pred_proba)[:2]),
                'pr_auc': average_precision_score(y_test, y_pred_proba),
                'total_samples': len(y_test),
                'positive_samples': np.sum(y_test),
                'true_positives': tp,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn,
                'attack_detection_rate': recall_score(y_test, y_pred, zero_division=0),
                'false_alarm_rate': fp / (fp + tn) if (fp + tn) > 0 else 0
            }

        # Federated learning stats
        if federated_history_path and Path(federated_history_path).exists():
            try:
                with open(federated_history_path, 'r') as f:
                    fed_history = json.load(f)

                train_acc = fed_history['train_accuracy']
                stats['federated'] = {
                    'total_rounds': len(train_acc),
                    'final_train_acc': train_acc[-1],
                    'best_train_acc': max(train_acc),
                    'improvement': train_acc[-1] - train_acc[0],
                    'convergence_rounds': len(train_acc)
                }

                if 'test_accuracy' in fed_history:
                    test_acc = [
                        acc for acc in fed_history['test_accuracy'] if acc is not None]
                    if test_acc:
                        stats['federated'].update({
                            'final_test_acc': test_acc[-1],
                            'best_test_acc': max(test_acc),
                            'generalization_gap': train_acc[len(test_acc)-1] - test_acc[-1]
                        })
            except:
                stats['federated'] = None

        return stats

    def _create_detailed_summary_dashboard(self, stats, timestamp: str, model_name: str) -> str:
        """Create a comprehensive and informative summary dashboard."""

        # Create a comprehensive dashboard
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[
                              1, 2, 1], hspace=0.3, wspace=0.3)

        # Header with key metrics
        ax_header = fig.add_subplot(gs[0, :])
        ax_header.axis('off')

        header_text = f"{model_name} - Training Summary Report\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ax_header.text(0.5, 0.7, header_text, ha='center', va='center',
                       fontsize=16, fontweight='bold', transform=ax_header.transAxes)

        # Model Performance Summary
        ax_model = fig.add_subplot(gs[1, 0])
        ax_model.axis('off')

        if 'model' in stats and stats['model']:
            model_stats = stats['model']

            perf_text = f"""
🎯 MODEL PERFORMANCE SUMMARY
{'='*45}

CLASSIFICATION METRICS:
✓ Accuracy: {model_stats['accuracy']:.3f} ({model_stats['accuracy']*100:.1f}%)
✓ Precision: {model_stats['precision']:.3f} ({model_stats['precision']*100:.1f}%)
✓ Recall: {model_stats['recall']:.3f} ({model_stats['recall']*100:.1f}%)
✓ F1-Score: {model_stats['f1']:.3f}

AREA UNDER CURVE:
📈 ROC-AUC: {model_stats['roc_auc']:.3f}
📊 PR-AUC: {model_stats['pr_auc']:.3f}

SECURITY METRICS:
🛡️ Attack Detection Rate: {model_stats['attack_detection_rate']*100:.1f}%
🚨 False Alarm Rate: {model_stats['false_alarm_rate']*100:.1f}%

CONFUSION MATRIX:
┌─────────────────────────────┐
│     Predicted    │          │
│  Benign │ Attack │   Total  │
├─────────┼────────┼──────────┤
│ {model_stats['true_negatives']:7} │ {model_stats['false_positives']:6} │ {model_stats['true_negatives'] + model_stats['false_positives']:8} │ Benign
│ {model_stats['false_negatives']:7} │ {model_stats['true_positives']:6} │ {model_stats['false_negatives'] + model_stats['true_positives']:8} │ Attack
├─────────┼────────┼──────────┤
│ {model_stats['true_negatives'] + model_stats['false_negatives']:7} │ {model_stats['false_positives'] + model_stats['true_positives']:6} │ {model_stats['total_samples']:8} │ Total
└─────────────────────────────┘

DATASET INFORMATION:
📊 Total Samples: {model_stats['total_samples']:,}
⚔️ Attack Samples: {model_stats['positive_samples']:,} ({model_stats['positive_samples']/model_stats['total_samples']*100:.1f}%)
🛡️ Benign Samples: {model_stats['total_samples'] - model_stats['positive_samples']:,} ({(model_stats['total_samples'] - model_stats['positive_samples'])/model_stats['total_samples']*100:.1f}%)

PERFORMANCE ASSESSMENT:
"""

            # Add performance assessment
            if model_stats['accuracy'] > 0.9:
                perf_text += "🌟 EXCELLENT: Model shows excellent performance\n"
            elif model_stats['accuracy'] > 0.8:
                perf_text += "✅ GOOD: Model performance is satisfactory\n"
            elif model_stats['accuracy'] > 0.7:
                perf_text += "⚠️ FAIR: Model needs improvement\n"
            else:
                perf_text += "❌ POOR: Model requires significant improvement\n"

            if model_stats['recall'] > 0.9:
                perf_text += "🛡️ SECURITY: Excellent attack detection\n"
            elif model_stats['recall'] > 0.8:
                perf_text += "🔍 SECURITY: Good attack detection\n"
            else:
                perf_text += "⚠️ SECURITY: Attack detection needs improvement\n"

            if model_stats['false_alarm_rate'] < 0.05:
                perf_text += "📡 EFFICIENCY: Low false alarm rate\n"
            elif model_stats['false_alarm_rate'] < 0.1:
                perf_text += "⚖️ EFFICIENCY: Moderate false alarm rate\n"
            else:
                perf_text += "🚨 EFFICIENCY: High false alarm rate\n"

        else:
            perf_text = "Model performance data not available"

        ax_model.text(0.05, 0.95, perf_text, transform=ax_model.transAxes, fontsize=9,
                      verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))

        # Federated Learning Summary
        ax_fed = fig.add_subplot(gs[1, 1])
        ax_fed.axis('off')

        if 'federated' in stats and stats['federated']:
            fed_stats = stats['federated']

            fed_text = f"""
🌐 FEDERATED LEARNING SUMMARY
{'='*45}

TRAINING PROGRESS:
🔄 Total Rounds: {fed_stats['total_rounds']}
📈 Final Training Accuracy: {fed_stats['final_train_acc']:.3f}
🎯 Best Training Accuracy: {fed_stats['best_train_acc']:.3f}
📊 Total Improvement: {fed_stats['improvement']:.3f}

"""

            if 'final_test_acc' in fed_stats:
                fed_text += f"""GENERALIZATION:
🧪 Final Test Accuracy: {fed_stats['final_test_acc']:.3f}
🏆 Best Test Accuracy: {fed_stats['best_test_acc']:.3f}
📏 Generalization Gap: {fed_stats['generalization_gap']:.3f}

"""

                # Add federated assessment
                if fed_stats['generalization_gap'] < 0.05:
                    fed_text += "✅ FEDERATED STATUS: Excellent generalization\n"
                elif fed_stats['generalization_gap'] < 0.1:
                    fed_text += "📊 FEDERATED STATUS: Good generalization\n"
                else:
                    fed_text += "⚠️ FEDERATED STATUS: Overfitting detected\n"

            if fed_stats['improvement'] > 0.1:
                fed_text += "🚀 CONVERGENCE: Strong improvement achieved\n"
            elif fed_stats['improvement'] > 0.05:
                fed_text += "📈 CONVERGENCE: Moderate improvement\n"
            else:
                fed_text += "🔄 CONVERGENCE: Limited improvement\n"

            fed_text += f"""
FEDERATED LEARNING BENEFITS:
🔒 Data Privacy: Raw data never leaves clients
🌐 Distributed: Training across multiple nodes
🛡️ Robust: Multi-Krum aggregation used
📊 Scalable: Can add more clients easily

TECHNICAL DETAILS:
• Aggregation: FedAvg + Multi-Krum
• Privacy: Data stays local
• Communication: Model weights only
• Convergence: {fed_stats['total_rounds']} rounds completed
"""

        else:
            fed_text = """
🌐 FEDERATED LEARNING SUMMARY
{'='*45}

Federated learning data not available.
This may be a centralized training run.

FEDERATED CAPABILITIES:
🔒 Privacy-preserving training
🌐 Distributed learning
🛡️ Robust aggregation
📊 Scalable architecture
"""

        ax_fed.text(0.05, 0.95, fed_text, transform=ax_fed.transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))

        # Footer with recommendations
        ax_footer = fig.add_subplot(gs[2, :])
        ax_footer.axis('off')

        footer_text = """
📋 RECOMMENDATIONS & NEXT STEPS:
• Monitor model performance regularly in production environment
• Consider retraining when performance degrades below acceptable thresholds
• Implement automated monitoring for attack detection accuracy
• Evaluate model on new attack patterns as they emerge
• Consider ensemble methods for improved robustness
"""

        if 'model' in stats and stats['model']:
            model_stats = stats['model']
            if model_stats['recall'] < 0.8:
                footer_text += "• Priority: Improve attack detection rate (recall) to reduce missed attacks\n"
            if model_stats['false_alarm_rate'] > 0.1:
                footer_text += "• Priority: Reduce false positive rate to minimize disruption\n"

        footer_text += f"\n📁 Generated Files: All visualizations saved to {self.viz_dir}/"

        ax_footer.text(0.05, 0.9, footer_text, transform=ax_footer.transAxes, fontsize=10,
                       verticalalignment='top', fontfamily='sans-serif',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

        filename = f"detailed_summary_dashboard_{timestamp}.png"
        filepath = self.viz_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        return str(filepath)

    def _generate_performance_report(self, model, X_test, y_test, timestamp: str, model_name: str):
        """Generate a detailed text performance report."""

        # Get predictions
        y_pred_proba = model.predict(X_test, verbose=0)
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            y_pred_proba = y_pred_proba[:, 1]
        else:
            y_pred_proba = y_pred_proba.flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        try:
            class_report = classification_report(y_test, y_pred,
                                                 target_names=[
                                                     'Benign', 'Attack'],
                                                 output_dict=True)
        except:
            # Fallback if classification report fails
            class_report = {
                '0': {'precision': precision, 'recall': recall, 'f1-score': f1},
                '1': {'precision': precision, 'recall': recall, 'f1-score': f1}
            }

        # Ensure confusion matrix has the right shape
        if cm.shape == (1, 1):
            # Only one class predicted, expand to 2x2
            if len(np.unique(y_test)) == 2 and len(np.unique(y_pred)) == 1:
                if np.unique(y_pred)[0] == 0:
                    # Only predicted class 0
                    cm = np.array([[np.sum((y_test == 0) & (y_pred == 0)), 0],
                                   [np.sum((y_test == 1) & (y_pred == 0)), 0]])
                else:
                    # Only predicted class 1
                    cm = np.array([[0, np.sum((y_test == 0) & (y_pred == 1))],
                                   [0, np.sum((y_test == 1) & (y_pred == 1))]])
            else:
                cm = np.array([[cm[0, 0], 0], [0, 0]])
        elif cm.shape == (2, 1):
            cm = np.column_stack([cm, np.zeros((2, 1))])
        elif cm.shape == (1, 2):
            cm = np.row_stack([cm, np.zeros((1, 2))])

        # Ensure we have all elements for safe access
        if cm.shape[0] < 2 or cm.shape[1] < 2:
            cm = np.pad(
                cm, ((0, 2-cm.shape[0]), (0, 2-cm.shape[1])), mode='constant')

        # Generate report
        report = f"""
# {model_name} - Detailed Performance Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
This model achieves {accuracy:.1%} accuracy in distinguishing between benign network traffic and DDoS attacks.
The model demonstrates {'excellent' if accuracy > 0.9 else 'good' if accuracy > 0.8 else 'moderate'} performance 
with {recall:.1%} attack detection rate and {precision:.1%} attack prediction precision.

## Performance Metrics

### Core Classification Metrics
- **Accuracy**: {accuracy:.4f} ({accuracy:.1%})
- **Precision**: {precision:.4f} ({precision:.1%})
- **Recall (Sensitivity)**: {recall:.4f} ({recall:.1%})
- **F1-Score**: {f1:.4f}
- **Specificity**: {cm[0, 0]/(cm[0, 0]+cm[0, 1]):.4f}

### Security-Focused Metrics
- **Attack Detection Rate**: {recall:.1%}
- **False Alarm Rate**: {cm[0, 1]/(cm[0, 0]+cm[0, 1]):.1%}
- **Missed Attack Rate**: {cm[1, 0]/(cm[1, 0]+cm[1, 1]):.1%}

### ROC & PR Analysis
- **ROC AUC**: {auc(*roc_curve(y_test, y_pred_proba)[:2]):.4f}
- **PR AUC**: {average_precision_score(y_test, y_pred_proba):.4f}

## Confusion Matrix Analysis
```
                Predicted
                Benign  Attack  Total
Actual Benign   {cm[0, 0]:6}  {cm[0, 1]:6}  {cm[0, 0]+cm[0, 1]:6}
Actual Attack   {cm[1, 0]:6}  {cm[1, 1]:6}  {cm[1, 0]+cm[1, 1]:6}
Total          {cm[0, 0]+cm[1, 0]:6}  {cm[0, 1]+cm[1, 1]:6}  {cm.sum():6}
```

## Class-wise Performance
### Benign Traffic Detection
- Precision: {class_report['0']['precision']:.4f}
- Recall: {class_report['0']['recall']:.4f}
- F1-Score: {class_report['0']['f1-score']:.4f}

### Attack Traffic Detection
- Precision: {class_report['1']['precision']:.4f}
- Recall: {class_report['1']['recall']:.4f}
- F1-Score: {class_report['1']['f1-score']:.4f}

## Model Assessment

### Strengths
"""

        if accuracy > 0.9:
            report += "- Excellent overall classification accuracy\n"
        if recall > 0.9:
            report += "- Very high attack detection rate\n"
        if cm[0, 1]/(cm[0, 0]+cm[0, 1]) < 0.05:
            report += "- Low false alarm rate\n"
        if f1 > 0.85:
            report += "- Well-balanced precision and recall\n"

        report += f"""
### Areas for Improvement
"""

        if recall < 0.8:
            report += "- Attack detection rate could be improved to reduce security risks\n"
        if precision < 0.8:
            report += "- Attack prediction precision needs improvement to reduce false alarms\n"
        if accuracy < 0.85:
            report += "- Overall accuracy requires enhancement\n"

        report += f"""
## Recommendations

### Immediate Actions
1. {'✅' if recall > 0.9 else '⚠️'} Attack Detection: Current rate is {recall:.1%}
2. {'✅' if cm[0, 1]/(cm[0, 0]+cm[0, 1]) < 0.05 else '⚠️'} False Alarms: Current rate is {cm[0, 1]/(cm[0, 0]+cm[0, 1]):.1%}
3. {'✅' if accuracy > 0.9 else '⚠️'} Overall Performance: {accuracy:.1%} accuracy achieved

### Strategic Improvements
- Consider ensemble methods for improved robustness
- Implement threshold optimization for specific security requirements
- Regular retraining as attack patterns evolve
- Monitor performance degradation over time

---
Report generated by Advanced DDoS Detection Visualization System
"""

        # Save report
        report_file = self.viz_dir / f"performance_report_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(report)

        print(f"📝 Detailed performance report saved: {report_file}")

    def _create_summary_dashboard(self, timestamp: str) -> str:
        """Legacy method - replaced by detailed summary dashboard."""
        return self._create_detailed_summary_dashboard({}, timestamp, "DDoS Detection Model")


def generate_training_visualizations(model=None, history=None, X_test=None, y_test=None,
                                     federated_history_path=None, results_dir="results",
                                     model_name="DDoS_Detection_Model"):
    """
    Convenience function to generate all advanced training visualizations.

    Args:
        model: Trained model for performance analysis
        history: Training history object (for centralized training)
        X_test: Test features for evaluation
        y_test: Test labels for evaluation
        federated_history_path: Path to federated metrics JSON file
        results_dir: Directory to save results
        model_name: Name of the model for titles and reports

    Returns:
        List of generated plot file paths
    """
    visualizer = FederatedTrainingVisualizer(results_dir)
    return visualizer.generate_all_plots(
        model=model,
        history=history,
        X_test=X_test,
        y_test=y_test,
        federated_history_path=federated_history_path,
        model_name=model_name
    )
