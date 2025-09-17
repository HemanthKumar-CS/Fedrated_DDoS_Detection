import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, auc,
                             precision_score, recall_score, f1_score, accuracy_score)
from pathlib import Path
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_essential_visualizations(history, y_test, y_pred, y_pred_proba, results_dir="results"):
    """Plot and save essential visualizations separately"""
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Training and Validation Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='orange')
    plt.title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/01_accuracy_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Accuracy curves saved to {results_dir}/01_accuracy_curves.png")
    
    # 2. Training and Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2, color='red')
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='purple')
    plt.title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/02_loss_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Loss curves saved to {results_dir}/02_loss_curves.png")
    
    # 3. Classification Report Visualization
    report = classification_report(y_test, y_pred, target_names=['Benign', 'Attack'], output_dict=True)
    
    # Extract metrics for visualization
    classes = ['Benign', 'Attack']
    precision = [report['Benign']['precision'], report['Attack']['precision']]
    recall = [report['Benign']['recall'], report['Attack']['recall']]
    f1_score_vals = [report['Benign']['f1-score'], report['Attack']['f1-score']]
    
    x = np.arange(len(classes))
    width = 0.25
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
    plt.bar(x, recall, width, label='Recall', alpha=0.8)
    plt.bar(x + width, f1_score_vals, width, label='F1-Score', alpha=0.8)
    
    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.title('Classification Report - Per Class Metrics', fontsize=14, fontweight='bold')
    plt.xticks(x, classes)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/03_classification_report.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Classification report saved to {results_dir}/03_classification_report.png")
    
    # 4. Confusion Matrix with Percentages
    cm = confusion_matrix(y_test, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'],
                cbar_kws={'label': 'Percentage (%)'})
    plt.title('Confusion Matrix (Percentage)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add count annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j+0.5, i+0.7, f'({cm[i,j]})', ha='center', va='center', 
                    fontsize=10, color='darkred')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/04_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to {results_dir}/04_confusion_matrix.png")
    
    # 5. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba.flatten())
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/05_roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"ROC curve saved to {results_dir}/05_roc_curve.png")
    
    return roc_auc


def plot_federated_stats(history, save_dir="results"):
    """Plot federated learning statistics"""
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # 6. Federated Learning Stats
    if hasattr(history, 'metrics_distributed') and history.metrics_distributed:
        # Extract round-wise metrics
        rounds = []
        accuracies = []
        losses = []
        
        for round_num, (_, accuracy) in enumerate(history.metrics_distributed.get('accuracy', []), 1):
            rounds.append(round_num)
            accuracies.append(accuracy)
        
        for round_num, (_, loss) in enumerate(history.losses_distributed, 1):
            losses.append(loss)
        
        # Plot accuracy progression
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, accuracies, marker='o', linewidth=2, markersize=8, color='green')
        plt.title('Federated Learning - Accuracy Progression', fontsize=14, fontweight='bold')
        plt.xlabel('Communication Round')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/06_federated_accuracy.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Federated accuracy saved to {save_dir}/06_federated_accuracy.png")
        
        # Plot loss progression
        plt.figure(figsize=(10, 6))
        plt.plot(rounds[:len(losses)], losses, marker='s', linewidth=2, markersize=8, color='red')
        plt.title('Federated Learning - Loss Progression', fontsize=14, fontweight='bold')
        plt.xlabel('Communication Round')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/07_federated_loss.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Federated loss saved to {save_dir}/07_federated_loss.png")


def plot_federated_progress_trend(history, save_dir="results"):
    """Plot federated learning progress with trend analysis"""
    
    if not hasattr(history, 'metrics_distributed') or not history.metrics_distributed:
        logger.warning("No federated metrics available for trend analysis")
        return
    
    # 8. Federated Learning Progress with Trend
    rounds = []
    accuracies = []
    
    for round_num, (_, accuracy) in enumerate(history.metrics_distributed.get('accuracy', []), 1):
        rounds.append(round_num)
        accuracies.append(accuracy)
    
    if len(rounds) > 1:
        # Calculate trend line
        z = np.polyfit(rounds, accuracies, 1)
        p = np.poly1d(z)
        
        plt.figure(figsize=(12, 8))
        
        # Main progress plot
        plt.subplot(2, 1, 1)
        plt.plot(rounds, accuracies, 'bo-', linewidth=2, markersize=8, label='Actual Accuracy')
        plt.plot(rounds, p(rounds), 'r--', linewidth=2, label=f'Trend (slope: {z[0]:.4f})')
        plt.title('Federated Learning Progress with Trend Analysis', fontsize=14, fontweight='bold')
        plt.xlabel('Communication Round')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Improvement rate plot
        plt.subplot(2, 1, 2)
        if len(accuracies) > 1:
            improvements = [accuracies[i] - accuracies[i-1] for i in range(1, len(accuracies))]
            plt.bar(rounds[1:], improvements, alpha=0.7, color='orange')
            plt.title('Round-to-Round Improvement', fontsize=12, fontweight='bold')
            plt.xlabel('Communication Round')
            plt.ylabel('Accuracy Improvement')
            plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/08_federated_progress_trend.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Federated progress trend saved to {save_dir}/08_federated_progress_trend.png")


def plot_convergence_analysis(history, save_dir="results"):
    """Plot convergence analysis"""
    
    if not hasattr(history, 'metrics_distributed') or not history.metrics_distributed:
        logger.warning("No federated metrics available for convergence analysis")
        return
    
    # 9. Convergence Analysis
    rounds = []
    accuracies = []
    
    for round_num, (_, accuracy) in enumerate(history.metrics_distributed.get('accuracy', []), 1):
        rounds.append(round_num)
        accuracies.append(accuracy)
    
    if len(rounds) > 2:
        plt.figure(figsize=(12, 6))
        
        # Calculate moving average for smoothing
        window_size = min(3, len(accuracies))
        moving_avg = []
        for i in range(len(accuracies)):
            start_idx = max(0, i - window_size + 1)
            moving_avg.append(np.mean(accuracies[start_idx:i+1]))
        
        # Calculate convergence rate (derivative approximation)
        convergence_rate = []
        for i in range(1, len(accuracies)):
            rate = abs(accuracies[i] - accuracies[i-1])
            convergence_rate.append(rate)
        
        # Plot convergence analysis
        plt.subplot(1, 2, 1)
        plt.plot(rounds, accuracies, 'b-', linewidth=2, label='Raw Accuracy', alpha=0.6)
        plt.plot(rounds, moving_avg, 'r-', linewidth=3, label=f'Moving Average (window={window_size})')
        plt.title('Convergence Analysis', fontsize=14, fontweight='bold')
        plt.xlabel('Communication Round')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot convergence rate
        plt.subplot(1, 2, 2)
        plt.plot(rounds[1:], convergence_rate, 'g-', linewidth=2, marker='o', markersize=6)
        plt.title('Convergence Rate', fontsize=14, fontweight='bold')
        plt.xlabel('Communication Round')
        plt.ylabel('|Accuracy Change|')
        plt.grid(True, alpha=0.3)
        
        # Add convergence threshold line
        threshold = 0.01  # 1% change threshold
        plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
                   label=f'Convergence Threshold ({threshold:.2f})')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/09_convergence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Convergence analysis saved to {save_dir}/09_convergence_analysis.png")


# Convenience function for generating all essential visualizations
def generate_essential_visualizations(model=None, history=None, X_test=None, y_test=None,
                                      federated_history=None, results_dir="results"):
    """
    Generate only the essential visualizations requested by user.
    
    Args:
        model: Trained model for predictions
        history: Training history object
        X_test: Test features
        y_test: Test labels
        federated_history: Federated learning history object
        results_dir: Directory to save results
    
    Returns:
        Dictionary with generated file paths
    """
    
    generated_files = {}
    
    if model is not None and X_test is not None and y_test is not None and history is not None:
        # Get predictions
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Generate centralized training visualizations
        roc_auc = plot_essential_visualizations(history, y_test, y_pred, y_pred_proba, results_dir)
        
        generated_files.update({
            '01_accuracy_curves': f"{results_dir}/01_accuracy_curves.png",
            '02_loss_curves': f"{results_dir}/02_loss_curves.png",
            '03_classification_report': f"{results_dir}/03_classification_report.png",
            '04_confusion_matrix': f"{results_dir}/04_confusion_matrix.png",
            '05_roc_curve': f"{results_dir}/05_roc_curve.png",
            'roc_auc': roc_auc
        })
    
    # Generate federated learning visualizations if available
    if federated_history is not None:
        plot_federated_stats(federated_history, results_dir)
        plot_federated_progress_trend(federated_history, results_dir)
        plot_convergence_analysis(federated_history, results_dir)
        
        generated_files.update({
            '06_federated_accuracy': f"{results_dir}/06_federated_accuracy.png",
            '07_federated_loss': f"{results_dir}/07_federated_loss.png",
            '08_federated_progress_trend': f"{results_dir}/08_federated_progress_trend.png",
            '09_convergence_analysis': f"{results_dir}/09_convergence_analysis.png"
        })
    
    logger.info("=" * 70)
    logger.info("🎉 Essential Visualizations Generated Successfully!")
    logger.info("All visualizations saved as separate files:")
    for key, path in generated_files.items():
        if isinstance(path, str):
            logger.info(f"  {key}: {Path(path).name}")
    logger.info("=" * 70)
    
    return generated_files


# Legacy function for backward compatibility 
def generate_training_visualizations(model=None, history=None, X_test=None, y_test=None,
                                     federated_history_path=None, results_dir="results",
                                     model_name="DDoS_Detection_Model"):
    """
    Legacy function for backward compatibility with existing code.
    """
    return generate_essential_visualizations(
        model=model,
        history=history,
        X_test=X_test,
        y_test=y_test,
        federated_history=None,  # Will be handled separately for federated cases
        results_dir=results_dir
    )