#!/usr/bin/env python3
"""
Centralized Training with Balanced Dataset Trains a CNN model on the balanced dataset (with both benign and attack samples)
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append('src')
from src.models.cnn_model import create_ddos_cnn_model

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_balanced_data():
    """Load the balanced dataset"""
    logger.info("Loading balanced dataset...")
    
    balanced_file = "data/optimized/balanced_dataset.csv"
    if not os.path.exists(balanced_file):
        raise FileNotFoundError(f"Balanced dataset not found at {balanced_file}")
    
    df = pd.read_csv(balanced_file)
    
    # Split features and labels
    feature_cols = [col for col in df.columns if col not in ['Label', 'Binary_Label']]
    X = df[feature_cols].values
    y = df['Binary_Label'].values
    
    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Label distribution: {pd.Series(y).value_counts().to_dict()}")
    
    # Split into train/test with stratification to maintain balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    logger.info(f"Train labels: {pd.Series(y_train).value_counts().to_dict()}")
    logger.info(f"Test labels: {pd.Series(y_test).value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test

def preprocess_data(X_train, X_test):
    """Normalize and reshape data for CNN"""
    logger.info("Preprocessing data...")
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reshape for 1D CNN (samples, features, 1)
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    logger.info(f"Data shape after preprocessing: {X_train_reshaped.shape}")
    
    return X_train_reshaped, X_test_reshaped, scaler

def plot_training_results(history, y_test, y_pred_proba, results):
    """Plot training curves and ROC curve"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Federated DDoS Detection - Training Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Training & Validation Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0, 0].set_title('Model Accuracy Over Epochs', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0.8, 1.0])
    
    # Plot 2: Training & Validation Loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 1].set_title('Model Loss Over Epochs', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                    label='Random Classifier')
    axes[1, 0].set_xlim([0.0, 1.0])
    axes[1, 0].set_ylim([0.0, 1.05])
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('ROC Curve', fontweight='bold')
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Performance Metrics Bar Chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [results['test_accuracy'], results['test_precision'], 
              results['test_recall'], results['test_f1_score']]
    
    bars = axes[1, 1].bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[1, 1].set_title('Performance Metrics', fontweight='bold')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_ylim([0.8, 1.0])
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/training_results_visualization.png", dpi=300, bbox_inches='tight')
    logger.info("Training visualization saved to results/training_results_visualization.png")
    
    # Show the plot
    plt.show()
    
    return roc_auc

def train_model():
    """Main training function"""
    logger.info("🏋️ Starting Centralized Training with Balanced Dataset")
    logger.info("=" * 80)
    
    try:
        # Load data
        X_train, X_test, y_train, y_test = load_balanced_data()
        
        # Preprocess data
        X_train_processed, X_test_processed, scaler = preprocess_data(X_train, X_test)
        
        # Create model
        logger.info("Creating CNN model...")
        model = create_ddos_cnn_model(input_features=X_train_processed.shape[1], num_classes=1)
        cnn_model = model.get_model()
        
        # Print model architecture
        logger.info("Model architecture:")
        cnn_model.summary()
        
        # Prepare callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Training
        logger.info("Starting training...")
        start_time = datetime.now()
        
        history = cnn_model.fit(
            X_train_processed, y_train,
            batch_size=160,
            epochs=50,
            validation_data=(X_test_processed, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = datetime.now() - start_time
        logger.info(f"Training completed in: {training_time}")
        
        # Evaluation
        logger.info("Evaluating model...")
        
        # Predictions
        y_pred_proba = cnn_model.predict(X_test_processed)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred)
        test_recall = recall_score(y_test, y_pred)
        f1_score_val = f1_score(y_test, y_pred)
        test_loss = cnn_model.evaluate(X_test_processed, y_test, verbose=0)[0]
        
        # Save detailed results
        results = {
            'training_time': str(training_time),
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1_score': f1_score_val,
            'test_loss': test_loss,
            'dataset_info': {
                'total_samples': len(X_train) + len(X_test),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features': X_train.shape[1],
                'train_benign': int((y_train == 0).sum()),
                'train_attack': int((y_train == 1).sum()),
                'test_benign': int((y_test == 0).sum()),
                'test_attack': int((y_test == 1).sum())
            }
        }
        
        # Plot training results and ROC curve
        logger.info("Generating training visualization...")
        roc_auc = plot_training_results(history, y_test, y_pred_proba.flatten(), results)
        results['roc_auc'] = roc_auc
        
        logger.info("🎉 Training completed!")
        logger.info("=" * 80)
        logger.info("BALANCED CENTRALIZED MODEL RESULTS:")
        logger.info(f"Training time: {training_time}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test Precision: {test_precision:.4f}")
        logger.info(f"Test Recall: {test_recall:.4f}")
        logger.info(f"Test F1-score: {f1_score_val:.4f}")
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"ROC AUC: {roc_auc:.4f}")
        
        # Detailed classification report
        logger.info("\nDetailed Classification Report:")
        unique_test_labels = np.unique(y_test)
        unique_pred_labels = np.unique(y_pred)
        logger.info(f"Unique test labels: {unique_test_labels}")
        logger.info(f"Unique predicted labels: {unique_pred_labels}")
        
        # Create target names based on actual labels present
        if len(unique_test_labels) == 2 and len(unique_pred_labels) == 2:
            target_names = ['Benign', 'Attack']
        elif len(unique_test_labels) == 1:
            target_names = ['Benign'] if unique_test_labels[0] == 0 else ['Attack']
        else:
            target_names = None
            
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Confusion matrix
        logger.info("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Save model and results
        os.makedirs("results", exist_ok=True)
        cnn_model.save("results/balanced_centralized_model.h5")
        logger.info("Model saved to results/balanced_centralized_model.h5")
        
        import json
        with open("results/balanced_training_results.json", "w") as f:
            json.dump(results, f, indent=4)
        
        logger.info("Results saved to results/balanced_training_results.json")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Ensure proper GPU configuration if available
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    results = train_model()
