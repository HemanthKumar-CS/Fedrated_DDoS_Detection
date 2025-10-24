#!/usr/bin/env python3
"""
Demo script to showcase the enhanced visualization capabilities.

This script demonstrates the comprehensive visualization features that have been
added to the federated DDoS detection system, including:

1. Enhanced Training History Analysis
2. Comprehensive Model Performance Analysis 
3. Advanced Federated Learning Analysis
4. Detailed Summary Dashboard
5. Performance Reports

Usage:
    python demo_enhanced_visualizations.py
"""

import tensorflow as tf
from src.visualization.training_visualizer import generate_essential_visualizations
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Add src to path
sys.path.append('src')


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_demo_model():
    """Create a demo CNN model for visualization testing."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(29, 1)),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_demo_training_history():
    """Create realistic demo training history."""
    epochs = 25

    # Simulate realistic training curves
    base_acc = 0.6
    noise = 0.02

    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []

    for epoch in range(epochs):
        # Simulate learning curve with some noise
        progress = epoch / epochs

        # Training accuracy improves with slight overfitting towards end
        ta = base_acc + 0.35 * progress + np.random.normal(0, noise)
        train_acc.append(min(0.98, max(0.5, ta)))

        # Validation accuracy improves but plateaus earlier
        va = base_acc + 0.3 * progress * \
            (1 - 0.3 * progress) + np.random.normal(0, noise * 1.5)
        val_acc.append(min(0.95, max(0.5, va)))

        # Loss decreases
        tl = 0.7 * np.exp(-progress * 2) + 0.1 + np.random.normal(0, noise)
        train_loss.append(max(0.05, tl))

        vl = 0.8 * np.exp(-progress * 1.8) + 0.15 + \
            np.random.normal(0, noise * 1.2)
        val_loss.append(max(0.1, vl))

    # Create mock history object
    class MockHistory:
        def __init__(self):
            self.history = {
                'accuracy': train_acc,
                'val_accuracy': val_acc,
                'loss': train_loss,
                'val_loss': val_loss,
                'precision': [min(0.95, max(0.5, acc + np.random.normal(0, 0.01))) for acc in train_acc],
                'recall': [min(0.95, max(0.5, acc + np.random.normal(0, 0.01))) for acc in train_acc],
                'lr': [0.001 * (0.5 ** (epoch // 8)) for epoch in range(epochs)]
            }

    return MockHistory()


def create_demo_test_data():
    """Create realistic demo test data."""
    n_samples = 5000
    n_features = 29

    # Create somewhat realistic network traffic features
    np.random.seed(42)

    # Generate features that might represent network traffic
    X_test = np.random.randn(n_samples, n_features, 1)

    # Create binary labels with some class imbalance (typical for DDoS detection)
    y_test = np.random.binomial(1, 0.3, n_samples)  # 30% attack traffic

    # Make the features somewhat correlated with labels for realistic predictions
    for i in range(n_samples):
        if y_test[i] == 1:  # Attack traffic
            X_test[i] += 0.5  # Shift attack features slightly

    return X_test, y_test


def demo_enhanced_visualizations():
    """Demonstrate the enhanced visualization capabilities."""

    logger.info("🎨 Starting Enhanced Visualization Demo")
    logger.info("=" * 60)

    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    # Create demo components
    logger.info("🔧 Creating demo model and data...")
    model = create_demo_model()
    history = create_demo_training_history()
    X_test, y_test = create_demo_test_data()

    # Generate essential visualizations
    logger.info("🚀 Generating essential visualizations...")

    try:
        plots = generate_essential_visualizations(
            model=model,
            history=history,
            X_test=X_test,
            y_test=y_test,
            federated_history=None,  # Could add federated history object here
            results_dir="results"
        )

        logger.info("✅ Essential visualization demo completed successfully!")
        logger.info("=" * 60)
        logger.info(f"📊 Generated essential visualization plots:")

        for key, path in plots.items():
            if isinstance(path, str):
                plot_name = Path(path).name
                logger.info(f"   📈 {plot_name}")

        logger.info("")
        logger.info("🔍 Essential Visualization Features Demonstrated:")
        logger.info("   📊 Training and Testing Accuracy")
        logger.info("   📊 Training and Testing Loss") 
        logger.info("   📊 Classification Report")
        logger.info("   📊 Confusion Matrix with Percentages")
        logger.info("   📊 ROC Curve")
        logger.info("   📊 Federated Learning Stats (if available)")
        logger.info("   📊 Federated Learning Progress with Trend")
        logger.info("   📊 Convergence Analysis")
        logger.info("     - Prediction score distribution")
        logger.info("     - Threshold analysis and optimization")
        logger.info("     - Class-wise performance comparison")
        logger.info("     - Model confidence analysis")
        logger.info("     - Error analysis (false positives/negatives)")
        logger.info("")
        logger.info("   🌐 Advanced Federated Learning Analysis")
        logger.info("     - Round-by-round progress with trends")
        logger.info("     - Convergence analysis")
        logger.info("     - Generalization gap tracking")
        logger.info("")
        logger.info(f" All visualizations saved to: results/")
        logger.info("🎉 Demo completed! Check the generated plots for essential insights.")

        return plots

    except Exception as e:
        logger.error(f"❌ Error during visualization demo: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    """Main demo function."""
    print("🎨 Essential DDoS Detection Visualization Demo")
    print("=" * 60)
    print("")
    print("This demo showcases the essential visualization capabilities")
    print("for the federated DDoS detection system.")
    print("")
    print("Features demonstrated:")
    print("• Training and testing accuracy curves")
    print("• Training and testing loss curves")
    print("• Classification report visualization")
    print("• Confusion matrix with percentages")
    print("• ROC curve analysis")
    print("• Federated learning statistics")
    print("• Federated learning progress with trend")
    print("• Convergence analysis")
    print("")

    input("Press Enter to start the demo...")

    plots = demo_enhanced_visualizations()

    if plots:
        print("")
        print("✅ Demo completed successfully!")
        print(f"📊 Generated essential visualization plots")
        print("📁 Check results/ for all generated files")
        print("")
        print("🔍 Essential visualizations include:")
        print("• 01_accuracy_curves.png - Training/Testing Accuracy")
        print("• 02_loss_curves.png - Training/Testing Loss")
        print("• 03_classification_report.png - Per-class Metrics")
        print("• 04_confusion_matrix.png - Confusion Matrix with %")
        print("• 05_roc_curve.png - ROC Curve")
        print("• 06_federated_accuracy.png - Federated Accuracy")
        print("• 07_federated_loss.png - Federated Loss")
        print("• 08_federated_progress_trend.png - Progress with Trend")
        print("• 09_convergence_analysis.png - Convergence Analysis")
    else:
        print("❌ Demo encountered errors. Check the logs above.")


if __name__ == "__main__":
    main()
