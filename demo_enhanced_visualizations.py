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
from src.visualization.training_visualizer import FederatedTrainingVisualizer
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

    # Initialize enhanced visualizer
    logger.info("🎨 Initializing Enhanced Visualization System...")
    visualizer = FederatedTrainingVisualizer(results_dir="results")

    # Check if federated history exists
    fed_history_path = "results/federated_metrics_history.json"
    if not Path(fed_history_path).exists():
        logger.info("📊 Creating demo federated history...")
        # Create demo federated history
        demo_fed_history = {
            "train_accuracy": [0.75, 0.82, 0.86, 0.88, 0.90, 0.91, 0.92, 0.92, 0.93, 0.93],
            "test_accuracy": [0.72, 0.78, 0.81, 0.83, 0.85, 0.86, 0.86, 0.87, 0.87, 0.87]
        }

        import json
        with open(fed_history_path, 'w') as f:
            json.dump(demo_fed_history, f, indent=2)

    # Generate all enhanced visualizations
    logger.info("🚀 Generating comprehensive visualization suite...")

    try:
        plots = visualizer.generate_all_plots(
            model=model,
            history=history,
            X_test=X_test,
            y_test=y_test,
            federated_history_path=fed_history_path,
            model_name="Demo_Enhanced_DDoS_CNN"
        )

        logger.info("✅ Enhanced visualization demo completed successfully!")
        logger.info("=" * 60)
        logger.info(
            f"📊 Generated {len(plots)} comprehensive visualization plots:")

        for plot in plots:
            plot_name = Path(plot).name
            logger.info(f"   📈 {plot_name}")

        logger.info("")
        logger.info("🔍 Enhanced Visualization Features Demonstrated:")
        logger.info("   📊 Enhanced Training History Analysis")
        logger.info("     - Loss/accuracy curves with trend analysis")
        logger.info("     - Generalization gap analysis")
        logger.info("     - Learning rate scheduling visualization")
        logger.info("     - Precision/recall tracking")
        logger.info("     - Training efficiency metrics")
        logger.info("")
        logger.info("   🎯 Comprehensive Model Performance Analysis")
        logger.info("     - Detailed confusion matrix with percentages")
        logger.info("     - ROC curve with optimal threshold marking")
        logger.info("     - Precision-recall curve analysis")
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
        logger.info("     - Performance heatmaps")
        logger.info("     - Statistical summaries")
        logger.info("     - Learning stability analysis")
        logger.info("")
        logger.info("   📋 Detailed Summary Dashboard")
        logger.info("     - Comprehensive performance metrics")
        logger.info("     - Security-focused analysis")
        logger.info("     - Federated learning insights")
        logger.info("     - Actionable recommendations")
        logger.info("")
        logger.info("   📝 Performance Reports")
        logger.info("     - Detailed markdown reports")
        logger.info("     - Executive summaries")
        logger.info("     - Technical recommendations")

        logger.info("")
        logger.info(f"📁 All visualizations saved to: results/visualizations/")
        logger.info(
            "🎉 Demo completed! Check the generated plots for enhanced insights.")

        return plots

    except Exception as e:
        logger.error(f"❌ Error during visualization demo: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    """Main demo function."""
    print("🎨 Enhanced DDoS Detection Visualization Demo")
    print("=" * 60)
    print("")
    print("This demo showcases the advanced visualization capabilities")
    print("that have been added to the federated DDoS detection system.")
    print("")
    print("Features demonstrated:")
    print("• Enhanced training history analysis")
    print("• Comprehensive model performance metrics")
    print("• Advanced federated learning insights")
    print("• Detailed summary dashboards")
    print("• Performance reports")
    print("")

    input("Press Enter to start the demo...")

    plots = demo_enhanced_visualizations()

    if plots:
        print("")
        print("✅ Demo completed successfully!")
        print(f"📊 Generated {len(plots)} visualization plots")
        print("📁 Check results/visualizations/ for all generated files")
        print("")
        print("🔍 Key improvements over basic visualizations:")
        print("• Much more detailed confusion matrix analysis")
        print("• Advanced threshold optimization")
        print("• Security-focused metrics (attack detection, false alarms)")
        print("• Comprehensive federated learning insights")
        print("• Professional summary dashboards")
        print("• Actionable recommendations")
    else:
        print("❌ Demo encountered errors. Check the logs above.")


if __name__ == "__main__":
    main()
