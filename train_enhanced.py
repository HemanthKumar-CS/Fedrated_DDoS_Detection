#!/usr/bin/env python3
"""
Enhanced Training Script with Strategic Improvements
Addresses overfitting, class imbalance, and attack detection recall issues
"""

from src.models.cnn_model import DDoSCNNModel
import os
import sys
import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Add src to path
sys.path.append('src')

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedTrainer:
    """Enhanced trainer with strategic improvements"""

    def __init__(self):
        self.scaler = None
        self.model = None
        self.history = None
        self.class_weights = None

    def load_and_prepare_data(self):
        """Load data from client partitions and combine them"""
        logger.info("🔄 Loading and preparing data from client partitions...")

        # Check for client partition files
        client_files = []
        for i in range(4):  # Check for 4 clients
            train_file = f"data/optimized/clean_partitions/client_{i}_train.csv"
            test_file = f"data/optimized/clean_partitions/client_{i}_test.csv"
            if os.path.exists(train_file) and os.path.exists(test_file):
                client_files.append((train_file, test_file))

        if not client_files:
            # Fallback: try to find realistic datasets only
            potential_files = [
                # Preferred realistic combined dataset
                "data/optimized/realistic_balanced_dataset.csv",
                "data/optimized/realistic_train.csv",
                "data/optimized/realistic_test.csv"
            ]

            for file_path in potential_files:
                if os.path.exists(file_path):
                    logger.info(f"Found dataset: {file_path}")
                    df = pd.read_csv(file_path)

                    # Split features and labels
                    feature_cols = [col for col in df.columns if col not in [
                        'Label', 'Binary_Label']]
                    X = df[feature_cols].values
                    y = df['Binary_Label'].values

                    logger.info(f"Dataset shape: {X.shape}")
                    logger.info(
                        f"Label distribution: {pd.Series(y).value_counts().to_dict()}")

                    # Enhanced train/validation/test split
                    X_temp, X_test, y_temp, y_test = train_test_split(
                        X, y, test_size=0.15, random_state=42, stratify=y
                    )

                    X_train, X_val, y_train, y_val = train_test_split(
                        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
                    )

                    break
            else:
                # No realistic dataset found; abort rather than creating synthetic data
                logger.error(
                    "No realistic dataset found. Expected one of: data/optimized/realistic_balanced_dataset.csv, realistic_train.csv, realistic_test.csv or client partition files under data/optimized/clean_partitions/.")
                raise FileNotFoundError("Realistic dataset not found")
        else:
            # Load from client partitions
            logger.info(
                f"Loading data from {len(client_files)} client partitions...")

            train_dfs = []
            test_dfs = []

            for train_file, test_file in client_files:
                train_df = pd.read_csv(train_file)
                test_df = pd.read_csv(test_file)
                train_dfs.append(train_df)
                test_dfs.append(test_df)

            # Combine all client data
            combined_train = pd.concat(train_dfs, ignore_index=True)
            combined_test = pd.concat(test_dfs, ignore_index=True)

            # Split features and labels
            feature_cols = [col for col in combined_train.columns if col not in [
                'Label', 'Binary_Label']]

            X_temp = combined_train[feature_cols].values
            y_temp = combined_train['Binary_Label'].values
            X_test = combined_test[feature_cols].values
            y_test = combined_test['Binary_Label'].values

            logger.info(f"Combined train shape: {X_temp.shape}")
            logger.info(f"Combined test shape: {X_test.shape}")
            logger.info(
                f"Train label distribution: {pd.Series(y_temp).value_counts().to_dict()}")
            logger.info(
                f"Test label distribution: {pd.Series(y_test).value_counts().to_dict()}")

            # Create validation split from training data
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
            )

        logger.info(
            f"Final splits - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

        # Robust scaling (better for outliers than StandardScaler)
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # Reshape for CNN (add channel dimension)
        X_train_cnn = X_train_scaled.reshape(
            X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
        X_val_cnn = X_val_scaled.reshape(
            X_val_scaled.shape[0], X_val_scaled.shape[1], 1)
        X_test_cnn = X_test_scaled.reshape(
            X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

        # Compute class weights to handle imbalance
        self.class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: self.class_weights[i]
                             for i in range(len(self.class_weights))}
        logger.info(f"Class weights: {class_weight_dict}")

        return (X_train_cnn, X_val_cnn, X_test_cnn), (y_train, y_val, y_test), class_weight_dict

    # Removed synthetic data generator to enforce realistic-only workflow

    def create_enhanced_model(self, input_features):
        """Create enhanced CNN model with improvements"""
        logger.info("🔧 Building enhanced CNN model...")

        inputs = tf.keras.Input(
            shape=(input_features, 1), name="traffic_features")

        # Enhanced architecture with residual connections and attention

        # First Conv Block with more filters
        x1 = tf.keras.layers.Conv1D(
            filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        x1 = tf.keras.layers.BatchNormalization()(x1)
        x1 = tf.keras.layers.Dropout(0.2)(x1)

        # Second Conv Block
        x2 = tf.keras.layers.Conv1D(
            filters=128, kernel_size=3, activation='relu', padding='same')(x1)
        x2 = tf.keras.layers.BatchNormalization()(x2)
        x2 = tf.keras.layers.MaxPooling1D(pool_size=2)(x2)
        x2 = tf.keras.layers.Dropout(0.3)(x2)

        # Third Conv Block with residual connection
        x3 = tf.keras.layers.Conv1D(
            filters=256, kernel_size=3, activation='relu', padding='same')(x2)
        x3 = tf.keras.layers.BatchNormalization()(x3)

        # Attention mechanism (simplified)
        attention = tf.keras.layers.GlobalAveragePooling1D()(x3)
        attention = tf.keras.layers.Dense(256, activation='sigmoid')(attention)
        attention = tf.keras.layers.Reshape((1, 256))(attention)
        x3_attended = tf.keras.layers.Multiply()([x3, attention])

        # Global pooling with both max and average
        global_max = tf.keras.layers.GlobalMaxPooling1D()(x3_attended)
        global_avg = tf.keras.layers.GlobalAveragePooling1D()(x3_attended)
        x = tf.keras.layers.Concatenate()([global_max, global_avg])

        # Enhanced dense layers with progressive dropout
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        # Output layer with bias initialization for imbalanced data
        output_bias = np.log(0.5 / 0.5)  # Adjust based on your class ratio
        outputs = tf.keras.layers.Dense(1, activation='sigmoid',
                                        bias_initializer=tf.keras.initializers.Constant(output_bias))(x)

        model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name="Enhanced_DDoS_CNN")

        # Enhanced optimizer with learning rate scheduling
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=0.001,
            weight_decay=0.01  # L2 regularization
        )

        # Focal loss for handling class imbalance
        def focal_loss(alpha=0.75, gamma=2.0):
            def focal_loss_fixed(y_true, y_pred):
                epsilon = tf.keras.backend.epsilon()
                y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

                # Calculate Cross Entropy
                ce = -y_true * tf.math.log(y_pred)

                # Calculate Focal Weight
                weight = alpha * y_true * tf.pow((1 - y_pred), gamma)

                # Calculate Focal Loss
                fl = weight * ce

                return tf.reduce_mean(fl)
            return focal_loss_fixed

        model.compile(
            optimizer=optimizer,
            # Focal loss for imbalanced data
            loss=focal_loss(alpha=0.75, gamma=2.0),
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.AUC(name='pr_auc', curve='PR')
            ]
        )

        logger.info(
            f"✅ Enhanced model created with {model.count_params()} parameters")
        return model

    def train_with_enhancements(self, X_data, y_data, class_weights):
        """Train with enhanced strategies"""
        X_train, X_val, X_test = X_data
        y_train, y_val, y_test = y_data

        # Create enhanced model
        self.model = self.create_enhanced_model(X_train.shape[1])

        # Enhanced callbacks
        callbacks = [
            # Early stopping with patience
            tf.keras.callbacks.EarlyStopping(
                monitor='val_recall',
                patience=15,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),

            # Learning rate reduction
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),

            # Model checkpoint
            tf.keras.callbacks.ModelCheckpoint(
                filepath='results/best_enhanced_model.keras',
                monitor='val_recall',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]

        logger.info("🚀 Starting enhanced training...")

        # Train with class weights and validation
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,  # More epochs with early stopping
            batch_size=32,  # Smaller batch size for better gradients
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate on test set
        logger.info("📊 Evaluating on test set...")
        test_results = self.model.evaluate(X_test, y_test, verbose=0)

        # Predictions for detailed analysis
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # Find optimal threshold using validation set
        val_pred_proba = self.model.predict(X_val)
        precision_vals, recall_vals, thresholds = precision_recall_curve(
            y_val, val_pred_proba)
        f1_scores = 2 * (precision_vals * recall_vals) / \
            (precision_vals + recall_vals + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]

        logger.info(f"📈 Optimal threshold found: {optimal_threshold:.4f}")

        # Apply optimal threshold
        y_pred_optimal = (
            y_pred_proba > optimal_threshold).astype(int).flatten()

        # Comprehensive evaluation
        results = {
            'test_metrics': {
                'accuracy': float(accuracy_score(y_test, y_pred_optimal)),
                'precision': float(precision_score(y_test, y_pred_optimal)),
                'recall': float(recall_score(y_test, y_pred_optimal)),
                'f1': float(f1_score(y_test, y_pred_optimal)),
                'optimal_threshold': float(optimal_threshold)
            },
            'confusion_matrix': confusion_matrix(y_test, y_pred_optimal).tolist(),
            'classification_report': classification_report(y_test, y_pred_optimal, output_dict=True),
            'model_params': int(self.model.count_params()),
            'training_epochs': len(self.history.history['loss']),
            'class_weights': class_weights
        }

        return results

    def create_visualizations(self, results, y_test, y_pred_proba):
        """Create enhanced visualizations"""
        logger.info("📊 Creating enhanced visualizations...")

        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Enhanced Model Performance Analysis',
                     fontsize=16, fontweight='bold')

        # 1. Training History
        if self.history:
            ax1 = axes[0, 0]
            ax1.plot(self.history.history['loss'],
                     label='Training Loss', linewidth=2)
            ax1.plot(self.history.history['val_loss'],
                     label='Validation Loss', linewidth=2)
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. Accuracy History
            ax2 = axes[0, 1]
            ax2.plot(self.history.history['accuracy'],
                     label='Training Accuracy', linewidth=2)
            ax2.plot(self.history.history['val_accuracy'],
                     label='Validation Accuracy', linewidth=2)
            ax2.set_title('Training and Validation Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 3. Recall History
            ax3 = axes[0, 2]
            ax3.plot(self.history.history['recall'],
                     label='Training Recall', linewidth=2)
            ax3.plot(self.history.history['val_recall'],
                     label='Validation Recall', linewidth=2)
            ax3.set_title('Training and Validation Recall')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Recall')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. Confusion Matrix
        ax4 = axes[1, 0]
        cm = np.array(results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
        ax4.set_title('Confusion Matrix')
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Actual')

        # 5. ROC Curve
        ax5 = axes[1, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        ax5.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax5.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax5.set_xlim([0.0, 1.0])
        ax5.set_ylim([0.0, 1.05])
        ax5.set_xlabel('False Positive Rate')
        ax5.set_ylabel('True Positive Rate')
        ax5.set_title('ROC Curve')
        ax5.legend(loc="lower right")
        ax5.grid(True, alpha=0.3)

        # 6. Precision-Recall Curve
        ax6 = axes[1, 2]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        ax6.plot(recall, precision, color='blue', lw=2,
                 label=f'PR curve (AUC = {pr_auc:.3f})')
        ax6.set_xlabel('Recall')
        ax6.set_ylabel('Precision')
        ax6.set_title('Precision-Recall Curve')
        ax6.legend(loc="lower left")
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/enhanced_training_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

        logger.info(
            "✅ Visualizations saved to results/enhanced_training_analysis.png")


def main():
    """Main training function"""
    logger.info("🎯 Starting Enhanced Training with Strategic Improvements")
    logger.info("=" * 60)

    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    # Initialize trainer
    trainer = EnhancedTrainer()

    try:
        # Load and prepare data
        X_data, y_data, class_weights = trainer.load_and_prepare_data()

        # Train with enhancements
        results = trainer.train_with_enhancements(
            X_data, y_data, class_weights)

        # Create visualizations
        X_train, X_val, X_test = X_data
        y_train, y_val, y_test = y_data
        y_pred_proba = trainer.model.predict(X_test)
        trainer.create_visualizations(results, y_test, y_pred_proba)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'results/enhanced_training_results_{timestamp}.json'

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Print summary
        logger.info("🎉 Enhanced Training Complete!")
        logger.info("=" * 60)
        logger.info(
            f"📊 Final Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
        logger.info(
            f"📊 Final Test Precision: {results['test_metrics']['precision']:.4f}")
        logger.info(
            f"📊 Final Test Recall: {results['test_metrics']['recall']:.4f}")
        logger.info(
            f"📊 Final Test F1-Score: {results['test_metrics']['f1']:.4f}")
        logger.info(
            f"📊 Optimal Threshold: {results['test_metrics']['optimal_threshold']:.4f}")
        logger.info(f"📊 Model Parameters: {results['model_params']:,}")
        logger.info(f"📊 Training Epochs: {results['training_epochs']}")
        logger.info(f"💾 Results saved to: {results_file}")

        return results

    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    results = main()
