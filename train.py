#!/usr/bin/env python3
"""
Production Training Script - DDoS Detection
Works with REAL data only - No simulations, no synthetic data
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)


class ProductionTrainer:
    """Production DDoS Detection Trainer"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.history = None

    def load_real_data(self):
        """Load REAL data from clean partitions"""
        logger.info("📂 Loading real data from clean partitions...")

        X_train_list, y_train_list = [], []
        X_test_list, y_test_list = [], []

        for client_id in range(4):
            train_file = f"data/optimized/clean_partitions/client_{client_id}_train.csv"
            test_file = f"data/optimized/clean_partitions/client_{client_id}_test.csv"

            if not os.path.exists(train_file) or not os.path.exists(test_file):
                raise FileNotFoundError(
                    f"Data not found for client {client_id}")

            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file)

            logger.info(
                f"  Client {client_id}: Train={train_df.shape[0]} samples, Test={test_df.shape[0]} samples")
            logger.info(
                f"    Train labels: {dict(train_df['Binary_Label'].value_counts())}")

            # Get feature columns
            feature_cols = [col for col in train_df.columns if col not in [
                'Binary_Label', 'Label']]

            # Extract data
            X_train = train_df[feature_cols].values.astype(np.float32)
            y_train = train_df['Binary_Label'].values.astype(np.int32)

            X_test = test_df[feature_cols].values.astype(np.float32)
            y_test = test_df['Binary_Label'].values.astype(np.int32)

            X_train_list.append(X_train)
            y_train_list.append(y_train)
            X_test_list.append(X_test)
            y_test_list.append(y_test)

        # Combine all clients
        X_train = np.vstack(X_train_list)
        y_train = np.hstack(y_train_list)
        X_test = np.vstack(X_test_list)
        y_test = np.hstack(y_test_list)

        logger.info(f"\n✅ Combined data:")
        logger.info(
            f"   Train: {X_train.shape}, Labels: {np.bincount(y_train)}")
        logger.info(f"   Test: {X_test.shape}, Labels: {np.bincount(y_test)}")

        return X_train, y_train, X_test, y_test

    def preprocess(self, X_train, X_test):
        """Normalize and reshape data"""
        logger.info("🔧 Preprocessing data...")

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        logger.info(f"✅ Shapes: Train={X_train.shape}, Test={X_test.shape}")
        return X_train, X_test

    def build_model(self, input_shape):
        """Build CNN model"""
        logger.info("🏗️ Building CNN model...")

        model = tf.keras.Sequential([
            # Conv Block 1
            tf.keras.layers.Conv1D(
                64, 3, activation='relu', padding='same', input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.MaxPooling1D(2),

            # Conv Block 2
            tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.MaxPooling1D(2),

            # Conv Block 3
            tf.keras.layers.Conv1D(256, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.GlobalAveragePooling1D(),

            # Dense layers
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                     tf.keras.metrics.AUC()]
        )

        logger.info(f"✅ Model built. Parameters: {model.count_params():,}")
        self.model = model
        return model

    def train(self, X_train, y_train, X_test, y_test):
        """Train on real data"""
        logger.info("\n🚀 Starting training...")

        # Class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        logger.info(f"Class weights: {class_weight_dict}")

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=20,
            batch_size=32,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )

        logger.info("✅ Training completed")

    def evaluate(self, X_test, y_test):
        """Evaluate on real test data"""
        logger.info("\n📊 Evaluating model...")

        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)

        logger.info(f"✅ Performance:")
        logger.info(f"   Accuracy:  {acc:.4f}")
        logger.info(f"   Precision: {prec:.4f}")
        logger.info(f"   Recall:    {rec:.4f}")
        logger.info(f"   F1-Score:  {f1:.4f}")
        logger.info(f"   ROC-AUC:   {auc:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\n📋 Confusion Matrix:\n{cm}")
        logger.info(
            f"\n{classification_report(y_test, y_pred, target_names=['Benign', 'Attack'])}")

        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': auc,
            'confusion_matrix': cm.tolist()
        }

    def save_model(self):
        """Save trained model"""
        os.makedirs('results', exist_ok=True)

        model_path = 'results/ddos_model.h5'
        self.model.save(model_path)
        logger.info(f"✅ Model saved: {model_path}")

        import joblib
        scaler_path = 'results/scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"✅ Scaler saved: {scaler_path}")

    def visualize(self, y_test, y_pred_proba, metrics):
        """Create visualizations"""
        logger.info("\n📈 Creating visualizations...")

        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        cm = confusion_matrix(y_test, y_pred)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('DDoS Detection Model Performance',
                     fontsize=16, fontweight='bold')

        # Plot 1: Training history
        if self.history:
            axes[0, 0].plot(self.history.history['accuracy'],
                            label='Train Acc')
            axes[0, 0].plot(
                self.history.history['val_accuracy'], label='Val Acc')
            axes[0, 0].set_title('Accuracy')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                    xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
        axes[0, 1].set_title('Confusion Matrix')
        axes[0, 1].set_ylabel('True Label')
        axes[0, 1].set_xlabel('Predicted Label')

        # Plot 3: ROC Curve
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        axes[1, 0].plot(
            fpr, tpr, label=f'ROC (AUC={roc_auc:.3f})', linewidth=2)
        axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        axes[1, 0].set_title('ROC Curve')
        axes[1, 0].set_xlabel('FPR')
        axes[1, 0].set_ylabel('TPR')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Metrics
        axes[1, 1].axis('off')
        metrics_text = f"""
        PERFORMANCE METRICS
        {'='*30}
        Accuracy:  {metrics['accuracy']:.4f}
        Precision: {metrics['precision']:.4f}
        Recall:    {metrics['recall']:.4f}
        F1-Score:  {metrics['f1']:.4f}
        ROC-AUC:   {metrics['roc_auc']:.4f}
        
        CONFUSION MATRIX
        {'='*30}
        TN: {cm[0, 0]:,} | FP: {cm[0, 1]:,}
        FN: {cm[1, 0]:,} | TP: {cm[1, 1]:,}
        """
        axes[1, 1].text(0.1, 0.5, metrics_text,
                        fontsize=11, family='monospace')

        plt.tight_layout()
        plt.savefig('results/training_results.png',
                    dpi=300, bbox_inches='tight')
        logger.info("✅ Visualization saved: results/training_results.png")
        plt.close()


def main():
    try:
        logger.info("="*70)
        logger.info("🔥 PRODUCTION DDOS DETECTION TRAINING")
        logger.info("="*70)

        trainer = ProductionTrainer()

        # Load real data
        X_train, y_train, X_test, y_test = trainer.load_real_data()

        # Preprocess
        X_train, X_test = trainer.preprocess(X_train, X_test)

        # Build model
        trainer.build_model((X_train.shape[1], X_train.shape[2]))

        # Train
        trainer.train(X_train, y_train, X_test, y_test)

        # Evaluate
        metrics = trainer.evaluate(X_test, y_test)

        # Visualize
        y_pred_proba = trainer.model.predict(X_test, verbose=0)
        trainer.visualize(y_test, y_pred_proba, metrics)

        # Save
        trainer.save_model()

        # Save metrics
        with open('results/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info("\n" + "="*70)
        logger.info("✅ TRAINING COMPLETE!")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
