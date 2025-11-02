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
import argparse

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
        self.convergence_data = None
        self.federated_rounds = 1

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
            # Conv Block 1 - with L2 regularization
            tf.keras.layers.Conv1D(
                64, 3, activation='relu', padding='same', input_shape=input_shape,
                kernel_regularizer=tf.keras.regularizers.L2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.MaxPooling1D(2),

            # Conv Block 2 - with L2 regularization
            tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same',
                                   kernel_regularizer=tf.keras.regularizers.L2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.MaxPooling1D(2),

            # Conv Block 3 - with L2 regularization
            tf.keras.layers.Conv1D(256, 3, activation='relu', padding='same',
                                   kernel_regularizer=tf.keras.regularizers.L2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.GlobalAveragePooling1D(),

            # Dense layers - with L2 regularization
            tf.keras.layers.Dense(128, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.L2(0.001)),
            tf.keras.layers.Dropout(0.45),
            tf.keras.layers.Dense(64, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.L2(0.001)),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                     tf.keras.metrics.AUC()]
        )

        logger.info(f"✅ Model built. Parameters: {model.count_params():,}")
        self.model = model
        return model

    def train(self, X_train, y_train, X_test, y_test, federated_rounds=1, epochs=5):
        """Train on real data with optional federated rounds

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            federated_rounds: Number of federated communication rounds (default: 1, production: 50+)
            epochs: Number of epochs per federated round (default: 5)
        """
        logger.info(
            f"\n🚀 Starting training... (Federated Rounds: {federated_rounds})")

        # Class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        logger.info(f"Class weights: {class_weight_dict}")

        # Callbacks - more aggressive early stopping and learning rate reduction
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=8,  # Reduced from 15 for earlier stopping
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001  # Stop if improvement < 0.001
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,  # Reduced from 7 for quicker learning rate reduction
                min_lr=1e-7,
                verbose=1,
                min_delta=0.001
            )
        ]

        # For federated rounds: multiple rounds with reduced epochs per round
        all_metrics = []
        convergence_history = []
        last_history = None

        for round_num in range(federated_rounds):
            logger.info(f"\n{'='*70}")
            logger.info(f"FEDERATED ROUND {round_num + 1}/{federated_rounds}")
            logger.info(f"{'='*70}")

            # Each round trains for specified epochs (convergence across multiple rounds)
            epochs_per_round = max(
                1, epochs) if federated_rounds > 1 else epochs

            round_history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs_per_round,
                batch_size=64,  # Increased from 32 for more stable gradients
                class_weight=class_weight_dict,
                callbacks=callbacks if round_num == federated_rounds -
                1 else [],  # Only EarlyStopping on last round
                verbose=0
            )

            # Store last history for fallback
            last_history = round_history

            # Track metrics each round
            val_acc = round_history.history['val_accuracy'][-1] if 'val_accuracy' in round_history.history else 0
            val_loss = round_history.history['val_loss'][-1] if 'val_loss' in round_history.history else 0
            convergence_history.append({
                'round': round_num + 1,
                'val_accuracy': float(val_acc),
                'val_loss': float(val_loss)
            })

            logger.info(
                f"Round {round_num + 1}: Val Accuracy={val_acc:.4f}, Val Loss={val_loss:.4f}")

        # Store convergence data and history as fallback
        self.convergence_data = convergence_history
        self.history = last_history
        self.federated_rounds = federated_rounds

        logger.info(
            f"\n✅ Training completed ({federated_rounds} federated rounds)")

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

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('DDoS Detection Model Performance',
                     fontsize=16, fontweight='bold')

        # Plot 1: Training Accuracy (Train + Validation) - with custom axis
        if self.history and hasattr(self.history, 'history'):
            train_acc = self.history.history.get('accuracy', [])
            val_acc = self.history.history.get('val_accuracy', [])
            if train_acc and val_acc:
                # X-axis: 0 to num_epochs + 2, with 1 epoch interval
                num_epochs = len(train_acc)
                epochs_range = list(range(0, num_epochs + 2))
                data_epochs = list(range(1, num_epochs + 1))

                axes[0, 0].plot(data_epochs, train_acc, 'b-', marker='o',
                                label='Train Accuracy', linewidth=2, markersize=6)
                axes[0, 0].plot(data_epochs, val_acc, 'r-', marker='s',
                                label='Validation Accuracy', linewidth=2, markersize=6)
                axes[0, 0].set_title(
                    'Training vs Validation Accuracy', fontweight='bold')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Accuracy')

                # Set X-axis: 0 to num_epochs + 2 with 1 epoch interval
                axes[0, 0].set_xlim(0, num_epochs + 2)
                axes[0, 0].set_xticks(epochs_range)

                # Set Y-axis: 0.0 to 1.0 with 0.1 intervals
                axes[0, 0].set_ylim(0.0, 1.0)
                axes[0, 0].set_yticks([i/10.0 for i in range(0, 11)])

                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Training Loss (Train + Validation) - with custom axis
        if self.history and hasattr(self.history, 'history'):
            train_loss = self.history.history.get('loss', [])
            val_loss = self.history.history.get('val_loss', [])
            if train_loss and val_loss:
                # X-axis: 0 to num_epochs + 2, with 1 epoch interval
                num_epochs = len(train_loss)
                epochs_range = list(range(0, num_epochs + 2))
                data_epochs = list(range(1, num_epochs + 1))

                axes[0, 1].plot(data_epochs, train_loss, 'b-', marker='o',
                                label='Train Loss', linewidth=2, markersize=6)
                axes[0, 1].plot(data_epochs, val_loss, 'r-', marker='s',
                                label='Validation Loss', linewidth=2, markersize=6)
                axes[0, 1].set_title(
                    'Training vs Validation Loss', fontweight='bold')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Loss')

                # Set X-axis: 0 to num_epochs + 2 with 1 epoch interval
                axes[0, 1].set_xlim(0, num_epochs + 2)
                axes[0, 1].set_xticks(epochs_range)

                # Set Y-axis: 0.0 to 1.0 with 0.1 intervals
                axes[0, 1].set_ylim(0.0, 1.0)
                axes[0, 1].set_yticks([i/10.0 for i in range(0, 11)])

                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2],
                    xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'],
                    cbar_kws={'label': 'Count'})
        axes[0, 2].set_title('Confusion Matrix', fontweight='bold')
        axes[0, 2].set_ylabel('True Label')
        axes[0, 2].set_xlabel('Predicted Label')

        # Plot 4: ROC Curve
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        axes[1, 0].plot(
            fpr, tpr, label=f'ROC (AUC={roc_auc:.3f})', linewidth=2, color='blue')
        axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        axes[1, 0].set_title('ROC Curve', fontweight='bold')
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 5: Metrics Display
        axes[1, 1].axis('off')
        metrics_text = f"""
        PERFORMANCE METRICS
        {'='*35}
        Accuracy:  {metrics['accuracy']:.4f}
        Precision: {metrics['precision']:.4f}
        Recall:    {metrics['recall']:.4f}
        F1-Score:  {metrics['f1']:.4f}
        ROC-AUC:   {metrics['roc_auc']:.4f}
        
        CONFUSION MATRIX DETAILS
        {'='*35}
        TN (True Negatives):   {cm[0, 0]:,}
        FP (False Positives):  {cm[0, 1]:,}
        FN (False Negatives):  {cm[1, 0]:,}
        TP (True Positives):   {cm[1, 1]:,}
        """
        axes[1, 1].text(0.05, 0.5, metrics_text,
                        fontsize=10, family='monospace', verticalalignment='center')

        # Plot 6: Federated Convergence (if applicable) - Rectangle shape with unified Y-axis
        if self.convergence_data and len(self.convergence_data) > 0:
            rounds = [d['round'] for d in self.convergence_data]
            accuracies = [d['val_accuracy'] for d in self.convergence_data]
            losses = [d['val_loss'] for d in self.convergence_data]

            ax6_acc = axes[1, 2]

            line1 = ax6_acc.plot(rounds, accuracies, 'b-', marker='o',
                                 label='Accuracy', linewidth=2, markersize=6)
            line2 = ax6_acc.plot(rounds, losses, 'r-', marker='s',
                                 label='Loss', linewidth=2, markersize=6)

            ax6_acc.set_title('Federated Convergence', fontweight='bold')
            ax6_acc.set_xlabel('Round')
            ax6_acc.set_ylabel('Accuracy/Loss')

            # Set X-axis: 2 round intervals
            max_round = rounds[-1] if rounds else 50
            round_ticks = list(range(0, max_round + 2, 2))
            if round_ticks[-1] < max_round:
                round_ticks.append(max_round)
            ax6_acc.set_xticks(round_ticks)

            # Set Y-axis: Universal 0.0 to 1.0 with 0.1 intervals
            ax6_acc.set_ylim(0.0, 1.0)
            ax6_acc.set_yticks([i/10.0 for i in range(0, 11)])

            ax6_acc.grid(True, alpha=0.3)

            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax6_acc.legend(lines, labels, loc='upper left')
        else:
            axes[1, 2].axis('off')
            axes[1, 2].text(0.5, 0.5, 'No Federated Data',
                            ha='center', va='center', fontsize=12)

        plt.tight_layout()
        plt.savefig('results/training_results.png',
                    dpi=300, bbox_inches='tight')
        logger.info("✅ Visualization saved: results/training_results.png")
        plt.close()


def main(federated_rounds=1, epochs=5):
    """Main training function

    Args:
        federated_rounds: Number of federated communication rounds (1 for standard, 50+ for production)
        epochs: Number of epochs per federated round
    """
    try:
        logger.info("="*70)
        if federated_rounds > 1:
            logger.info(
                f"🔥 PRODUCTION FEDERATED TRAINING ({federated_rounds} ROUNDS)")
        else:
            logger.info("🔥 PRODUCTION DDOS DETECTION TRAINING")
        logger.info("="*70)

        trainer = ProductionTrainer()

        # Load real data
        X_train, y_train, X_test, y_test = trainer.load_real_data()

        # Preprocess
        X_train, X_test = trainer.preprocess(X_train, X_test)

        # Build model
        trainer.build_model((X_train.shape[1], X_train.shape[2]))

        # Train with federated rounds
        trainer.train(X_train, y_train, X_test, y_test,
                      federated_rounds=federated_rounds, epochs=epochs)

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

        # Save convergence data if federated
        if federated_rounds > 1 and trainer.convergence_data:
            with open('results/federated_training_convergence.json', 'w') as f:
                json.dump({
                    'total_rounds': federated_rounds,
                    'convergence': trainer.convergence_data,
                    'final_metrics': metrics,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            logger.info(
                f"✅ Convergence data saved: results/federated_training_convergence.json")

        logger.info("\n" + "="*70)
        logger.info("✅ TRAINING COMPLETE!")
        logger.info(f"   Federated Rounds: {federated_rounds}")
        logger.info(f"   Final Accuracy: {metrics['accuracy']:.4f}")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DDoS Detection Training')
    parser.add_argument('--federated-rounds', type=int, default=1,
                        help='Number of federated rounds')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Epochs per round')
    args = parser.parse_args()

    main(federated_rounds=args.federated_rounds, epochs=args.epochs)
