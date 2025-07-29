#!/usr/bin/env python3
"""
Model Training Module for Federated DDoS Detection
Handles training, validation, and evaluation of the CNN model
"""

from src.models.cnn_model import create_ddos_cnn_model, CLASS_NAMES, LABEL_TO_ID
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, List
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Handles training and evaluation of the DDoS CNN model
    """

    def __init__(self, input_features: int = 31, num_classes: int = 1):
        """
        Initialize the trainer

        Args:
            input_features: Number of input features (31 for binary dataset)
            num_classes: Number of output classes (1 for binary classification)
        """
        self.input_features = input_features
        self.num_classes = num_classes
        self.model = None
        self.label_encoder = LabelEncoder()
        self.history = None

    def load_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess data from CSV file

        Args:
            data_path: Path to the data CSV file

        Returns:
            Tuple of (features, labels)
        """
        logger.info(f"Loading data from {data_path}")

        # Load data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded dataset shape: {df.shape}")

        # Separate features and labels
        if 'Binary_Label' in df.columns:
            # Use Binary_Label for binary classification
            X = df.drop(['Label', 'Binary_Label'], axis=1)
            y = df['Binary_Label']
        elif 'Label' in df.columns:
            # Fallback to Label column for backward compatibility
            X = df.drop(['Label'], axis=1)
            y = df['Label']
        else:
            raise ValueError(
                "No 'Binary_Label' or 'Label' column found in dataset")

        # Handle any remaining non-numeric columns
        X = X.select_dtypes(include=[np.number])

        # For binary classification, labels are already 0/1, no encoding needed
        if 'Binary_Label' in df.columns:
            y_encoded = y.values
            logger.info("Using binary labels (0=BENIGN, 1=ATTACK)")
        else:
            # Encode labels to integers for multi-class (backward compatibility)
            y_encoded = self.label_encoder.fit_transform(y)

        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Labels shape: {y_encoded.shape}")
        logger.info(
            f"Label distribution: {dict(zip(*np.unique(y_encoded, return_counts=True)))}")

        return X.values, y_encoded

    def prepare_training_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and validation sets

        Args:
            X: Features array
            y: Labels array
            test_size: Proportion of data for validation
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        logger.info("Preparing training and validation data...")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Normalize features (StandardScaler equivalent)
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)

        # Add small epsilon to avoid division by zero
        X_train_norm = (X_train - mean) / (std + 1e-8)
        X_val_norm = (X_val - mean) / (std + 1e-8)

        logger.info(f"Training data shape: {X_train_norm.shape}")
        logger.info(f"Validation data shape: {X_val_norm.shape}")

        return X_train_norm, X_val_norm, y_train, y_val

    def create_model(self, learning_rate: float = 0.001):
        """
        Create and compile the CNN model

        Args:
            learning_rate: Learning rate for optimizer
        """
        logger.info("Creating CNN model...")
        self.model = create_ddos_cnn_model(
            input_features=self.input_features,
            num_classes=self.num_classes,
            learning_rate=learning_rate
        )

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    epochs: int = 50, batch_size: int = 32, verbose: int = 1) -> Dict[str, Any]:
        """
        Train the CNN model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level

        Returns:
            Training history dictionary
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")

        logger.info(
            f"Starting training for {epochs} epochs with batch_size={batch_size}")

        # Prepare data for CNN
        X_train_cnn = self.model.prepare_data(X_train)
        X_val_cnn = self.model.prepare_data(X_val)

        # Callbacks for better training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Train model
        start_time = datetime.now()
        self.history = self.model.get_model().fit(
            X_train_cnn, y_train,
            validation_data=(X_val_cnn, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        training_time = datetime.now() - start_time
        logger.info(f"✅ Training completed in {training_time}")

        return self.history.history

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the trained model

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")

        logger.info("Evaluating model performance...")

        # Prepare test data
        X_test_cnn = self.model.prepare_data(X_test)

        # Get predictions
        y_pred_proba = self.model.get_model().predict(X_test_cnn, verbose=0)
        # For binary classification, use threshold 0.5
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # Calculate metrics
        evaluation_results = self.model.get_model().evaluate(X_test_cnn, y_test, verbose=0)
        # evaluation_results is a list: [loss, accuracy, precision, recall, ...]
        test_loss = evaluation_results[0]
        test_accuracy = evaluation_results[1]

        # Classification report
        class_report = classification_report(
            y_test, y_pred,
            target_names=[CLASS_NAMES[i] for i in range(len(CLASS_NAMES))],
            output_dict=True
        )

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)

        results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),  # Convert numpy array to list
            'predictions': y_pred.tolist(),  # Convert numpy array to list
            'prediction_probabilities': y_pred_proba.tolist()  # Convert numpy array to list
        }

        logger.info(f"✅ Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"✅ Test Loss: {test_loss:.4f}")

        return results

    def plot_training_history(self, save_path: str = None):
        """
        Plot training history

        Args:
            save_path: Path to save the plot (optional)
        """
        if self.history is None:
            logger.warning("No training history available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training')
        axes[0, 0].plot(self.history.history['val_accuracy'],
                        label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Plot loss
        axes[0, 1].plot(self.history.history['loss'], label='Training')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Plot precision if available
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'],
                            label='Training')
            axes[1, 0].plot(
                self.history.history['val_precision'], label='Validation')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        # Plot recall if available
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Training')
            axes[1, 1].plot(self.history.history['val_recall'],
                            label='Validation')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")

        plt.show()

    def plot_confusion_matrix(self, conf_matrix: np.ndarray, save_path: str = None):
        """
        Plot confusion matrix

        Args:
            conf_matrix: Confusion matrix array
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=[CLASS_NAMES[i] for i in range(self.num_classes)],
            yticklabels=[CLASS_NAMES[i] for i in range(self.num_classes)]
        )

        plt.title('Confusion Matrix - DDoS Detection')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved to {save_path}")

        plt.show()

    def save_model(self, model_path: str):
        """
        Save the trained model

        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not created")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        self.model.save_model(model_path)
        logger.info(f"✅ Model saved to {model_path}")


def main():
    """
    Example training pipeline
    """
    print("🚀 DDoS CNN Model Training Pipeline")
    print("=" * 50)

    # Configuration
    DATA_PATH = "../../data/optimized/optimized_dataset.csv"
    MODEL_SAVE_PATH = "../../models/ddos_cnn_model.h5"

    try:
        # Initialize trainer
        trainer = ModelTrainer()

        # Load data
        X, y = trainer.load_data(DATA_PATH)

        # Prepare training data
        X_train, X_val, y_train, y_val = trainer.prepare_training_data(X, y)

        # Create model
        trainer.create_model(learning_rate=0.001)

        # Print model summary
        print("\n📊 Model Architecture:")
        print(trainer.model.get_model_summary())

        # Train model
        print("\n🔧 Starting training...")
        history = trainer.train_model(
            X_train, y_train, X_val, y_val, epochs=30)

        # Evaluate model
        print("\n📈 Evaluating model...")
        results = trainer.evaluate_model(X_val, y_val)

        # Print results
        print(f"\n✅ Final Results:")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"Test Loss: {results['test_loss']:.4f}")

        # Save model
        trainer.save_model(MODEL_SAVE_PATH)

        print("\n🎉 Training completed successfully!")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
