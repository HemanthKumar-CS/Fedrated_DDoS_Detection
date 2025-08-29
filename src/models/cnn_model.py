#!/usr/bin/env python3
"""
CNN Model for Federated DDoS Detection
Implements a 1D CNN architecture optimized for network traffic classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DDoSCNNModel:
    """
    1D CNN model for DDoS attack classification
    Designed for federated learning with optimized performance
    """

    def __init__(self, input_features: int = 29, num_classes: int = 1):
        """
        Initialize the CNN model

        Args:
            input_features: Number of input features (default: 29 from our optimized dataset)
            num_classes: Number of output classes (default: 1 for binary classification - benign vs attack)
        """
        self.input_features = input_features
        self.num_classes = num_classes
        self.model = None
        self._build_model()

    def _build_model(self):
        """
        Build the 1D CNN architecture

        Architecture Design:
        - Input Layer: (None, 29, 1) for 29 features
        - Conv1D Layers: Extract patterns from network traffic
        - MaxPooling: Reduce dimensionality
        - Dropout: Prevent overfitting
        - Dense Layers: Classification head
        - Output: Binary classification with sigmoid
        """
        logger.info("Building 1D CNN model for DDoS detection...")

        # Input layer - reshape features for 1D convolution
        inputs = keras.Input(shape=(self.input_features, 1),
                             name="traffic_features")

        # First Convolutional Block
        x = layers.Conv1D(filters=32, kernel_size=3,
                          activation='relu', name="conv1d_1")(inputs)
        x = layers.BatchNormalization(name="batch_norm_1")(x)
        x = layers.MaxPooling1D(pool_size=2, name="maxpool_1")(x)
        x = layers.Dropout(0.25, name="dropout_1")(x)

        # Second Convolutional Block
        x = layers.Conv1D(filters=64, kernel_size=3,
                          activation='relu', name="conv1d_2")(x)
        x = layers.BatchNormalization(name="batch_norm_2")(x)
        x = layers.MaxPooling1D(pool_size=2, name="maxpool_2")(x)
        x = layers.Dropout(0.25, name="dropout_2")(x)

        # Third Convolutional Block
        x = layers.Conv1D(filters=128, kernel_size=3,
                          activation='relu', name="conv1d_3")(x)
        x = layers.BatchNormalization(name="batch_norm_3")(x)
        x = layers.GlobalMaxPooling1D(name="global_maxpool")(
            x)  # Global pooling instead of flatten

        # Dense Classification Head
        x = layers.Dense(256, activation='relu', name="dense_1")(x)
        x = layers.BatchNormalization(name="batch_norm_4")(x)
        x = layers.Dropout(0.5, name="dropout_3")(x)

        x = layers.Dense(128, activation='relu', name="dense_2")(x)
        x = layers.Dropout(0.3, name="dropout_4")(x)

        # Output layer
        if self.num_classes == 1:
            # Binary classification
            outputs = layers.Dense(1, activation='sigmoid', name="classification_output")(x)
        else:
            # Multi-class classification (softmax)
            outputs = layers.Dense(self.num_classes, activation='softmax', name="classification_output")(x)

        # Create model
        self.model = keras.Model(
            inputs=inputs, outputs=outputs, name="DDoS_CNN")

        logger.info(
            f"✅ Model built successfully with {self.input_features} input features and {self.num_classes} output classes")

    def compile_model(self, learning_rate: float = 0.001, metrics: list = None):
        """
        Compile the model with optimizer, loss, and metrics

        Args:
            learning_rate: Learning rate for Adam optimizer
            metrics: List of metrics to track during training
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall']

        logger.info(f"Compiling model with learning_rate={learning_rate}")

        if self.num_classes == 1:
            loss_fn = 'binary_crossentropy'
        else:
            loss_fn = 'sparse_categorical_crossentropy'

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss_fn,
            metrics=metrics
        )

        logger.info("✅ Model compiled successfully")

    def get_model_summary(self) -> str:
        """
        Get model architecture summary

        Returns:
            String representation of model architecture
        """
        if self.model is None:
            return "Model not built yet"

        import io
        import sys

        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        self.model.summary()

        sys.stdout = old_stdout
        summary = buffer.getvalue()

        return summary

    def save_model(self, filepath: str):
        """
        Save the trained model

        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not built yet")

        self.model.save(filepath)
        logger.info(f"✅ Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load a pre-trained model

        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        logger.info(f"✅ Model loaded from {filepath}")

    def get_model(self) -> keras.Model:
        """
        Get the Keras model instance

        Returns:
            Compiled Keras model
        """
        if self.model is None:
            raise ValueError("Model not built yet")
        return self.model

    def prepare_data(self, X: np.ndarray) -> np.ndarray:
        """
        Prepare input data for the model

        Args:
            X: Input features array (samples, features)

        Returns:
            Reshaped data for 1D CNN (samples, features, 1)
        """
        if len(X.shape) == 2:
            # Reshape for 1D CNN: (samples, features) -> (samples, features, 1)
            X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
            logger.info(f"Data reshaped from {X.shape} to {X_reshaped.shape}")
            return X_reshaped
        return X


def create_ddos_cnn_model(input_features: int = 29, num_classes: int = 1, learning_rate: float = 0.001) -> DDoSCNNModel:
    """
    Factory function to create and compile a DDoS CNN model

    Args:
        input_features: Number of input features
        num_classes: Number of output classes
        learning_rate: Learning rate for optimization

    Returns:
        Compiled DDoSCNNModel instance
    """
    logger.info("Creating DDoS CNN model...")

    # Create model
    cnn_model = DDoSCNNModel(
        input_features=input_features, num_classes=num_classes)

    # Compile model
    cnn_model.compile_model(learning_rate=learning_rate)

    logger.info("✅ DDoS CNN model created and compiled successfully")

    return cnn_model


# Class mapping for our dataset
CLASS_NAMES = {
    0: "BENIGN",
    1: "ATTACK"
}

# Reverse mapping
LABEL_TO_ID = {v: k for k, v in CLASS_NAMES.items()}


def main():
    """
    Test function to demonstrate model creation
    """
    print("🔧 Testing DDoS CNN Model Creation")
    print("=" * 50)

    try:
        # Create model
        model = create_ddos_cnn_model()

        # Print model summary
        print("\n📊 Model Architecture:")
        print(model.get_model_summary())

        # Test data preparation
        print("\n🧪 Testing data preparation:")
        dummy_data = np.random.random((100, 29))  # 100 samples, 29 features
        prepared_data = model.prepare_data(dummy_data)
        print(f"Original shape: {dummy_data.shape}")
        print(f"Prepared shape: {prepared_data.shape}")

        # Test prediction (without training)
        print("\n🔮 Testing prediction capability:")
        predictions = model.get_model().predict(
            prepared_data[:5])  # Test with 5 samples
        print(f"Prediction shape: {predictions.shape}")
        print(f"Sample prediction: {predictions[0]}")

        print("\n✅ All tests passed! Model is ready for training.")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        raise


if __name__ == "__main__":
    main()
