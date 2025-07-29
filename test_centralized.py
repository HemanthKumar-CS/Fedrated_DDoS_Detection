#!/usr/bin/env python3
"""
Test centralized training only (fixed version)
"""

import numpy as np
from src.models.trainer import ModelTrainer
import sys
import os
sys.path.append('src')


def test_centralized_only():
    print("🧪 Testing Centralized Training (Fixed)")
    print("=" * 50)

    try:
        # Create trainer
        trainer = ModelTrainer()
        trainer.create_model(learning_rate=0.001)
        print("✅ Model created")

        # Load just one client's data for quick test
        X_train, y_train = trainer.load_data(
            "data/optimized/client_0_train.csv")
        X_test, y_test = trainer.load_data("data/optimized/client_0_test.csv")
        print(f"✅ Data loaded: Train {X_train.shape}, Test {X_test.shape}")

        # Split training data for validation
        from sklearn.model_selection import train_test_split
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        print("🏋️ Training for 2 epochs...")
        results = trainer.train_model(
            X_train_split, y_train_split,
            X_val_split, y_val_split,
            epochs=2,
            batch_size=64,
            verbose=1
        )
        print("✅ Training completed")

        print("📊 Evaluating on test set...")
        test_results = trainer.evaluate_model(X_test, y_test)
        print(f"✅ Evaluation completed")
        print(f"   Test Accuracy: {test_results['test_accuracy']:.4f}")
        print(f"   Test Loss: {test_results['test_loss']:.4f}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_centralized_only()
    if success:
        print("\n🎉 Centralized training test passed!")
        print("🚀 Ready to run full demo again!")
    else:
        print("\n💥 Test failed!")
