#!/usr/bin/env python3
"""
Simple test for centralized training
"""

import pandas as pd
import numpy as np
from src.models.trainer import ModelTrainer
import sys
import os
sys.path.append('src')


def test_simple_training():
    print("🧪 Simple Training Test")
    print("=" * 40)

    try:
        print("Step 1: Importing trainer...")
        from src.models.trainer import ModelTrainer
        print("✅ Import successful")

        print("Step 2: Creating trainer...")
        trainer = ModelTrainer()
        print("✅ Trainer created")

        print("Step 3: Creating model...")
        trainer.create_model(learning_rate=0.001)
        print("✅ Model created")

        print("Step 4: Loading data...")
        train_path = "data/optimized/client_0_train.csv"
        print(f"Loading from: {train_path}")
        X_train, y_train = trainer.load_data(train_path)
        print(f"✅ Data loaded: {X_train.shape}, {y_train.shape}")

        print("Step 5: Splitting data...")
        from sklearn.model_selection import train_test_split
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        print(
            f"✅ Data split: Train {X_train_split.shape}, Val {X_val_split.shape}")

        print("Step 6: Starting training...")
        print("This is where it might hang - training for 1 epoch...")
        results = trainer.train_model(
            X_train_split, y_train_split,
            X_val_split, y_val_split,
            epochs=1,
            batch_size=32,
            verbose=1
        )

        print("✅ Training completed!")
        print(f"History keys: {list(results.keys())}")

        return True

    except Exception as e:
        print(f"❌ Error at some step: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_simple_training()
    if success:
        print("\n🎉 Simple test passed!")
    else:
        print("\n💥 Simple test failed!")
