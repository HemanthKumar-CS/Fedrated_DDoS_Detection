#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Test Script for Federated DDoS Detection Setup
Tests basic functionality before running full system
"""

import os
import sys
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Set encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'


def test_imports():
    """Test all required imports"""
    print("🧪 Testing Imports...")

    try:
        import numpy as np
        print("✅ NumPy imported")

        import pandas as pd
        print("✅ Pandas imported")

        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} imported")

        import flwr as fl
        print(f"✅ Flower {fl.__version__} imported")

        print("🧪 Testing project modules...")
        from src.models.cnn_model import DDoSCNNModel
        print("✅ CNN Model imported")

        from src.models.trainer import ModelTrainer
        print("✅ Model Trainer imported")

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_data():
    """Test data availability"""
    print("\n🧪 Testing Data...")

    data_dir = PROJECT_ROOT / "data" / "optimized"

    if not data_dir.exists():
        print("❌ Data directory not found")
        return False

    # Check binary dataset
    binary_file = data_dir / "binary_dataset.csv"
    if binary_file.exists():
        print("✅ Binary dataset found")
    else:
        print("❌ Binary dataset not found")
        return False

    # Check client files
    missing_files = []
    for i in range(4):
        train_file = data_dir / f"client_{i}_train.csv"
        test_file = data_dir / f"client_{i}_test.csv"

        if not train_file.exists():
            missing_files.append(str(train_file))
        if not test_file.exists():
            missing_files.append(str(test_file))

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False

    print("✅ All client data files found")
    return True


def test_model():
    """Test model creation"""
    print("\n🧪 Testing Model Creation...")

    try:
        from src.models.cnn_model import DDoSCNNModel

        # Create binary classification model
        model = DDoSCNNModel(input_features=31, num_classes=1)
        model.compile_model()

        print("✅ Binary classification model created successfully")
        print(f"✅ Input shape: (None, 31, 1)")
        print(f"✅ Output shape: (None, 1)")

        return True

    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False


def test_data_loading():
    """Test data loading"""
    print("\n🧪 Testing Data Loading...")

    try:
        from src.models.trainer import ModelTrainer

        trainer = ModelTrainer(input_features=31, num_classes=1)

        # Test loading client 0 data
        train_path = str(PROJECT_ROOT / "data" /
                         "optimized" / "client_0_train.csv")
        X, y = trainer.load_data(train_path)

        print(f"✅ Data loaded: X shape {X.shape}, y shape {y.shape}")
        print(f"✅ Binary labels: min={y.min()}, max={y.max()}")

        return True

    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🔬 FEDERATED DDOS DETECTION - SETUP TEST")
    print("=" * 50)

    tests = [
        ("Imports", test_imports),
        ("Data Files", test_data),
        ("Model Creation", test_model),
        ("Data Loading", test_data_loading)
    ]

    passed = 0
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test error: {e}")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{len(tests)} passed")

    if passed == len(tests):
        print("🎉 All tests passed! Your setup is ready!")
        print("🚀 Next step: Run 'python launcher.py' to start the system")
        return True
    else:
        print("⚠️ Some tests failed. Please fix the issues before proceeding.")
        print("💡 Common fixes:")
        print("   - Install requirements: pip install -r requirements.txt")
        print("   - Create binary dataset: python create_binary_dataset.py")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
