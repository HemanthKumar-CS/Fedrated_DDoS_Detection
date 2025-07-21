#!/usr/bin/env python3
"""
Environment Verification Script for Federated Learning Project
This script tests if all required packages are properly installed.
"""


def test_imports():
    """Test importing all required packages"""
    print("🔍 Testing package imports...")

    try:
        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False

    try:
        import pandas as pd
        print(f"✅ Pandas version: {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False

    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        print(f"   GPU Available: {tf.config.list_physical_devices('GPU')}")
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False

    try:
        import sklearn
        print(f"✅ Scikit-learn version: {sklearn.__version__}")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False

    try:
        import flwr as fl
        print(f"✅ Flower version: {fl.__version__}")
    except ImportError as e:
        print(f"❌ Flower import failed: {e}")
        return False

    try:
        import matplotlib
        print(f"✅ Matplotlib version: {matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False

    return True


def test_basic_functionality():
    """Test basic functionality of key packages"""
    print("\n🧪 Testing basic functionality...")

    try:
        # Test numpy
        import numpy as np
        arr = np.array([1, 2, 3, 4, 5])
        print(f"✅ NumPy array operations work: {arr.mean()}")

        # Test pandas
        import pandas as pd
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        print(f"✅ Pandas DataFrame operations work: {df.shape}")

        # Test TensorFlow
        import tensorflow as tf
        # Create a simple tensor
        x = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
        print(f"✅ TensorFlow tensor operations work: {x.shape}")

        # Test a simple neural network layer
        layer = tf.keras.layers.Dense(2)
        output = layer(x)
        print(f"✅ Keras layer operations work: {output.shape}")

        return True

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def main():
    """Main function to run all tests"""
    print("🚀 Federated Learning Environment Verification")
    print("=" * 50)

    # Test imports
    import_success = test_imports()

    if not import_success:
        print("\n❌ Environment setup incomplete. Please check package installations.")
        return False

    # Test functionality
    func_success = test_basic_functionality()

    if import_success and func_success:
        print("\n🎉 Environment setup successful!")
        print("✅ All packages installed and working correctly.")
        print("🔥 Ready to start federated learning development!")
        return True
    else:
        print("\n❌ Some tests failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
