#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick launcher for Federated DDoS Detection project
Provides easy commands to run different parts of the system
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Add src directory to Python path
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data" / "optimized"

# Add src to Python path for imports
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

# Set environment variables for encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONPATH'] = str(PROJECT_ROOT)


def print_banner():
    """Print project banner"""
    print("🛡️  " + "="*60 + " 🛡️")
    print("🚀      FEDERATED DDOS DETECTION SYSTEM         🚀")
    print("🛡️  " + "="*60 + " 🛡️")
    print()


def check_environment():
    """Check if environment is ready"""
    print("🔍 Checking environment...")

    # Check if data exists
    if not DATA_DIR.exists():
        print("❌ Data directory not found. Please run data preparation first.")
        return False

    # Check if client data exists
    for i in range(4):
        train_file = DATA_DIR / f"client_{i}_train.csv"
        test_file = DATA_DIR / f"client_{i}_test.csv"
        if not train_file.exists() or not test_file.exists():
            print(f"❌ Client {i} data files not found.")
            return False

    # Test imports
    try:
        print("🔍 Testing imports...")
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} imported successfully")

        import flwr as fl
        print(f"✅ Flower {fl.__version__} imported successfully")

        # Test project imports
        from src.models.cnn_model import DDoSCNNModel
        print("✅ Project modules imported successfully")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Please install required packages: pip install -r requirements.txt")
        return False

    print("✅ Environment check passed!")
    return True


def run_command(cmd, description, cwd=None):
    """Run a command and handle errors"""
    print(f"\n🚀 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)

    # Set up environment for subprocess
    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT)
    env['PYTHONIOENCODING'] = 'utf-8'

    try:
        result = subprocess.run(
            cmd, cwd=cwd or PROJECT_ROOT, env=env, check=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError as e:
        print(f"❌ {description} failed: {e}")
        print("💡 Make sure Python and required packages are installed")
        return False
    except KeyboardInterrupt:
        print(f"⚠️ {description} interrupted by user")
        return False


def test_model():
    """Test the CNN model standalone"""
    cmd = [sys.executable, "src/models/trainer.py", "--test"]
    return run_command(cmd, "Testing CNN Model")


def test_client():
    """Test federated client"""
    cmd = [sys.executable, "src/federated/flower_client.py", "--test"]
    return run_command(cmd, "Testing Federated Client")


def test_server():
    """Test federated server"""
    cmd = [sys.executable, "src/federated/flower_server.py", "--test"]
    return run_command(cmd, "Testing Federated Server")


def run_centralized():
    """Run centralized baseline"""
    print("\n📊 Running Centralized Baseline Training")
    print("This will train a model using all data combined...")

    # Create a simple centralized training script
    script_content = '''
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.models.trainer import ModelTrainer
import numpy as np

def main():
    print("🏋️ Starting centralized training...")
    
    trainer = ModelTrainer()
    trainer.create_model(learning_rate=0.001)
    
    # Load all client data
    all_X_train, all_y_train = [], []
    all_X_test, all_y_test = [], []
    
    for client_id in range(4):
        print(f"Loading client {client_id} data...")
        train_path = f"data/optimized/client_{client_id}_train.csv"
        test_path = f"data/optimized/client_{client_id}_test.csv"
        
        X_train, y_train = trainer.load_data(train_path)
        X_test, y_test = trainer.load_data(test_path)
        
        all_X_train.append(X_train)
        all_y_train.append(y_train)
        all_X_test.append(X_test)
        all_y_test.append(y_test)
    
    # Combine all data
    X_train_combined = np.vstack(all_X_train)
    y_train_combined = np.hstack(all_y_train)
    X_test_combined = np.vstack(all_X_test)
    y_test_combined = np.hstack(all_y_test)
    
    print(f"Training with {len(X_train_combined)} samples...")
    
    # Train model
    results = trainer.train_model(
        X_train_combined, y_train_combined,
        X_test_combined, y_test_combined,
        epochs=10,
        batch_size=64,
        validation_split=0.2
    )
    
    print("\\n🎉 Centralized training completed!")
    print(f"Final accuracy: {results['evaluation']['accuracy']:.4f}")

if __name__ == "__main__":
    main()
'''

    # Write and run script
    temp_script = PROJECT_ROOT / "temp_centralized.py"
    with open(temp_script, 'w') as f:
        f.write(script_content)

    try:
        cmd = [sys.executable, str(temp_script)]
        success = run_command(cmd, "Centralized Training")
        return success
    finally:
        # Cleanup
        if temp_script.exists():
            temp_script.unlink()


def run_federated_quick():
    """Run a quick federated learning demo"""
    print("\n🌐 Running Quick Federated Learning Demo")
    print("This will run 3 rounds with 2 clients...")

    # Start server in background
    print("\n1. Starting server...")
    server_cmd = [sys.executable, "src/federated/flower_server.py",
                  "--rounds", "3", "--clients", "2"]

    # Set up environment
    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT)
    env['PYTHONIOENCODING'] = 'utf-8'

    server_process = subprocess.Popen(server_cmd, cwd=PROJECT_ROOT, env=env)

    # Give server time to start
    time.sleep(5)

    try:
        # Start clients
        print("2. Starting clients...")
        client_processes = []

        for client_id in [0, 1]:  # Only use 2 clients for quick demo
            print(f"   Starting client {client_id}...")
            client_cmd = [
                sys.executable, "src/federated/flower_client.py",
                "--client_id", str(client_id)
            ]
            process = subprocess.Popen(client_cmd, cwd=PROJECT_ROOT, env=env)
            client_processes.append(process)
            time.sleep(2)  # Small delay between clients

        print("3. Waiting for completion...")

        # Wait for server to complete
        server_process.wait(timeout=300)  # 5 minutes timeout

        # Wait for clients
        for process in client_processes:
            process.wait(timeout=60)

        print("✅ Quick federated demo completed!")
        return True

    except subprocess.TimeoutExpired:
        print("⏰ Demo timed out!")
        return False
    except KeyboardInterrupt:
        print("⚠️ Demo interrupted!")
        return False
    finally:
        # Cleanup processes
        try:
            server_process.terminate()
        except:
            pass
        for process in client_processes:
            try:
                process.terminate()
            except:
                pass


def run_full_demo():
    """Run the complete demo"""
    cmd = [sys.executable, "demo.py"]
    return run_command(cmd, "Full Demo (Centralized + Federated)")


def show_menu():
    """Show the main menu"""
    print("\n📋 Available Commands:")
    print("1. 🧪 test-model     - Test CNN model standalone")
    print("2. 🧪 test-client    - Test federated client")
    print("3. 🧪 test-server    - Test federated server")
    print("4. 📊 centralized    - Run centralized baseline")
    print("5. 🌐 federated      - Run quick federated demo (3 rounds, 2 clients)")
    print("6. 🚀 full-demo      - Run complete demo (centralized + federated)")
    print("7. 📖 help           - Show this menu")
    print("8. 🚪 exit           - Exit launcher")
    print()


def main():
    """Main launcher function"""
    print_banner()

    if not check_environment():
        print("\n❌ Environment check failed. Please set up the project first.")
        return

    print("🎯 Federated DDoS Detection System Ready!")

    show_menu()

    while True:
        try:
            choice = input("Enter command (or number): ").strip().lower()

            if choice in ['1', 'test-model']:
                test_model()
            elif choice in ['2', 'test-client']:
                test_client()
            elif choice in ['3', 'test-server']:
                test_server()
            elif choice in ['4', 'centralized']:
                run_centralized()
            elif choice in ['5', 'federated']:
                run_federated_quick()
            elif choice in ['6', 'full-demo']:
                run_full_demo()
            elif choice in ['7', 'help']:
                show_menu()
            elif choice in ['8', 'exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Type 'help' to see available commands.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
