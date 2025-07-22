#!/usr/bin/env python3
"""
Quick test script to verify data loading
"""

from data.data_loader import CICDDoS2019Loader
import sys
import os
sys.path.append('src')


def test_data_loading():
    """Test data loading with correct path"""

    # Correct path to the data
    data_path = "data/raw/CSV-01-12/01-12"

    print(f"🔍 Testing data loading from: {data_path}")
    print(f"📁 Current directory: {os.getcwd()}")

    # Check if directory exists
    if os.path.exists(data_path):
        print(f"✅ Data directory exists: {data_path}")

        # List files in directory
        files = os.listdir(data_path)
        print(f"📄 Files found: {files}")

        # Initialize loader with correct path
        loader = CICDDoS2019Loader(data_path=data_path)

        # Get info
        info = loader.get_dataset_info()
        print(f"\n📊 Dataset info: {info}")

        if info['total_files'] > 0:
            # Load small sample
            sample_df = loader.load_all_data(sample_size=100)
            print(f"✅ Sample loaded: {sample_df.shape}")
            if not sample_df.empty:
                print(f"Columns: {list(sample_df.columns)}")
                print(f"Labels: {sample_df['Label'].unique()}")
        else:
            print("❌ No CSV files found")
    else:
        print(f"❌ Data directory does not exist: {data_path}")
        print("📂 Available directories:")
        if os.path.exists("data"):
            for item in os.listdir("data"):
                print(f"  - {item}")


if __name__ == "__main__":
    test_data_loading()
