#!/usr/bin/env python3
"""
Simple data exploration script
"""

import pandas as pd
import os


def quick_data_check():
    """Quick check of one data file"""

    # Path to one data file
    file_path = "data/raw/CSV-01-12/01-12/DrDoS_DNS.csv"

    print(f"🔍 Quick check of: {file_path}")

    if os.path.exists(file_path):
        print("✅ File exists")

        # Check file size
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"📏 File size: {file_size:.2f} MB")

        # Load just first few rows
        print("📖 Loading first 5 rows...")
        sample = pd.read_csv(file_path, nrows=5)

        print(f"Shape: {sample.shape}")
        print(f"Columns: {list(sample.columns)}")
        print("\nFirst few rows:")
        print(sample.head())

        # Check total rows (this might take a moment)
        print("\n📊 Counting total rows...")
        try:
            total_rows = sum(1 for line in open(file_path)) - \
                1  # Subtract header
            print(f"Total rows: {total_rows:,}")
        except Exception as e:
            print(f"Could not count rows: {e}")

    else:
        print("❌ File not found")


if __name__ == "__main__":
    quick_data_check()
