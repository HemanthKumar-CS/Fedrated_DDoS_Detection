#!/usr/bin/env python3
"""
Data Cleanup Script - Remove Unnecessary Files
Keeps only optimized dataset and essential files
"""

import os
import shutil
import sys


def get_directory_size(path):
    """Calculate directory size in MB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)  # Convert to MB


def cleanup_data():
    """Remove unnecessary data files"""

    print("🧹 DATA CLEANUP ANALYSIS")
    print("=" * 50)

    # Check current sizes
    raw_size = get_directory_size(
        "data/raw") if os.path.exists("data/raw") else 0
    processed_size = get_directory_size(
        "data/processed") if os.path.exists("data/processed") else 0
    federated_size = get_directory_size(
        "data/federated") if os.path.exists("data/federated") else 0
    optimized_size = get_directory_size(
        "data/optimized") if os.path.exists("data/optimized") else 0

    print(f"📁 data/raw/: {raw_size:.1f} MB (original dataset)")
    print(
        f"📁 data/processed/: {processed_size:.1f} MB (large processed dataset)")
    print(
        f"📁 data/federated/: {federated_size:.1f} MB (large federated splits)")
    print(f"📁 data/optimized/: {optimized_size:.1f} MB (optimized dataset)")

    total_current = raw_size + processed_size + federated_size + optimized_size
    space_to_save = raw_size + processed_size + federated_size

    print(f"\n💾 Current total: {total_current:.1f} MB")
    print(f"💾 After cleanup: {optimized_size:.1f} MB")
    print(
        f"🎯 Space saved: {space_to_save:.1f} MB ({space_to_save/1024:.1f} GB)")

    # Ask user confirmation
    response = input(
        "\n🤔 Do you want to remove unnecessary data? (y/n): ").strip().lower()

    if response == 'y' or response == 'yes':
        print("\n🗑️ Removing unnecessary files...")

        # Remove raw data (but keep structure for reference)
        if os.path.exists("data/raw/CSV-01-12"):
            print("Removing raw CSV files...")
            shutil.rmtree("data/raw/CSV-01-12")

        # Keep the zip file but remove extracted folder
        if os.path.exists("data/raw/CSV-01-12.zip"):
            print("Keeping CSV-01-12.zip for reference")

        # Remove large processed data
        if os.path.exists("data/processed"):
            print("Removing large processed data...")
            # Keep only the summary report
            if os.path.exists("data/processed/PHASE2_COMPLETION_REPORT.txt"):
                shutil.copy(
                    "data/processed/PHASE2_COMPLETION_REPORT.txt", "data/optimized/")
            shutil.rmtree("data/processed")

        # Remove large federated data
        if os.path.exists("data/federated"):
            print("Removing large federated data...")
            shutil.rmtree("data/federated")

        print("\n✅ Cleanup completed!")

        # Check new size
        new_total = get_directory_size("data")
        print(f"📁 Final data size: {new_total:.1f} MB")
        print(f"🎉 Freed up: {space_to_save:.1f} MB")

        # Create cleanup report
        with open("CLEANUP_REPORT.txt", 'w') as f:
            f.write("DATA CLEANUP REPORT\n")
            f.write("==================\n\n")
            f.write(f"Original size: {total_current:.1f} MB\n")
            f.write(f"Final size: {new_total:.1f} MB\n")
            f.write(f"Space saved: {space_to_save:.1f} MB\n\n")
            f.write("KEPT FILES:\n")
            f.write("- data/optimized/ - Optimized dataset for FL training\n")
            f.write("- data/raw/CSV-01-12.zip - Original dataset archive\n")
            f.write("- All source code and notebooks\n\n")
            f.write("REMOVED FILES:\n")
            f.write("- data/raw/CSV-01-12/ - Extracted large CSV files\n")
            f.write("- data/processed/ - Large processed dataset\n")
            f.write("- data/federated/ - Large federated splits\n\n")
            f.write("RECOMMENDATION:\n")
            f.write("Use data/optimized/ for your federated learning training.\n")
            f.write(
                "It contains 50k samples with 4 attack types - perfect for demonstration.\n")

        print("\n📋 Created CLEANUP_REPORT.txt")

    else:
        print("\n⏸️ Cleanup cancelled. No files removed.")
        print("💡 You can run this script again anytime to cleanup.")


def show_project_status():
    """Show current project status"""
    print("\n📊 PROJECT STATUS AFTER OPTIMIZATION")
    print("=" * 40)

    # Check what we have
    optimized_exists = os.path.exists("data/optimized/optimized_dataset.csv")
    clients_exist = all(os.path.exists(
        f"data/optimized/client_{i}_train.csv") for i in range(4))

    if optimized_exists and clients_exist:
        print("✅ Optimized dataset ready")
        print("✅ 4 federated clients configured")
        print("✅ ~50k samples, 4 attack types")
        print("✅ ~8MB total size")
        print("\n🚀 READY FOR PHASE 3: CNN Development!")
    else:
        print("❌ Optimization incomplete")
        print("🔄 Run optimize_dataset.py again")


if __name__ == "__main__":
    cleanup_data()
    show_project_status()
