#!/usr/bin/env python3
"""
Main Data Processing Script for CICIDDOS2019 Dataset
This script orchestrates the complete data preparation pipeline for federated learning
"""

from data.federated_split import FederatedDataDistributor
from data.preprocessing import DataPreprocessor
from data.data_loader import CICDDoS2019Loader
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """
    Main data processing pipeline
    """
    logger.info("🚀 Starting CICIDDOS2019 data processing pipeline")
    start_time = datetime.now()

    # Configuration
    DATA_PATH = "../data/raw/CSV-01-12/01-12"
    PROCESSED_PATH = "../data/processed"
    FEDERATED_PATH = "../data/federated"
    SAMPLE_SIZE = 5000  # Adjust based on your system's memory
    NUM_CLIENTS = 5

    try:
        # Step 1: Load data
        logger.info("📥 Step 1: Loading CICIDDOS2019 dataset...")
        loader = CICDDoS2019Loader(data_path=DATA_PATH)

        # Get dataset info first
        info = loader.get_dataset_info()
        logger.info(
            f"Found {info['total_files']} files with {sum(info['file_sizes'].values()):,} total records")

        # Load sample data
        df = loader.load_all_data(sample_size=SAMPLE_SIZE)
        if df.empty:
            raise ValueError("No data loaded!")

        logger.info(f"Loaded dataset shape: {df.shape}")

        # Step 2: Preprocess data
        logger.info("🔧 Step 2: Preprocessing data...")
        preprocessor = DataPreprocessor()
        processed_df = preprocessor.preprocess_pipeline(df, fit=True)

        # Save preprocessor
        os.makedirs(PROCESSED_PATH, exist_ok=True)
        preprocessor.save_preprocessor(PROCESSED_PATH)

        # Save processed data
        processed_file = os.path.join(PROCESSED_PATH, "processed_data.csv")
        processed_df.to_csv(processed_file, index=False)
        logger.info(f"Processed data saved to {processed_file}")

        # Step 3: Create federated data splits
        logger.info("🏢 Step 3: Creating federated data distribution...")
        distributor = FederatedDataDistributor(num_clients=NUM_CLIENTS)

        # Create non-IID distribution
        client_data = distributor.create_non_iid_by_label(
            processed_df, concentration=0.5)

        # Split into train/test
        train_data, test_data = distributor.split_train_test(
            client_data, test_size=0.2)

        # Save federated data
        os.makedirs(FEDERATED_PATH, exist_ok=True)
        distributor.save_federated_data(train_data, FEDERATED_PATH, "train")
        distributor.save_federated_data(test_data, FEDERATED_PATH, "test")

        # Step 4: Generate summary report
        logger.info("📊 Step 4: Generating summary report...")
        generate_summary_report(
            df, processed_df, train_data, test_data, PROCESSED_PATH)

        # Success!
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"✅ Data processing completed successfully in {duration}")

        print("\n" + "="*60)
        print("🎉 DATA PROCESSING COMPLETE!")
        print("="*60)
        print(
            f"✅ Original data: {df.shape[0]:,} samples, {df.shape[1]} features")
        print(
            f"✅ Processed data: {processed_df.shape[0]:,} samples, {processed_df.shape[1]} features")
        print(f"✅ Federated clients: {NUM_CLIENTS}")
        print(f"✅ Processing time: {duration}")
        print(f"✅ Files saved in: {PROCESSED_PATH} and {FEDERATED_PATH}")
        print("\n🚀 Ready for Phase 3: CNN Model Development!")

    except Exception as e:
        logger.error(f"❌ Error in data processing: {e}")
        raise


def generate_summary_report(original_df, processed_df, train_data, test_data, save_path):
    """
    Generate a comprehensive summary report
    """
    report_path = os.path.join(save_path, "data_summary_report.txt")

    with open(report_path, 'w') as f:
        f.write("CICIDDOS2019 DATA PROCESSING SUMMARY REPORT\n")
        f.write("="*60 + "\n")
        f.write(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Original data summary
        f.write("ORIGINAL DATA:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Shape: {original_df.shape}\n")
        f.write(
            f"Memory usage: {original_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
        f.write(f"Label distribution:\n")
        for label, count in original_df['Label'].value_counts().items():
            f.write(
                f"  {label}: {count:,} ({count/len(original_df)*100:.1f}%)\n")
        f.write(f"Attack ratio: {original_df['Binary_Label'].mean():.2%}\n\n")

        # Processed data summary
        f.write("PROCESSED DATA:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Shape: {processed_df.shape}\n")
        # Excluding Label columns
        f.write(f"Features selected: {processed_df.shape[1] - 2}\n")
        f.write(
            f"Memory usage: {processed_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n")

        # Federated data summary
        f.write("FEDERATED DATA DISTRIBUTION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Number of clients: {len(train_data)}\n")

        total_train = sum(len(df) for df in train_data.values())
        total_test = sum(len(df) for df in test_data.values())

        f.write(f"Total training samples: {total_train:,}\n")
        f.write(f"Total test samples: {total_test:,}\n")

        f.write("\nPer-client distribution:\n")
        for client_id in range(len(train_data)):
            train_size = len(train_data[client_id])
            test_size = len(test_data[client_id])
            f.write(
                f"  Client {client_id}: Train={train_size:,}, Test={test_size:,}\n")

            if train_size > 0:
                train_labels = train_data[client_id]['Label'].value_counts()
                f.write(f"    Train labels: {dict(train_labels)}\n")

    logger.info(f"Summary report saved to {report_path}")


if __name__ == "__main__":
    main()
