#!/usr/bin/env python3
"""
Memory-Efficient Data Processing for CICIDDOS2019
Processes large dataset in chunks to avoid memory issues
"""

from data.federated_split import FederatedDataDistributor
from data.preprocessing import DataPreprocessor
from data.data_loader import CICDDoS2019Loader
import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import gc

# Add src to path
sys.path.append('src')


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MemoryEfficientProcessor:
    """Memory-efficient processor for large datasets"""

    def __init__(self, chunk_size=10000, sample_size_per_file=50000):
        self.chunk_size = chunk_size
        self.sample_size_per_file = sample_size_per_file
        self.preprocessor = DataPreprocessor()

    def process_single_file(self, file_path, attack_type):
        """Process a single CSV file with memory management"""
        logger.info(
            f"Processing {attack_type} from {os.path.basename(file_path)}")

        try:
            # Get file size first
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            logger.info(f"File size: {file_size_mb:.1f} MB")

            # Read with sampling for large files
            if file_size_mb > 500:  # If file > 500MB, sample it
                logger.info(
                    f"Large file detected, sampling {self.sample_size_per_file} records")
                # Read total lines first
                total_lines = sum(1 for line in open(
                    file_path, 'r', encoding='utf-8', errors='ignore')) - 1

                # Calculate skip probability
                if total_lines > self.sample_size_per_file:
                    skip_prob = 1 - (self.sample_size_per_file / total_lines)
                    skip_indices = np.random.choice(range(1, total_lines + 1),
                                                    size=int(
                                                        total_lines * skip_prob),
                                                    replace=False)
                    df = pd.read_csv(file_path, skiprows=skip_indices)
                else:
                    df = pd.read_csv(file_path)
            else:
                df = pd.read_csv(file_path)

            # Clean column names (remove spaces)
            df.columns = df.columns.str.strip()

            # Add labels
            df['Label'] = attack_type
            df['Binary_Label'] = 1 if attack_type != 'BENIGN' else 0

            # Basic cleaning
            df = df.dropna(how='all')
            df = df.replace([np.inf, -np.inf], np.nan)

            logger.info(
                f"Processed {attack_type}: {len(df)} records, {df.shape[1]} features")

            # Free memory
            gc.collect()

            return df

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return pd.DataFrame()

    def create_balanced_dataset(self):
        """Create a balanced dataset from all attack types"""
        logger.info("Creating balanced dataset from CICIDDOS2019")

        data_path = "data/raw/CSV-01-12/01-12"
        attack_files = {
            'DrDoS_DNS': 'DrDoS_DNS.csv',
            'DrDoS_LDAP': 'DrDoS_LDAP.csv',
            'DrDoS_MSSQL': 'DrDoS_MSSQL.csv',
            'DrDoS_NetBIOS': 'DrDoS_NetBIOS.csv',
            'DrDoS_NTP': 'DrDoS_NTP.csv',
            'DrDoS_SNMP': 'DrDoS_SNMP.csv',
            'DrDoS_SSDP': 'DrDoS_SSDP.csv',
            'DrDoS_UDP': 'DrDoS_UDP.csv',
            'Syn': 'Syn.csv',
            'TFTP': 'TFTP.csv',
            'UDPLag': 'UDPLag.csv'
        }

        all_dataframes = []

        # Process each attack type
        for attack_type, filename in attack_files.items():
            file_path = os.path.join(data_path, filename)
            if os.path.exists(file_path):
                df = self.process_single_file(file_path, attack_type)
                if not df.empty:
                    all_dataframes.append(df)

                # Memory management
                gc.collect()

        if not all_dataframes:
            raise ValueError("No data was loaded!")

        # Combine all dataframes
        logger.info("Combining all attack types...")
        combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)

        # Clear individual dataframes from memory
        del all_dataframes
        gc.collect()

        logger.info(f"Combined dataset shape: {combined_df.shape}")
        logger.info(
            f"Attack distribution:\n{combined_df['Label'].value_counts()}")

        return combined_df


def main():
    """Main processing pipeline"""
    logger.info("🚀 Starting CICIDDOS2019 Memory-Efficient Processing")
    start_time = datetime.now()

    try:
        # Create directories
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("data/federated", exist_ok=True)

        # Step 1: Process dataset
        processor = MemoryEfficientProcessor(
            sample_size_per_file=30000)  # Reduced for memory
        combined_df = processor.create_balanced_dataset()

        # Step 2: Preprocessing
        logger.info("🔧 Starting preprocessing...")
        preprocessor = DataPreprocessor()

        # Process in smaller chunks if dataset is too large
        if len(combined_df) > 200000:  # If > 200k records, sample down
            logger.info(
                f"Large dataset detected ({len(combined_df)} records), sampling to 200k")
            combined_df = combined_df.sample(
                n=200000, random_state=42).reset_index(drop=True)

        processed_df = preprocessor.preprocess_pipeline(combined_df, fit=True)

        # Save preprocessor
        preprocessor.save_preprocessor("data/processed")

        # Save processed data
        processed_file = "data/processed/processed_data.csv"
        processed_df.to_csv(processed_file, index=False)
        logger.info(f"Processed data saved: {processed_df.shape}")

        # Step 3: Create federated splits
        logger.info("🏢 Creating federated data distribution...")
        distributor = FederatedDataDistributor(num_clients=6, random_state=42)

        # Create non-IID distribution
        client_data = distributor.create_non_iid_by_label(
            processed_df, concentration=0.6)

        # Split train/test
        train_data, test_data = distributor.split_train_test(
            client_data, test_size=0.2)

        # Save federated data
        distributor.save_federated_data(train_data, "data/federated", "train")
        distributor.save_federated_data(test_data, "data/federated", "test")

        # Step 4: Generate summary
        generate_comprehensive_summary(
            combined_df, processed_df, train_data, test_data)

        # Success!
        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "="*60)
        print("🎉 PHASE 2 COMPLETE!")
        print("="*60)
        print(f"✅ Original samples: {len(combined_df):,}")
        print(f"✅ Processed samples: {len(processed_df):,}")
        print(f"✅ Features: {processed_df.shape[1] - 2}")  # Excluding labels
        print(f"✅ Federated clients: {len(train_data)}")
        print(f"✅ Processing time: {duration}")
        print(f"✅ Files saved in: data/processed/ and data/federated/")
        print("\n🚀 Ready for Phase 3: CNN Model Development!")

        return True

    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        return False


def generate_comprehensive_summary(original_df, processed_df, train_data, test_data):
    """Generate detailed summary report"""

    summary_path = "data/processed/PHASE2_COMPLETION_REPORT.txt"

    with open(summary_path, 'w') as f:
        f.write("CICIDDOS2019 PHASE 2 COMPLETION REPORT\n")
        f.write("="*60 + "\n")
        f.write(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Dataset summary
        f.write("ORIGINAL DATASET:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total samples: {len(original_df):,}\n")
        f.write(f"Total features: {original_df.shape[1]}\n")
        f.write(
            f"Memory usage: {original_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB\n")

        f.write(f"\nAttack distribution:\n")
        for label, count in original_df['Label'].value_counts().items():
            percentage = count / len(original_df) * 100
            f.write(f"  {label}: {count:,} ({percentage:.1f}%)\n")

        # Processed dataset
        f.write(f"\nPROCESSED DATASET:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Samples: {len(processed_df):,}\n")
        f.write(f"Features selected: {processed_df.shape[1] - 2}\n")
        f.write(
            f"Data quality: {((processed_df.isnull().sum().sum() / processed_df.size) * 100):.2f}% missing\n")

        # Federated distribution
        f.write(f"\nFEDERATED DISTRIBUTION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Number of clients: {len(train_data)}\n")

        total_train = sum(len(df) for df in train_data.values())
        total_test = sum(len(df) for df in test_data.values())

        f.write(f"Training samples: {total_train:,}\n")
        f.write(f"Test samples: {total_test:,}\n")

        f.write(f"\nClient distribution:\n")
        for client_id in range(len(train_data)):
            train_size = len(train_data[client_id])
            test_size = len(test_data[client_id])
            f.write(
                f"  Client {client_id}: Train={train_size:,}, Test={test_size:,}\n")

            if train_size > 0:
                labels = train_data[client_id]['Label'].value_counts()
                f.write(f"    Labels: {dict(labels)}\n")

        # Phase 2 completion
        f.write(f"\nPHASE 2 STATUS: ✅ COMPLETE\n")
        f.write("Next: Phase 3 - CNN Architecture Development\n")

    logger.info(f"Comprehensive report saved: {summary_path}")


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Phase 2 completed successfully!")
        print("📁 Check data/processed/PHASE2_COMPLETION_REPORT.txt for details")
    else:
        print("\n❌ Phase 2 completion failed. Check logs for details.")
