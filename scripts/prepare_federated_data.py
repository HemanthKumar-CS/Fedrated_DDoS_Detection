#!/usr/bin/env python3
"""
Optimized Data Processing - Create Minimal Dataset for FL Demonstration
Reduces data size while maintaining project effectiveness
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

# Add src to path
sys.path.append('src')


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizedProcessor:
    """Create minimal but effective dataset for FL demonstration"""

    def __init__(self):
        self.selected_attacks = [
            'DrDoS_DNS',    # Amplification attack
            'Syn',          # SYN flood attack
            'TFTP',         # Protocol-specific attack
            'UDPLag'        # UDP-based attack
        ]
        self.samples_per_attack = 12500  # 50k total samples
        self.num_clients = 4
        self.target_features = 30

    def create_minimal_dataset(self):
        """Create minimal balanced dataset"""
        logger.info("Creating minimal dataset for FL demonstration")

        data_path = "data/raw/CSV-01-12/01-12"
        dataframes = []

        for attack_type in self.selected_attacks:
            file_path = os.path.join(data_path, f"{attack_type}.csv")

            if os.path.exists(file_path):
                logger.info(f"Loading {attack_type}...")

                # Read with sampling
                if attack_type == 'UDPLag':
                    # UDPLag is smaller, take more
                    df = pd.read_csv(file_path)
                    if len(df) > self.samples_per_attack:
                        df = df.sample(
                            n=self.samples_per_attack, random_state=42)
                else:
                    # Sample from large files
                    total_lines = sum(1 for line in open(
                        file_path, 'r', encoding='utf-8', errors='ignore')) - 1
                    if total_lines > self.samples_per_attack:
                        skip_prob = 1 - (self.samples_per_attack / total_lines)
                        skip_indices = np.random.choice(range(1, total_lines + 1),
                                                        size=int(
                                                            total_lines * skip_prob),
                                                        replace=False)
                        df = pd.read_csv(file_path, skiprows=skip_indices)
                    else:
                        df = pd.read_csv(file_path)

                # Clean and label
                df.columns = df.columns.str.strip()
                df['Label'] = attack_type
                df['Binary_Label'] = 1

                # Take exactly the amount we need
                if len(df) > self.samples_per_attack:
                    df = df.sample(n=self.samples_per_attack, random_state=42)

                dataframes.append(df)
                logger.info(f"{attack_type}: {len(df)} samples")

        # Combine
        combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
        logger.info(f"Total samples: {len(combined_df)}")

        return combined_df

    def optimize_features(self, df):
        """Select most important features for DDoS detection"""
        logger.info("Selecting optimal features for DDoS detection")

        # Key network traffic features for DDoS detection
        important_features = [
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
            'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
            'Fwd IAT Mean', 'Bwd IAT Mean', 'Fwd Packet Length Mean',
            'Bwd Packet Length Mean', 'Fwd Packets/s', 'Bwd Packets/s',
            'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
            'Packet Length Std', 'Down/Up Ratio', 'Average Packet Size',
            'Fwd Header Length', 'Bwd Header Length', 'FIN Flag Count',
            'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
            'ACK Flag Count', 'URG Flag Count', 'Active Mean', 'Idle Mean'
        ]

        # Find available features (handle column name variations)
        available_features = []
        for feature in important_features:
            # Try exact match
            if feature in df.columns:
                available_features.append(feature)
            # Try with leading space
            elif f' {feature}' in df.columns:
                available_features.append(f' {feature}')
            # Try variations
            elif feature.replace(' ', '_') in df.columns:
                available_features.append(feature.replace(' ', '_'))

        # Add label columns
        selected_columns = available_features + ['Label', 'Binary_Label']

        # Keep only selected columns that exist
        final_columns = [col for col in selected_columns if col in df.columns]

        optimized_df = df[final_columns].copy()

        logger.info(
            f"Features reduced from {df.shape[1]} to {len(final_columns)-2}")
        return optimized_df


def main():
    """Create optimized dataset"""
    logger.info("🎯 Creating Optimized Dataset for FL Demonstration")
    start_time = datetime.now()

    try:
        processor = OptimizedProcessor()

        # Step 1: Create minimal dataset
        df = processor.create_minimal_dataset()

        # Step 2: Optimize features
        df = processor.optimize_features(df)

        # Step 3: Basic preprocessing
        logger.info("Applying basic preprocessing...")

        # Remove problematic columns
        df = df.dropna(how='all')
        df = df.replace([np.inf, -np.inf], np.nan)

        # Fill missing values with median for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        feature_columns = [
            col for col in numeric_columns if col not in ['Label', 'Binary_Label']]

        for col in feature_columns:
            df[col] = df[col].fillna(df[col].median())

        # Step 4: Create federated splits
        logger.info("Creating optimized federated distribution...")

        distributor = FederatedDataDistributor(
            num_clients=processor.num_clients, random_state=42)
        client_data = distributor.create_non_iid_by_attack_type(df)
        train_data, test_data = distributor.split_train_test(
            client_data, test_size=0.2)

        # Step 5: Save optimized data
        os.makedirs("data/optimized", exist_ok=True)

        # Save combined dataset
        df.to_csv("data/optimized/optimized_dataset.csv", index=False)

        # Save federated splits
        for client_id in range(processor.num_clients):
            train_data[client_id].to_csv(
                f"data/optimized/client_{client_id}_train.csv", index=False)
            test_data[client_id].to_csv(
                f"data/optimized/client_{client_id}_test.csv", index=False)

        # Generate summary
        total_train = sum(len(df) for df in train_data.values())
        total_test = sum(len(df) for df in test_data.values())

        summary = f"""OPTIMIZED DATASET SUMMARY
=========================
Attack types: {len(processor.selected_attacks)} (reduced from 11)
Total samples: {len(df):,} (reduced from 200k)
Features: {len(feature_columns)} (reduced from 50)
Clients: {processor.num_clients}
Training samples: {total_train:,}
Test samples: {total_test:,}

Per-client distribution:
"""
        for client_id in range(processor.num_clients):
            train_size = len(train_data[client_id])
            test_size = len(test_data[client_id])
            summary += f"Client {client_id}: Train={train_size:,}, Test={test_size:,}\n"

            if train_size > 0:
                labels = train_data[client_id]['Label'].value_counts()
                summary += f"  Primary attacks: {', '.join(labels.head(2).index.tolist())}\n"

        summary += f"""
Memory usage: ~{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB
Processing time: {datetime.now() - start_time}

✅ OPTIMIZED DATASET READY FOR FEDERATED LEARNING!
"""

        with open("data/optimized/OPTIMIZATION_SUMMARY.txt", 'w') as f:
            f.write(summary)

        print("\n" + "="*60)
        print("🎯 DATASET OPTIMIZATION COMPLETE!")
        print("="*60)
        print(summary)

        return True

    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Use data/optimized/ for your FL training!")
        print("💡 You can now delete data/processed/ to save space")
    else:
        print("\n❌ Optimization failed")
