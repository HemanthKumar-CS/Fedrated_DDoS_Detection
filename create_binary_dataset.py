#!/usr/bin/env python3
"""
Create Binary Classification Dataset for Federated DDoS Detection
Converts multi-class attack data to binary (benign vs attack) classification
"""

import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BinaryDatasetCreator:
    """Create binary classification dataset from existing optimized data"""

    def __init__(self):
        self.input_file = "data/optimized/optimized_dataset.csv"
        self.output_dir = "data/optimized"
        self.benign_ratio = 0.5  # 50% benign, 50% attack

    def create_binary_dataset(self):
        """
        Create binary dataset by:
        1. Using existing attack data as 'ATTACK' (label=1)
        2. Creating synthetic 'BENIGN' data by modifying attack patterns (label=0)
        """
        logger.info("Creating binary classification dataset...")

        # Load existing attack data
        if not os.path.exists(self.input_file):
            logger.error(f"Input file not found: {self.input_file}")
            return False

        df = pd.read_csv(self.input_file)
        logger.info(f"Loaded {len(df):,} attack samples")

        # All existing data becomes 'ATTACK' class
        attack_data = df.copy()
        attack_data['Label'] = 'ATTACK'
        attack_data['Binary_Label'] = 1

        # Create benign data by modifying attack patterns
        # This simulates normal network traffic characteristics
        benign_data = self._create_benign_samples(attack_data)

        # Combine attack and benign data
        binary_dataset = pd.concat(
            [attack_data, benign_data], ignore_index=True)

        # Shuffle the combined dataset
        binary_dataset = binary_dataset.sample(
            frac=1, random_state=42).reset_index(drop=True)

        # Save the binary dataset
        output_file = os.path.join(self.output_dir, "binary_dataset.csv")
        binary_dataset.to_csv(output_file, index=False)
        logger.info(f"Saved binary dataset: {output_file}")

        # Create federated splits
        self._create_federated_splits(binary_dataset)

        # Generate summary
        self._generate_summary(binary_dataset)

        return True

    def _create_benign_samples(self, attack_data):
        """
        Create benign samples by modifying attack data patterns
        This simulates normal network traffic behavior
        """
        logger.info("Creating benign samples...")

        # Calculate target number of benign samples
        num_attack = len(attack_data)
        num_benign = int(num_attack * self.benign_ratio /
                         (1 - self.benign_ratio))

        # Sample attack data to use as template for benign
        template_data = attack_data.sample(
            n=num_benign, replace=True, random_state=42)
        benign_data = template_data.copy()

        # Feature columns (exclude Label and Binary_Label)
        feature_cols = [col for col in benign_data.columns
                        if col not in ['Label', 'Binary_Label']]

        # Modify patterns to simulate benign traffic
        for col in feature_cols:
            if benign_data[col].dtype in ['int64', 'float64']:
                # For numeric features, apply transformations to simulate normal traffic
                if 'Duration' in col:
                    # Normal flows have shorter durations
                    benign_data[col] = benign_data[col] * \
                        np.random.uniform(0.1, 0.5, len(benign_data))
                elif 'Packets' in col or 'Length' in col:
                    # Normal flows have moderate packet counts/sizes
                    benign_data[col] = benign_data[col] * \
                        np.random.uniform(0.2, 0.8, len(benign_data))
                elif 'Bytes/s' in col or 'Packets/s' in col:
                    # Normal flows have lower rates
                    benign_data[col] = benign_data[col] * \
                        np.random.uniform(0.05, 0.3, len(benign_data))
                elif 'Flag' in col:
                    # Normal flows have fewer flag anomalies
                    benign_data[col] = benign_data[col] * \
                        np.random.uniform(0.0, 0.2, len(benign_data))
                else:
                    # General reduction for other features
                    benign_data[col] = benign_data[col] * \
                        np.random.uniform(0.3, 0.7, len(benign_data))

                # Ensure non-negative values
                benign_data[col] = benign_data[col].abs()

        # Set benign labels
        benign_data['Label'] = 'BENIGN'
        benign_data['Binary_Label'] = 0

        logger.info(f"Created {len(benign_data):,} benign samples")
        return benign_data

    def _create_federated_splits(self, binary_dataset):
        """Create federated client splits for the binary dataset"""
        logger.info("Creating federated client splits...")

        num_clients = 4

        # Separate benign and attack data
        benign_data = binary_dataset[binary_dataset['Binary_Label'] == 0]
        attack_data = binary_dataset[binary_dataset['Binary_Label'] == 1]

        # Split both classes among clients
        benign_splits = np.array_split(benign_data, num_clients)
        attack_splits = np.array_split(attack_data, num_clients)

        for client_id in range(num_clients):
            # Combine benign and attack data for each client
            client_data = pd.concat([benign_splits[client_id], attack_splits[client_id]],
                                    ignore_index=True)

            # Shuffle client data
            client_data = client_data.sample(
                frac=1, random_state=42+client_id).reset_index(drop=True)

            # Split into train/test (80/20)
            train_data, test_data = train_test_split(
                client_data, test_size=0.2, random_state=42,
                stratify=client_data['Binary_Label']
            )

            # Save client splits
            train_file = os.path.join(
                self.output_dir, f"client_{client_id}_train.csv")
            test_file = os.path.join(
                self.output_dir, f"client_{client_id}_test.csv")

            train_data.to_csv(train_file, index=False)
            test_data.to_csv(test_file, index=False)

            logger.info(
                f"Client {client_id}: Train={len(train_data)}, Test={len(test_data)}")

    def _generate_summary(self, binary_dataset):
        """Generate summary of the binary dataset"""

        # Calculate statistics
        total_samples = len(binary_dataset)
        benign_count = len(binary_dataset[binary_dataset['Binary_Label'] == 0])
        attack_count = len(binary_dataset[binary_dataset['Binary_Label'] == 1])

        summary = f"""BINARY CLASSIFICATION DATASET SUMMARY
=====================================
Total samples: {total_samples:,}
Benign samples: {benign_count:,} ({benign_count/total_samples*100:.1f}%)
Attack samples: {attack_count:,} ({attack_count/total_samples*100:.1f}%)

Class distribution:
{binary_dataset['Binary_Label'].value_counts().sort_index()}

Features: {len(binary_dataset.columns) - 2} (excluding Label and Binary_Label)
Memory usage: {binary_dataset.memory_usage(deep=True).sum() / 1024**2:.1f} MB

✅ BINARY DATASET READY FOR FEDERATED LEARNING!

Key advantages of binary classification:
- Faster convergence in federated learning
- More robust to data heterogeneity
- Simpler model architecture
- Better performance with limited data
- Easier to interpret results

Next steps:
1. Run federated training with binary model
2. Use binary_dataset.csv for centralized training
3. Use client_*_train.csv and client_*_test.csv for FL simulation
"""

        # Save summary
        summary_file = os.path.join(
            self.output_dir, "BINARY_DATASET_SUMMARY.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)

        print("\n" + "="*60)
        print("🎯 BINARY DATASET CREATION COMPLETE!")
        print("="*60)
        print(summary)


def main():
    """Main execution function"""
    try:
        start_time = datetime.now()

        creator = BinaryDatasetCreator()
        success = creator.create_binary_dataset()

        if success:
            print(
                f"\n✅ Binary dataset creation completed in {datetime.now() - start_time}")
            print(
                "💡 You can now train with binary classification for better FL performance!")
        else:
            print("\n❌ Binary dataset creation failed")

        return success

    except Exception as e:
        logger.error(f"❌ Error creating binary dataset: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Ready to start federated learning with binary classification!")
    else:
        print("\n❌ Please check the logs and try again")
