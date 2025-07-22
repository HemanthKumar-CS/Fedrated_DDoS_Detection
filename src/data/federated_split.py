#!/usr/bin/env python3
"""
Federated Data Distribution Module
Creates non-IID data partitions for simulating federated learning scenarios
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import os
import logging
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FederatedDataDistributor:
    """
    Creates non-IID data distributions for federated learning simulation
    """

    def __init__(self, num_clients: int = 5, random_state: int = 42):
        """
        Initialize the data distributor

        Args:
            num_clients: Number of federated clients to simulate
            random_state: Random seed for reproducibility
        """
        self.num_clients = num_clients
        self.random_state = random_state
        np.random.seed(random_state)

    def create_iid_distribution(self, df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """
        Create IID (identical and independently distributed) data distribution

        Args:
            df: Input DataFrame

        Returns:
            Dictionary mapping client_id to DataFrame
        """
        logger.info(
            f"Creating IID distribution for {self.num_clients} clients...")

        # Shuffle the data
        df_shuffled = df.sample(
            frac=1, random_state=self.random_state).reset_index(drop=True)

        # Split into equal chunks
        chunk_size = len(df_shuffled) // self.num_clients
        client_data = {}

        for client_id in range(self.num_clients):
            start_idx = client_id * chunk_size
            if client_id == self.num_clients - 1:  # Last client gets remaining data
                end_idx = len(df_shuffled)
            else:
                end_idx = (client_id + 1) * chunk_size

            client_data[client_id] = df_shuffled[start_idx:end_idx].copy()
            logger.info(
                f"Client {client_id}: {len(client_data[client_id])} samples")

        return client_data

    def create_non_iid_by_label(self, df: pd.DataFrame, concentration: float = 0.5) -> Dict[int, pd.DataFrame]:
        """
        Create non-IID distribution based on label concentration

        Args:
            df: Input DataFrame
            concentration: How concentrated labels should be (0=uniform, 1=very concentrated)

        Returns:
            Dictionary mapping client_id to DataFrame
        """
        logger.info(
            f"Creating non-IID distribution by labels (concentration={concentration})...")

        # Get unique attack types
        unique_labels = df['Label'].unique()
        logger.info(
            f"Found {len(unique_labels)} unique labels: {unique_labels}")

        client_data = {i: [] for i in range(self.num_clients)}

        # For each label, distribute data across clients with varying concentrations
        for label in unique_labels:
            label_data = df[df['Label'] == label].copy()

            # Create concentration weights for each client
            weights = np.random.dirichlet([concentration] * self.num_clients)

            # Distribute this label's data according to weights
            shuffled_data = label_data.sample(
                frac=1, random_state=self.random_state).reset_index(drop=True)

            start_idx = 0
            for client_id, weight in enumerate(weights):
                num_samples = int(len(shuffled_data) * weight)
                if client_id == self.num_clients - 1:  # Last client gets remaining
                    end_idx = len(shuffled_data)
                else:
                    end_idx = start_idx + num_samples

                client_samples = shuffled_data[start_idx:end_idx]
                client_data[client_id].append(client_samples)
                start_idx = end_idx

        # Combine all labels for each client
        final_client_data = {}
        for client_id in range(self.num_clients):
            if client_data[client_id]:
                client_df = pd.concat(
                    client_data[client_id], ignore_index=True)
                # Shuffle client's combined data
                final_client_data[client_id] = client_df.sample(
                    frac=1, random_state=self.random_state).reset_index(drop=True)

                # Log distribution for this client
                label_dist = client_df['Label'].value_counts()
                logger.info(
                    f"Client {client_id}: {len(client_df)} samples, distribution: {dict(label_dist)}")
            else:
                final_client_data[client_id] = pd.DataFrame()

        return final_client_data

    def create_non_iid_by_attack_type(self, df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """
        Create non-IID distribution where each client specializes in certain attack types

        Args:
            df: Input DataFrame

        Returns:
            Dictionary mapping client_id to DataFrame
        """
        logger.info(
            "Creating non-IID distribution by attack type specialization...")

        # Get attack types (excluding BENIGN)
        attack_types = [label for label in df['Label'].unique()
                        if label != 'BENIGN']
        benign_data = df[df['Label'] == 'BENIGN'].copy()

        # Distribute benign data equally among all clients
        benign_per_client = len(benign_data) // self.num_clients

        client_data = {}

        for client_id in range(self.num_clients):
            client_samples = []

            # Add benign data
            start_idx = client_id * benign_per_client
            if client_id == self.num_clients - 1:
                end_idx = len(benign_data)
            else:
                end_idx = (client_id + 1) * benign_per_client

            client_benign = benign_data[start_idx:end_idx]
            client_samples.append(client_benign)

            # Assign 1-2 attack types per client
            num_attacks_per_client = min(2, len(attack_types))
            if client_id < len(attack_types):
                # Primary attack type
                primary_attack = attack_types[client_id % len(attack_types)]
                attack_data = df[df['Label'] == primary_attack].copy()
                client_samples.append(attack_data)

                # Secondary attack type (if available)
                if num_attacks_per_client > 1 and len(attack_types) > 1:
                    secondary_attack = attack_types[(
                        client_id + 1) % len(attack_types)]
                    if secondary_attack != primary_attack:
                        secondary_data = df[df['Label']
                                            == secondary_attack].copy()
                        # Use only 30% of secondary attack data
                        secondary_sample = secondary_data.sample(
                            frac=0.3, random_state=self.random_state)
                        client_samples.append(secondary_sample)

            # Combine and shuffle
            if client_samples:
                client_df = pd.concat(client_samples, ignore_index=True)
                client_data[client_id] = client_df.sample(
                    frac=1, random_state=self.random_state).reset_index(drop=True)

                # Log distribution
                label_dist = client_df['Label'].value_counts()
                logger.info(
                    f"Client {client_id}: {len(client_df)} samples, distribution: {dict(label_dist)}")
            else:
                client_data[client_id] = pd.DataFrame()

        return client_data

    def split_train_test(self, client_data: Dict[int, pd.DataFrame], test_size: float = 0.2) -> Tuple[Dict, Dict]:
        """
        Split each client's data into train and test sets

        Args:
            client_data: Dictionary of client DataFrames
            test_size: Proportion of data for testing

        Returns:
            Tuple of (train_data, test_data) dictionaries
        """
        logger.info(
            f"Splitting data into train/test (test_size={test_size})...")

        train_data = {}
        test_data = {}

        for client_id, df in client_data.items():
            if len(df) > 0:
                X = df.drop(['Label', 'Binary_Label'], axis=1)
                y = df['Binary_Label']

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=self.random_state, stratify=y
                )

                # Reconstruct DataFrames
                train_df = X_train.copy()
                train_df['Binary_Label'] = y_train
                train_df['Label'] = df.loc[y_train.index, 'Label']

                test_df = X_test.copy()
                test_df['Binary_Label'] = y_test
                test_df['Label'] = df.loc[y_test.index, 'Label']

                train_data[client_id] = train_df
                test_data[client_id] = test_df

                logger.info(
                    f"Client {client_id}: Train={len(train_df)}, Test={len(test_df)}")
            else:
                train_data[client_id] = pd.DataFrame()
                test_data[client_id] = pd.DataFrame()

        return train_data, test_data

    def save_federated_data(self, client_data: Dict[int, pd.DataFrame], save_path: str, data_type: str = "train"):
        """
        Save federated data partitions to disk

        Args:
            client_data: Dictionary of client DataFrames
            save_path: Base path for saving data
            data_type: Type of data (train/test/validation)
        """
        os.makedirs(save_path, exist_ok=True)

        for client_id, df in client_data.items():
            filename = f"client_{client_id}_{data_type}.csv"
            filepath = os.path.join(save_path, filename)
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {len(df)} samples to {filepath}")

        # Save summary
        summary = {
            'num_clients': len(client_data),
            'data_type': data_type,
            'client_sizes': {client_id: len(df) for client_id, df in client_data.items()}
        }

        summary_path = os.path.join(save_path, f"{data_type}_summary.txt")
        with open(summary_path, 'w') as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")


def main():
    """
    Test the federated data distribution
    """
    logger.info("FederatedDataDistributor module loaded successfully")
    print("✅ Federated data distribution module ready")
    print("Use this module to create non-IID data partitions for FL simulation")


if __name__ == "__main__":
    main()
