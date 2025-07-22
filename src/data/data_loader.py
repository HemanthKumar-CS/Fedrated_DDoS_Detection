#!/usr/bin/env python3
"""
Data Loading Module for CICDDoS2019 Dataset
Handles loading, combining, and basic preprocessing of DDoS attack data
"""

import pandas as pd
import numpy as np
import os
import glob
from typing import List, Tuple, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CICDDoS2019Loader:
    """
    Data loader for CICDDoS2019 dataset
    """

    def __init__(self, data_path: str = "../../data/raw/CSV-01-12/01-12"):
        """
        Initialize the data loader

        Args:
            data_path: Path to the directory containing CSV files
        """
        self.data_path = data_path
        self.attack_types = [
            'DrDoS_DNS', 'DrDoS_LDAP', 'DrDoS_MSSQL', 'DrDoS_NetBIOS',
            'DrDoS_NTP', 'DrDoS_SNMP', 'DrDoS_SSDP', 'DrDoS_UDP',
            'Syn', 'TFTP', 'UDPLag'
        ]

    def get_csv_files(self) -> List[str]:
        """
        Get list of all CSV files in the data directory

        Returns:
            List of CSV file paths
        """
        csv_pattern = os.path.join(self.data_path, "*.csv")
        csv_files = glob.glob(csv_pattern)
        logger.info(f"Found {len(csv_files)} CSV files")
        return csv_files

    def load_single_file(self, file_path: str) -> pd.DataFrame:
        """
        Load a single CSV file with error handling

        Args:
            file_path: Path to the CSV file

        Returns:
            DataFrame with loaded data
        """
        try:
            logger.info(f"Loading {os.path.basename(file_path)}...")
            df = pd.read_csv(file_path)

            # Add attack type label based on filename
            filename = os.path.basename(file_path).replace('.csv', '')
            if filename in self.attack_types:
                df['Label'] = filename  # Attack type
                df['Binary_Label'] = 1  # Attack
            else:
                df['Label'] = 'BENIGN'
                df['Binary_Label'] = 0  # Normal

            logger.info(f"Loaded {len(df)} records from {filename}")
            return df

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return pd.DataFrame()

    def load_all_data(self, sample_size: int = None) -> pd.DataFrame:
        """
        Load and combine all CSV files

        Args:
            sample_size: Optional limit on number of rows per file

        Returns:
            Combined DataFrame
        """
        csv_files = self.get_csv_files()
        dataframes = []

        for file_path in csv_files:
            df = self.load_single_file(file_path)
            if not df.empty:
                if sample_size:
                    df = df.sample(n=min(sample_size, len(df)),
                                   random_state=42)
                dataframes.append(df)

        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            logger.info(f"Combined dataset shape: {combined_df.shape}")
            return combined_df
        else:
            logger.error("No data loaded!")
            return pd.DataFrame()

    def get_dataset_info(self) -> Dict:
        """
        Get basic information about the dataset

        Returns:
            Dictionary with dataset statistics
        """
        csv_files = self.get_csv_files()
        info = {
            'total_files': len(csv_files),
            'file_sizes': {},
            'attack_types': self.attack_types
        }

        for file_path in csv_files:
            filename = os.path.basename(file_path).replace('.csv', '')
            try:
                df = pd.read_csv(file_path)
                info['file_sizes'][filename] = len(df)
            except Exception as e:
                logger.error(f"Error reading {filename}: {e}")
                info['file_sizes'][filename] = 0

        return info


def main():
    """
    Main function for testing the data loader
    """
    # Initialize loader
    loader = CICDDoS2019Loader()

    # Get dataset information
    logger.info("Getting dataset information...")
    info = loader.get_dataset_info()

    print("\n" + "="*50)
    print("CICDDOS2019 DATASET INFORMATION")
    print("="*50)
    print(f"Total files: {info['total_files']}")
    print(f"Attack types: {info['attack_types']}")
    print("\nFile sizes:")
    for filename, size in info['file_sizes'].items():
        print(f"  {filename}: {size:,} records")

    # Load a small sample for testing
    logger.info("Loading sample data...")
    sample_df = loader.load_all_data(sample_size=1000)  # 1000 records per file

    if not sample_df.empty:
        print(f"\nSample dataset shape: {sample_df.shape}")
        print(f"Columns: {sample_df.columns.tolist()}")
        print(f"\nLabel distribution:")
        print(sample_df['Label'].value_counts())
        print(f"\nBinary label distribution:")
        print(sample_df['Binary_Label'].value_counts())

        # Show basic statistics
        print(f"\nBasic statistics:")
        print(f"Total records: {len(sample_df):,}")
        print(f"Attack records: {(sample_df['Binary_Label'] == 1).sum():,}")
        print(f"Normal records: {(sample_df['Binary_Label'] == 0).sum():,}")
        print(f"Attack ratio: {(sample_df['Binary_Label'] == 1).mean():.2%}")


if __name__ == "__main__":
    main()
