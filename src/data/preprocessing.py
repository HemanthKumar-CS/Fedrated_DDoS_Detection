#!/usr/bin/env python3
"""
Data Preprocessing Module for CICDDoS2019 Dataset
Handles cleaning, feature engineering, and preparation for federated learning
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict, List
import logging
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessing for DDoS detection
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = None
        self.target_column = 'Binary_Label'

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values, duplicates, and outliers

        Args:
            df: Input DataFrame

        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning...")
        initial_shape = df.shape

        # Remove rows with all NaN values
        df = df.dropna(how='all')

        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)

        # Remove duplicates
        df = df.drop_duplicates()

        # Handle missing values in numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        non_target_numeric = [
            col for col in numeric_columns if col not in ['Label', 'Binary_Label']]

        if non_target_numeric:
            df[non_target_numeric] = self.imputer.fit_transform(
                df[non_target_numeric])

        logger.info(
            f"Data cleaning complete. Shape: {initial_shape} -> {df.shape}")
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features for better DDoS detection

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features...")
        df = df.copy()

        # Example feature engineering (adjust based on actual column names)
        try:
            # Traffic intensity features
            if ' Flow Duration' in df.columns:
                df['Flow_Duration_Log'] = np.log1p(df[' Flow Duration'].abs())

            if ' Total Fwd Packets' in df.columns and ' Total Backward Packets' in df.columns:
                df['Total_Packets'] = df[' Total Fwd Packets'] + \
                    df[' Total Backward Packets']
                df['Fwd_Bwd_Ratio'] = df[' Total Fwd Packets'] / \
                    (df[' Total Backward Packets'] + 1)

            # Packet size statistics
            if ' Fwd Packet Length Max' in df.columns and ' Fwd Packet Length Min' in df.columns:
                df['Fwd_Packet_Length_Range'] = df[' Fwd Packet Length Max'] - \
                    df[' Fwd Packet Length Min']

            # Flow rate features
            if ' Flow Duration' in df.columns and 'Total_Packets' in df.columns:
                df['Packet_Rate'] = df['Total_Packets'] / \
                    (df[' Flow Duration'] + 1)

            logger.info(f"Feature engineering complete. New shape: {df.shape}")

        except Exception as e:
            logger.warning(f"Feature engineering partially failed: {e}")

        return df

    def select_features(self, df: pd.DataFrame, max_features: int = 50) -> pd.DataFrame:
        """
        Select most relevant features for training

        Args:
            df: Input DataFrame
            max_features: Maximum number of features to select

        Returns:
            DataFrame with selected features
        """
        logger.info("Selecting features...")

        # Identify numeric columns (excluding target columns)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        feature_columns = [
            col for col in numeric_columns if col not in ['Label', 'Binary_Label']]

        # Remove columns with very low variance
        variances = df[feature_columns].var()
        high_variance_features = variances[variances > 0.001].index.tolist()

        # Limit to max_features if necessary
        if len(high_variance_features) > max_features:
            # Use correlation with target to select top features
            correlations = df[high_variance_features].corrwith(
                df[self.target_column]).abs()
            top_features = correlations.nlargest(max_features).index.tolist()
            selected_features = top_features
        else:
            selected_features = high_variance_features

        # Add target columns back
        selected_features.extend(['Label', 'Binary_Label'])

        self.feature_columns = [
            col for col in selected_features if col not in ['Label', 'Binary_Label']]

        logger.info(f"Selected {len(self.feature_columns)} features")
        return df[selected_features]

    def normalize_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normalize numeric features using StandardScaler

        Args:
            df: Input DataFrame
            fit: Whether to fit the scaler (True for training data)

        Returns:
            DataFrame with normalized features
        """
        logger.info("Normalizing features...")
        df = df.copy()

        if self.feature_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            self.feature_columns = [
                col for col in numeric_columns if col not in ['Label', 'Binary_Label']]

        if fit:
            df[self.feature_columns] = self.scaler.fit_transform(
                df[self.feature_columns])
        else:
            df[self.feature_columns] = self.scaler.transform(
                df[self.feature_columns])

        return df

    def preprocess_pipeline(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Complete preprocessing pipeline

        Args:
            df: Input DataFrame
            fit: Whether to fit preprocessing components

        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting preprocessing pipeline...")

        # Step 1: Clean data
        df = self.clean_data(df)

        # Step 2: Engineer features
        df = self.engineer_features(df)

        # Step 3: Select features
        df = self.select_features(df)

        # Step 4: Normalize features
        df = self.normalize_features(df, fit=fit)

        logger.info(f"Preprocessing complete. Final shape: {df.shape}")
        return df

    def save_preprocessor(self, save_path: str):
        """
        Save the fitted preprocessor components

        Args:
            save_path: Directory to save preprocessor components
        """
        os.makedirs(save_path, exist_ok=True)

        joblib.dump(self.scaler, os.path.join(save_path, 'scaler.pkl'))
        joblib.dump(self.imputer, os.path.join(save_path, 'imputer.pkl'))
        joblib.dump(self.feature_columns, os.path.join(
            save_path, 'feature_columns.pkl'))

        logger.info(f"Preprocessor saved to {save_path}")

    def load_preprocessor(self, load_path: str):
        """
        Load fitted preprocessor components

        Args:
            load_path: Directory containing preprocessor components
        """
        self.scaler = joblib.load(os.path.join(load_path, 'scaler.pkl'))
        self.imputer = joblib.load(os.path.join(load_path, 'imputer.pkl'))
        self.feature_columns = joblib.load(
            os.path.join(load_path, 'feature_columns.pkl'))

        logger.info(f"Preprocessor loaded from {load_path}")


def main():
    """
    Test the preprocessing pipeline
    """
    # This would be called from a separate script
    # For now, just demonstrate the structure
    logger.info("DataPreprocessor module loaded successfully")
    print("✅ Data preprocessing module ready")
    print("Use this module to preprocess your CICDDoS2019 data")


if __name__ == "__main__":
    main()
