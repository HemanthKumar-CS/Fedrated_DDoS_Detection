#!/usr/bin/env python3
"""
Threat Detection Evaluation System
Measures DDoS detection performance: TPR, FPR, precision, recall, F1-score per client
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Tuple, List

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ThreatDetectionEvaluator:
    """Evaluate DDoS detection performance"""

    def __init__(self, model_path: str = 'results/ddos_model.h5', scaler_path: str = 'results/scaler.pkl'):
        """Initialize evaluator with trained model"""
        try:
            import tensorflow as tf
            from sklearn.preprocessing import StandardScaler
            import pickle
            self.model = tf.keras.models.load_model(model_path)

            # Try to load scaler, if corrupted create new one
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            except:
                logger.warning(
                    "Scaler corrupted, creating new one from training data...")
                self.scaler = StandardScaler()
                # Fit on combined training data from all clients
                train_data = []
                for i in range(4):
                    df = pd.read_csv(
                        f'data/optimized/clean_partitions/client_{i}_train.csv')
                    # Use same columns as training script
                    feature_cols = [col for col in df.columns if col not in [
                        'Binary_Label', 'Label']]
                    X = df[feature_cols].values
                    train_data.append(X)
                combined_X = np.vstack(train_data)
                self.scaler.fit(combined_X)
                # Save new scaler
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)

            logger.info(f"✅ Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.model = None
            self.scaler = None

    def evaluate_client(self, client_id: int, test_data_path: str = None) -> Dict:
        """
        Evaluate threat detection for single client

        Args:
            client_id: Client ID (0-3)
            test_data_path: Path to client test data CSV

        Returns:
            Dictionary with threat detection metrics
        """
        if self.model is None or self.scaler is None:
            logger.error("Model not loaded")
            return {}

        # Load test data
        if test_data_path is None:
            test_data_path = f'data/optimized/clean_partitions/client_{client_id}_test.csv'

        try:
            df = pd.read_csv(test_data_path)
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            return {}

        # Separate features and labels (use Binary_Label if available, else convert Label)
        if 'Binary_Label' in df.columns:
            label_col = 'Binary_Label'
            X = df.drop(['Label', 'Binary_Label'], axis=1).values
        else:
            label_col = 'Label'
            X = df.drop('Label', axis=1).values

        y_true = df[label_col].values

        # Convert string labels to binary if needed
        if y_true.dtype == 'object':
            y_true = (y_true == 'DDoS').astype(int)

        # Normalize features
        X_scaled = self.scaler.transform(X)

        # Get predictions
        y_pred_proba = self.model.predict(X_scaled, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # Calculate metrics
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        # Rates
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity/Recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False positive rate
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False negative rate

        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision +
                                         recall) if (precision + recall) > 0 else 0.0

        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # ROC-AUC
        from sklearn.metrics import roc_auc_score, roc_curve
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        fpr_curve, tpr_curve, _ = roc_curve(y_true, y_pred_proba)

        metrics = {
            'client_id': client_id,
            'samples': len(y_true),
            'benign_samples': np.sum(y_true == 0),
            'attack_samples': np.sum(y_true == 1),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'tpr': round(tpr, 4),
            'fpr': round(fpr, 4),
            'tnr': round(tnr, 4),
            'fnr': round(fnr, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'accuracy': round(accuracy, 4),
            'roc_auc': round(roc_auc, 4),
            # Primary metric for DDoS detection
            'threat_detection_rate': round(tpr, 4),
            # Primary metric for benign traffic
            'false_alarm_rate': round(fpr, 4),
        }

        return metrics

    def evaluate_all_clients(self) -> Dict:
        """Evaluate all 4 clients and aggregate metrics"""
        logger.info("\n" + "="*70)
        logger.info("THREAT DETECTION EVALUATION - ALL CLIENTS")
        logger.info("="*70 + "\n")

        all_metrics = []

        for client_id in range(4):
            logger.info(f"Evaluating Client {client_id}...")
            metrics = self.evaluate_client(client_id)
            if metrics:
                all_metrics.append(metrics)

                # Log key metrics
                logger.info(f"  ✅ Accuracy: {metrics['accuracy']:.4f}")
                logger.info(
                    f"  🎯 Threat Detection Rate (TPR): {metrics['tpr']:.4f}")
                logger.info(
                    f"  🛑 False Alarm Rate (FPR): {metrics['fpr']:.4f}")
                logger.info(f"  📊 F1-Score: {metrics['f1_score']:.4f}")
                logger.info(f"  📈 ROC-AUC: {metrics['roc_auc']:.4f}\n")

        # Aggregate metrics
        df_metrics = pd.DataFrame(all_metrics)

        aggregated = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'total_clients': len(all_metrics),
            'per_client_metrics': all_metrics,
            'aggregated_metrics': {
                'mean_accuracy': round(df_metrics['accuracy'].mean(), 4),
                'mean_tpr': round(df_metrics['tpr'].mean(), 4),
                'mean_fpr': round(df_metrics['fpr'].mean(), 4),
                'mean_precision': round(df_metrics['precision'].mean(), 4),
                'mean_recall': round(df_metrics['recall'].mean(), 4),
                'mean_f1_score': round(df_metrics['f1_score'].mean(), 4),
                'mean_roc_auc': round(df_metrics['roc_auc'].mean(), 4),
                'std_accuracy': round(df_metrics['accuracy'].std(), 4),
                'std_tpr': round(df_metrics['tpr'].std(), 4),
                'std_fpr': round(df_metrics['fpr'].std(), 4),
            },
            'threat_detection_summary': {
                'all_clients_high_detection_rate': all(m['tpr'] > 0.85 for m in all_metrics),
                'all_clients_low_false_alarm': all(m['fpr'] < 0.10 for m in all_metrics),
                'min_tpr': round(min(m['tpr'] for m in all_metrics), 4),
                'max_fpr': round(max(m['fpr'] for m in all_metrics), 4),
                'average_threat_detection_rate': round(df_metrics['tpr'].mean(), 4),
                'average_false_alarm_rate': round(df_metrics['fpr'].mean(), 4),
            }
        }

        logger.info("="*70)
        logger.info("AGGREGATED METRICS ACROSS ALL CLIENTS")
        logger.info("="*70)
        logger.info(
            f"Mean Accuracy: {aggregated['aggregated_metrics']['mean_accuracy']:.4f}")
        logger.info(
            f"Mean Threat Detection Rate: {aggregated['aggregated_metrics']['mean_tpr']:.4f}")
        logger.info(
            f"Mean False Alarm Rate: {aggregated['aggregated_metrics']['mean_fpr']:.4f}")
        logger.info(
            f"Mean F1-Score: {aggregated['aggregated_metrics']['mean_f1_score']:.4f}")
        logger.info(
            f"Mean ROC-AUC: {aggregated['aggregated_metrics']['mean_roc_auc']:.4f}")
        logger.info("="*70 + "\n")

        return aggregated

    def save_results(self, results: Dict, filepath: str = 'results/threat_detection_report.json'):
        """Save evaluation results"""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"✅ Threat detection report saved: {filepath}")

    def generate_detection_matrix(self, results: Dict) -> pd.DataFrame:
        """Generate confusion matrix summary"""
        data = []
        for client_metrics in results['per_client_metrics']:
            data.append({
                'Client': client_metrics['client_id'],
                'TP': client_metrics['true_positives'],
                'TN': client_metrics['true_negatives'],
                'FP': client_metrics['false_positives'],
                'FN': client_metrics['false_negatives'],
                'Sensitivity': client_metrics['tpr'],
                'Specificity': client_metrics['tnr'],
                'Precision': client_metrics['precision'],
                'F1-Score': client_metrics['f1_score'],
            })

        df = pd.DataFrame(data)
        logger.info("\nConfusion Matrix Summary:")
        logger.info(df.to_string(index=False))

        return df


def test_threat_detection_evaluation():
    """Test threat detection evaluation system"""
    logger.info("\n" + "="*70)
    logger.info("TESTING THREAT DETECTION EVALUATION")
    logger.info("="*70 + "\n")

    evaluator = ThreatDetectionEvaluator()

    if evaluator.model is None:
        logger.error("❌ Model not available for testing")
        return False

    logger.info("Running threat detection evaluation on all clients...")
    results = evaluator.evaluate_all_clients()

    logger.info("Generating confusion matrix summary...")
    df_matrix = evaluator.generate_detection_matrix(results)

    logger.info("Saving results...")
    evaluator.save_results(results)

    logger.info("\n" + "="*70)
    logger.info("✅ THREAT DETECTION EVALUATION COMPLETE")
    logger.info("="*70 + "\n")

    return True


if __name__ == "__main__":
    import sys
    success = test_threat_detection_evaluation()
    sys.exit(0 if success else 1)
