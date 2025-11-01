#!/usr/bin/env python3
"""
Production Inference Script - Real Data Testing
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import tensorflow as tf
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionInference:
    """Real-time inference with trained model"""

    def __init__(self, model_path='results/ddos_model.h5'):
        self.model = None
        self.scaler = None
        self.load_model(model_path)

    def load_model(self, model_path):
        """Load trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = tf.keras.models.load_model(model_path)
        logger.info(f"✅ Model loaded: {model_path}")

        scaler_path = 'results/scaler.pkl'
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info(f"✅ Scaler loaded: {scaler_path}")

    def predict_on_file(self, file_path, limit=None):
        """Run inference on real test data"""
        logger.info(f"\n📂 Testing on: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)

        if limit:
            df = df.head(limit)

        logger.info(f"   Samples: {len(df)}")

        # Get data
        y_true = df['Binary_Label'].values
        feature_cols = [col for col in df.columns if col not in [
            'Binary_Label', 'Label']]
        X = df[feature_cols].values

        # Preprocess
        if self.scaler:
            X = self.scaler.transform(X)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Predict
        logger.info(f"   Running inference...")
        y_pred_proba = self.model.predict(X, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # Metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_pred_proba)

        logger.info(f"\n   ✅ Results:")
        logger.info(f"      Accuracy:  {acc:.4f}")
        logger.info(f"      Precision: {prec:.4f}")
        logger.info(f"      Recall:    {rec:.4f}")
        logger.info(f"      F1-Score:  {f1:.4f}")
        logger.info(f"      ROC-AUC:   {auc:.4f}")

        attacks = np.sum(y_pred == 1)
        benign = np.sum(y_pred == 0)
        logger.info(f"\n   📊 Predictions:")
        logger.info(
            f"      Attacks: {attacks} ({attacks/len(y_pred)*100:.1f}%)")
        logger.info(f"      Benign:  {benign} ({benign/len(y_pred)*100:.1f}%)")

        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': auc,
            'attacks_detected': int(attacks),
            'benign_detected': int(benign)
        }


def main():
    try:
        logger.info("="*70)
        logger.info("🔥 PRODUCTION INFERENCE - REAL DATA")
        logger.info("="*70)

        inferencer = ProductionInference()

        results = {}

        # Test on all 4 clients
        for client_id in range(4):
            test_file = f"data/optimized/clean_partitions/client_{client_id}_test.csv"

            if os.path.exists(test_file):
                logger.info(f"\n{'='*70}")
                logger.info(f"CLIENT {client_id}")
                logger.info(f"{'='*70}")

                result = inferencer.predict_on_file(test_file)
                results[f'client_{client_id}'] = result

        # Summary
        if results:
            logger.info(f"\n{'='*70}")
            logger.info("OVERALL SUMMARY")
            logger.info(f"{'='*70}")

            avg_acc = np.mean([r['accuracy'] for r in results.values()])
            avg_prec = np.mean([r['precision'] for r in results.values()])
            avg_rec = np.mean([r['recall'] for r in results.values()])
            avg_f1 = np.mean([r['f1'] for r in results.values()])
            avg_auc = np.mean([r['roc_auc'] for r in results.values()])

            logger.info(f"\n✅ Average Metrics:")
            logger.info(f"   Accuracy:  {avg_acc:.4f}")
            logger.info(f"   Precision: {avg_prec:.4f}")
            logger.info(f"   Recall:    {avg_rec:.4f}")
            logger.info(f"   F1-Score:  {avg_f1:.4f}")
            logger.info(f"   ROC-AUC:   {avg_auc:.4f}")

            # Save
            with open('results/inference_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"\n✅ Results saved: results/inference_results.json")

        logger.info("\n" + "="*70)
        logger.info("✅ INFERENCE COMPLETE")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"❌ Failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
