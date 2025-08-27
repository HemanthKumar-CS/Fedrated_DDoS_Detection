#!/usr/bin/env python3
"""
Real-time DDoS Detection Demo
Demonstrates model inference on new network traffic data
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import json
from sklearn.preprocessing import StandardScaler
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DDoSDetector:
    """Real-time DDoS Detection System"""
    
    def __init__(self, model_path: str):
        """Initialize the DDoS detector with a trained model"""
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.attack_types = {
            0: "BENIGN",
            1: "DDoS ATTACK"
        }
        
        self.load_model()
        self.setup_preprocessing()
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"✅ Model loaded from: {self.model_path}")
            logger.info(f"Model input shape: {self.model.input_shape}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            raise
    
    def setup_preprocessing(self):
        """Setup preprocessing pipeline"""
        # Load balanced dataset to get feature names and scaler parameters
        try:
            sample_data = pd.read_csv("data/optimized/balanced_dataset.csv")
            feature_columns = [col for col in sample_data.columns if col != 'Binary_Label']
            self.feature_names = feature_columns
            
            # Fit scaler on sample data
            X_sample = sample_data[feature_columns].values
            self.scaler = StandardScaler()
            self.scaler.fit(X_sample)
            
            logger.info(f"✅ Preprocessing setup complete. Features: {len(self.feature_names)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup preprocessing: {str(e)}")
            raise
    
    def preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """Preprocess input data for model inference"""
        # Normalize features
        data_scaled = self.scaler.transform(data)
        
        # Reshape for CNN input (samples, features, 1)
        data_reshaped = data_scaled.reshape(data_scaled.shape[0], data_scaled.shape[1], 1)
        
        return data_reshaped
    
    def predict(self, data: np.ndarray) -> dict:
        """Make predictions on network traffic data"""
        start_time = datetime.now()
        
        # Preprocess data
        processed_data = self.preprocess_data(data)
        
        # Make predictions
        predictions = self.model.predict(processed_data, verbose=0)
        predicted_classes = (predictions > 0.5).astype(int).flatten()
        confidence_scores = predictions.flatten()
        
        inference_time = datetime.now() - start_time
        
        # Format results
        results = []
        for i in range(len(predicted_classes)):
            result = {
                "sample_id": i,
                "prediction": self.attack_types[predicted_classes[i]],
                "confidence": float(confidence_scores[i]),
                "is_attack": bool(predicted_classes[i]),
                "risk_level": self.get_risk_level(confidence_scores[i], predicted_classes[i])
            }
            results.append(result)
        
        summary = {
            "total_samples": len(data),
            "attacks_detected": int(np.sum(predicted_classes)),
            "benign_traffic": int(len(data) - np.sum(predicted_classes)),
            "attack_rate": float(np.mean(predicted_classes)),
            "inference_time": str(inference_time),
            "samples_per_second": len(data) / inference_time.total_seconds()
        }
        
        return {
            "predictions": results,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_risk_level(self, confidence: float, is_attack: int) -> str:
        """Determine risk level based on confidence score"""
        if is_attack:
            if confidence >= 0.9:
                return "CRITICAL"
            elif confidence >= 0.7:
                return "HIGH"
            else:
                return "MEDIUM"
        else:
            return "LOW"
    
    def batch_predict_from_file(self, file_path: str, sample_size: int = 1000) -> dict:
        """Predict on batch of data from file"""
        logger.info(f"📂 Loading data from: {file_path}")
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Sample data if needed
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            logger.info(f"📊 Sampled {sample_size} records from {len(df)} total")
        
        # Prepare features
        X = df[self.feature_names].values
        
        # Get actual labels if available
        actual_labels = None
        if 'Binary_Label' in df.columns:
            actual_labels = df['Binary_Label'].values
        
        # Make predictions
        results = self.predict(X)
        
        # Add accuracy if actual labels are available
        if actual_labels is not None:
            predicted_labels = np.array([1 if r["is_attack"] else 0 for r in results["predictions"]])
            accuracy = np.mean(predicted_labels == actual_labels)
            results["validation"] = {
                "accuracy": float(accuracy),
                "true_positives": int(np.sum((predicted_labels == 1) & (actual_labels == 1))),
                "false_positives": int(np.sum((predicted_labels == 1) & (actual_labels == 0))),
                "true_negatives": int(np.sum((predicted_labels == 0) & (actual_labels == 0))),
                "false_negatives": int(np.sum((predicted_labels == 0) & (actual_labels == 1)))
            }
        
        return results

def create_sample_traffic():
    """Create sample network traffic for demonstration"""
    logger.info("🔧 Creating sample network traffic data...")
    
    # Load some real data for realistic sampling
    try:
        df = pd.read_csv("data/optimized/balanced_dataset.csv")
        # Sample some benign and attack traffic
        benign_samples = df[df['Binary_Label'] == 0].sample(n=5, random_state=42)
        attack_samples = df[df['Binary_Label'] == 1].sample(n=5, random_state=42)
        
        sample_data = pd.concat([benign_samples, attack_samples]).reset_index(drop=True)
        feature_columns = [col for col in sample_data.columns if col != 'Binary_Label']
        
        return sample_data[feature_columns].values, sample_data['Binary_Label'].values
        
    except Exception as e:
        logger.warning(f"Could not load real data, creating synthetic data: {str(e)}")
        # Create synthetic data if real data not available
        np.random.seed(42)
        num_samples = 10
        num_features = 29  # Based on optimized dataset
        
        # Create mixed traffic (some normal, some suspicious)
        data = np.random.randn(num_samples, num_features)
        labels = np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])
        
        return data, labels

def demo_real_time_detection():
    """Demonstrate real-time DDoS detection"""
    logger.info("🚀 Starting DDoS Detection Demo")
    logger.info("=" * 60)
    
    # Initialize detector
    detector = DDoSDetector("results/balanced_centralized_model.h5")
    
    # Create sample traffic
    sample_data, actual_labels = create_sample_traffic()
    
    logger.info(f"📡 Analyzing {len(sample_data)} network traffic samples...")
    
    # Make predictions
    results = detector.predict(sample_data)
    
    # Display results
    logger.info("\n📊 DETECTION RESULTS:")
    logger.info("-" * 40)
    
    for i, prediction in enumerate(results["predictions"]):
        actual = "ATTACK" if actual_labels[i] == 1 else "BENIGN"
        status = "✅ CORRECT" if (prediction["is_attack"] and actual_labels[i] == 1) or (not prediction["is_attack"] and actual_labels[i] == 0) else "❌ INCORRECT"
        
        logger.info(f"Sample {i+1:2d}: {prediction['prediction']:12s} | Confidence: {prediction['confidence']:.3f} | Risk: {prediction['risk_level']:8s} | Actual: {actual:6s} | {status}")
    
    # Display summary
    logger.info("\n📈 SUMMARY:")
    logger.info("-" * 40)
    summary = results["summary"]
    logger.info(f"Total Samples:     {summary['total_samples']}")
    logger.info(f"Attacks Detected:  {summary['attacks_detected']}")
    logger.info(f"Benign Traffic:    {summary['benign_traffic']}")
    logger.info(f"Attack Rate:       {summary['attack_rate']:.2%}")
    logger.info(f"Inference Time:    {summary['inference_time']}")
    logger.info(f"Processing Speed:  {summary['samples_per_second']:.1f} samples/second")
    
    # Calculate accuracy
    predicted = np.array([p["is_attack"] for p in results["predictions"]])
    accuracy = np.mean(predicted == actual_labels)
    logger.info(f"Demo Accuracy:     {accuracy:.2%}")
    
    # Save demo results
    demo_results = {
        "demo_timestamp": datetime.now().isoformat(),
        "model_path": "results/balanced_centralized_model.h5",
        "demo_accuracy": float(accuracy),
        "results": results
    }
    
    with open("results/demo_results.json", "w") as f:
        json.dump(demo_results, f, indent=2, default=str)
    
    logger.info("\n💾 Demo results saved to: results/demo_results.json")

def benchmark_model_performance():
    """Benchmark model performance on larger dataset"""
    logger.info("\n🔬 PERFORMANCE BENCHMARK")
    logger.info("=" * 60)
    
    detector = DDoSDetector("results/balanced_centralized_model.h5")
    
    # Test on client data if available
    test_files = [
        "data/optimized/client_0_test.csv",
        "data/optimized/client_1_test.csv", 
        "data/optimized/client_2_test.csv",
        "data/optimized/client_3_test.csv"
    ]
    
    total_accuracy = []
    total_samples = 0
    total_time = 0
    
    for i, test_file in enumerate(test_files):
        if os.path.exists(test_file):
            logger.info(f"\n📋 Testing on Client {i} data...")
            results = detector.batch_predict_from_file(test_file, sample_size=500)
            
            if "validation" in results:
                accuracy = results["validation"]["accuracy"]
                total_accuracy.append(accuracy)
                logger.info(f"Client {i} Accuracy: {accuracy:.4f}")
                logger.info(f"True Positives: {results['validation']['true_positives']}")
                logger.info(f"False Positives: {results['validation']['false_positives']}")
            
            total_samples += results["summary"]["total_samples"]
            inference_time = datetime.fromisoformat(results["timestamp"]) - datetime.fromisoformat(results["timestamp"])
    
    if total_accuracy:
        avg_accuracy = np.mean(total_accuracy)
        logger.info(f"\n🎯 OVERALL PERFORMANCE:")
        logger.info(f"Average Accuracy: {avg_accuracy:.4f}")
        logger.info(f"Total Samples Tested: {total_samples}")

if __name__ == "__main__":
    try:
        # Check if model exists
        model_path = "results/balanced_centralized_model.h5"
        if not os.path.exists(model_path):
            logger.error(f"❌ Model not found: {model_path}")
            logger.error("Please train the model first using: python train_balanced.py")
            sys.exit(1)
        
        # Run demonstration
        demo_real_time_detection()
        
        # Run benchmark if client data is available
        if os.path.exists("data/optimized/client_0_test.csv"):
            benchmark_model_performance()
        
        logger.info("\n🎉 DDoS Detection Demo Complete!")
        
    except Exception as e:
        logger.error(f"❌ Error during demo: {str(e)}")
        sys.exit(1)
