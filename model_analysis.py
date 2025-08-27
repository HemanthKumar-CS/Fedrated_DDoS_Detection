#!/usr/bin/env python3
"""
Advanced Model Analysis and Evaluation
Comprehensive analysis of the trained DDoS detection model
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelAnalyzer:
    """Advanced model analysis and evaluation"""
    
    def __init__(self, model_path: str, data_path: str):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        
        self.load_model()
        self.load_test_data()
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"✅ Model loaded: {self.model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            raise
    
    def load_test_data(self):
        """Load and prepare test data"""
        try:
            # Load data
            df = pd.read_csv(self.data_path)
            
            # Handle different label column names and types
            if 'Binary_Label' in df.columns:
                label_col = 'Binary_Label'
                feature_columns = [col for col in df.columns if col != 'Binary_Label']
            elif 'Label' in df.columns:
                # Convert string labels to binary if needed
                if df['Label'].dtype == 'object':
                    df['Binary_Label'] = (df['Label'] != 'BENIGN').astype(int)
                    label_col = 'Binary_Label'
                else:
                    label_col = 'Label'
                feature_columns = [col for col in df.columns if col not in ['Label', 'Binary_Label']]
            else:
                # Assume last column is label
                label_col = df.columns[-1]
                if df[label_col].dtype == 'object':
                    df['Binary_Label'] = (df[label_col] != 'BENIGN').astype(int)
                    label_col = 'Binary_Label'
                feature_columns = df.columns[:-1].tolist()
            
            # Select only numeric features
            numeric_features = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
            X = df[numeric_features].values
            y = df[label_col].values
            
            # Normalize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Reshape for CNN
            self.X_test = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
            self.y_test = y
            
            logger.info(f"✅ Test data loaded: {self.X_test.shape}")
            logger.info(f"Features used: {len(numeric_features)}")
            logger.info(f"Label distribution: {np.bincount(y)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load test data: {str(e)}")
            raise
    
    def make_predictions(self):
        """Generate predictions on test data"""
        logger.info("🔮 Generating predictions...")
        
        # Get probability predictions
        self.y_pred_proba = self.model.predict(self.X_test, verbose=0)
        
        # Convert to binary predictions
        self.y_pred = (self.y_pred_proba > 0.5).astype(int).flatten()
        self.y_pred_proba = self.y_pred_proba.flatten()
        
        logger.info("✅ Predictions generated")
    
    def analyze_performance_metrics(self):
        """Analyze detailed performance metrics"""
        logger.info("📊 Analyzing Performance Metrics...")
        
        if self.y_pred is None:
            self.make_predictions()
        
        # Basic metrics
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)
        
        # ROC AUC
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall AUC
        precision_curve, recall_curve, _ = precision_recall_curve(self.y_test, self.y_pred_proba)
        pr_auc = average_precision_score(self.y_test, self.y_pred_proba)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        }
        
        logger.info("📈 Performance Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
        
        return metrics
    
    def analyze_confusion_matrix(self):
        """Analyze confusion matrix and classification errors"""
        logger.info("🔍 Analyzing Classification Errors...")
        
        if self.y_pred is None:
            self.make_predictions()
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Error analysis
        error_analysis = {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0
        }
        
        logger.info("🔍 Error Analysis:")
        logger.info(f"  True Negatives (Correct Benign): {tn}")
        logger.info(f"  False Positives (Benign as Attack): {fp}")
        logger.info(f"  False Negatives (Attack as Benign): {fn}")
        logger.info(f"  True Positives (Correct Attack): {tp}")
        logger.info(f"  False Positive Rate: {error_analysis['false_positive_rate']:.4f}")
        logger.info(f"  False Negative Rate: {error_analysis['false_negative_rate']:.4f}")
        
        return error_analysis
    
    def analyze_confidence_distribution(self):
        """Analyze prediction confidence distribution"""
        logger.info("📊 Analyzing Confidence Distribution...")
        
        if self.y_pred_proba is None:
            self.make_predictions()
        
        # Separate confidence scores by actual class
        benign_confidence = self.y_pred_proba[self.y_test == 0]
        attack_confidence = self.y_pred_proba[self.y_test == 1]
        
        confidence_analysis = {
            "benign_confidence": {
                "mean": float(np.mean(benign_confidence)),
                "std": float(np.std(benign_confidence)),
                "min": float(np.min(benign_confidence)),
                "max": float(np.max(benign_confidence))
            },
            "attack_confidence": {
                "mean": float(np.mean(attack_confidence)),
                "std": float(np.std(attack_confidence)),
                "min": float(np.min(attack_confidence)),
                "max": float(np.max(attack_confidence))
            }
        }
        
        logger.info("📊 Confidence Analysis:")
        logger.info(f"  Benign Traffic - Mean Confidence: {confidence_analysis['benign_confidence']['mean']:.4f}")
        logger.info(f"  Attack Traffic - Mean Confidence: {confidence_analysis['attack_confidence']['mean']:.4f}")
        
        return confidence_analysis
    
    def create_advanced_visualizations(self):
        """Create advanced visualization plots"""
        logger.info("📈 Creating Advanced Visualizations...")
        
        if self.y_pred is None:
            self.make_predictions()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Advanced DDoS Detection Model Analysis', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix Heatmap
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        axes[0,0].set_xticklabels(['Benign', 'Attack'])
        axes[0,0].set_yticklabels(['Benign', 'Attack'])
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        axes[0,1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0,1].set_xlim([0.0, 1.0])
        axes[0,1].set_ylim([0.0, 1.05])
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curve')
        axes[0,1].legend(loc="lower right")
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        precision_curve, recall_curve, _ = precision_recall_curve(self.y_test, self.y_pred_proba)
        pr_auc = average_precision_score(self.y_test, self.y_pred_proba)
        axes[0,2].plot(recall_curve, precision_curve, color='darkgreen', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
        axes[0,2].set_xlabel('Recall')
        axes[0,2].set_ylabel('Precision')
        axes[0,2].set_title('Precision-Recall Curve')
        axes[0,2].legend(loc="lower left")
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Confidence Distribution
        benign_confidence = self.y_pred_proba[self.y_test == 0]
        attack_confidence = self.y_pred_proba[self.y_test == 1]
        axes[1,0].hist(benign_confidence, bins=50, alpha=0.7, label='Benign', color='blue', density=True)
        axes[1,0].hist(attack_confidence, bins=50, alpha=0.7, label='Attack', color='red', density=True)
        axes[1,0].set_xlabel('Prediction Confidence')
        axes[1,0].set_ylabel('Density')
        axes[1,0].set_title('Confidence Distribution by Class')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Threshold Analysis
        thresholds = np.linspace(0, 1, 100)
        precisions = []
        recalls = []
        f1_scores = []
        
        for threshold in thresholds:
            y_pred_thresh = (self.y_pred_proba > threshold).astype(int)
            if len(np.unique(y_pred_thresh)) > 1:
                precisions.append(precision_score(self.y_test, y_pred_thresh))
                recalls.append(recall_score(self.y_test, y_pred_thresh))
                f1_scores.append(f1_score(self.y_test, y_pred_thresh))
            else:
                precisions.append(0)
                recalls.append(0)
                f1_scores.append(0)
        
        axes[1,1].plot(thresholds, precisions, label='Precision', linewidth=2)
        axes[1,1].plot(thresholds, recalls, label='Recall', linewidth=2)
        axes[1,1].plot(thresholds, f1_scores, label='F1-Score', linewidth=2)
        axes[1,1].axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Default Threshold')
        axes[1,1].set_xlabel('Threshold')
        axes[1,1].set_ylabel('Score')
        axes[1,1].set_title('Threshold Analysis')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Model Architecture Summary
        axes[1,2].axis('off')
        model_summary = []
        model_summary.append("Model Architecture Summary:")
        model_summary.append("-" * 30)
        model_summary.append(f"Total Params: {self.model.count_params():,}")
        model_summary.append(f"Input Shape: {self.model.input_shape}")
        model_summary.append(f"Output Shape: {self.model.output_shape}")
        model_summary.append(f"Layers: {len(self.model.layers)}")
        model_summary.append("")
        model_summary.append("Layer Details:")
        for i, layer in enumerate(self.model.layers[:6]):  # Show first 6 layers
            layer_info = f"{i+1}. {layer.__class__.__name__}"
            if hasattr(layer, 'units'):
                layer_info += f" ({layer.units} units)"
            elif hasattr(layer, 'filters'):
                layer_info += f" ({layer.filters} filters)"
            model_summary.append(layer_info)
        
        axes[1,2].text(0.1, 0.9, '\n'.join(model_summary), transform=axes[1,2].transAxes, 
                      fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # Save the visualization
        plt.tight_layout()
        plt.savefig('results/advanced_model_analysis.png', dpi=300, bbox_inches='tight')
        logger.info("📊 Advanced visualizations saved to: results/advanced_model_analysis.png")
        
        return fig
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report"""
        logger.info("📋 Generating Comprehensive Analysis Report...")
        
        # Perform all analyses
        performance_metrics = self.analyze_performance_metrics()
        error_analysis = self.analyze_confusion_matrix()
        confidence_analysis = self.analyze_confidence_distribution()
        
        # Create comprehensive report
        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "model_path": self.model_path,
            "test_data_path": self.data_path,
            "test_samples": int(len(self.y_test)),
            "performance_metrics": performance_metrics,
            "error_analysis": error_analysis,
            "confidence_analysis": confidence_analysis,
            "model_summary": {
                "total_parameters": int(self.model.count_params()),
                "trainable_parameters": int(sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])),
                "input_shape": str(self.model.input_shape),
                "output_shape": str(self.model.output_shape),
                "total_layers": len(self.model.layers)
            }
        }
        
        # Add recommendations
        report["recommendations"] = self.generate_recommendations(performance_metrics, error_analysis)
        
        # Save report
        with open("results/comprehensive_model_analysis.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("💾 Comprehensive report saved to: results/comprehensive_model_analysis.json")
        
        return report
    
    def generate_recommendations(self, performance_metrics: dict, error_analysis: dict) -> list:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Performance-based recommendations
        if performance_metrics["accuracy"] < 0.90:
            recommendations.append("Consider increasing model complexity or training duration for better accuracy")
        
        if performance_metrics["precision"] < 0.90:
            recommendations.append("High false positive rate - consider adjusting classification threshold or improving feature engineering")
        
        if performance_metrics["recall"] < 0.85:
            recommendations.append("High false negative rate - critical for security, consider ensemble methods or more attack samples")
        
        # Error analysis recommendations
        if error_analysis["false_positive_rate"] > 0.10:
            recommendations.append("Consider reducing false positives to minimize disruption to legitimate traffic")
        
        if error_analysis["false_negative_rate"] > 0.15:
            recommendations.append("High miss rate for attacks - consider additional training data or feature engineering")
        
        # General recommendations
        recommendations.append("Regular model retraining recommended as attack patterns evolve")
        recommendations.append("Consider implementing ensemble methods for improved robustness")
        recommendations.append("Monitor model performance in production and retrain when performance degrades")
        
        return recommendations

def main():
    """Main analysis function"""
    logger.info("🔬 ADVANCED MODEL ANALYSIS")
    logger.info("=" * 60)
    
    # Check if required files exist
    model_path = "results/balanced_centralized_model.h5"
    data_path = "data/optimized/balanced_dataset.csv"
    
    if not os.path.exists(model_path):
        logger.error(f"❌ Model not found: {model_path}")
        return
    
    if not os.path.exists(data_path):
        logger.error(f"❌ Test data not found: {data_path}")
        return
    
    try:
        # Initialize analyzer
        analyzer = ModelAnalyzer(model_path, data_path)
        
        # Generate comprehensive analysis
        report = analyzer.generate_comprehensive_report()
        
        # Create visualizations
        analyzer.create_advanced_visualizations()
        
        # Display summary
        logger.info("\n🎯 ANALYSIS SUMMARY:")
        logger.info("-" * 40)
        logger.info(f"Test Samples: {report['test_samples']:,}")
        logger.info(f"Model Parameters: {report['model_summary']['total_parameters']:,}")
        logger.info(f"Accuracy: {report['performance_metrics']['accuracy']:.4f}")
        logger.info(f"F1-Score: {report['performance_metrics']['f1_score']:.4f}")
        logger.info(f"ROC AUC: {report['performance_metrics']['roc_auc']:.4f}")
        
        logger.info("\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'], 1):
            logger.info(f"  {i}. {rec}")
        
        logger.info("\n✅ Analysis Complete!")
        logger.info("📁 Generated Files:")
        logger.info("  - results/comprehensive_model_analysis.json")
        logger.info("  - results/advanced_model_analysis.png")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
