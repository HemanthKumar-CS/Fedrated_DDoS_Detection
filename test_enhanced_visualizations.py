#!/usr/bin/env python3
"""
Comprehensive test suite for enhanced federated visualizations
Tests all components with known test datasets and validates outputs
"""

import os
import sys
import unittest
import numpy as np
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.evaluation.metrics_calculator import MetricsCalculator
from src.visualization.evaluation_visualizer import EvaluationVisualizer
from src.visualization.federated_visualizer import FederatedVisualizer
from src.visualization.enhanced_federated_visualizer import EnhancedFederatedVisualizer
from src.evaluation.report_generator import ReportGenerator
from src.utils.visualization_utils import (
    create_output_directory, format_classification_table, 
    apply_consistent_styling, save_plot_with_metadata
)

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests

class TestDataGenerator:
    """Generate test data for federated learning scenarios"""
    
    @staticmethod
    def create_binary_classification_data(n_samples: int = 1000, random_state: int = 42) -> tuple:
        """Create binary classification test data"""
        np.random.seed(random_state)
        
        # Create imbalanced dataset (70% benign, 30% attack)
        n_benign = int(n_samples * 0.7)
        n_attack = n_samples - n_benign
        
        y_true = np.concatenate([np.zeros(n_benign), np.ones(n_attack)])
        
        # Generate predictions with realistic accuracy (~85%)
        y_pred = y_true.copy()
        n_errors = int(n_samples * 0.15)
        error_indices = np.random.choice(n_samples, n_errors, replace=False)
        y_pred[error_indices] = 1 - y_pred[error_indices]
        
        # Generate realistic probabilities
        y_pred_proba = np.random.beta(2, 2, size=n_samples)
        for i in range(n_samples):
            if y_true[i] == y_pred[i]:  # Correct prediction
                if y_true[i] == 1:  # Attack correctly predicted
                    y_pred_proba[i] = np.random.uniform(0.6, 0.95)
                else:  # Benign correctly predicted
                    y_pred_proba[i] = np.random.uniform(0.05, 0.4)
            else:  # Incorrect prediction
                if y_true[i] == 1:  # Attack incorrectly predicted as benign
                    y_pred_proba[i] = np.random.uniform(0.05, 0.4)
                else:  # Benign incorrectly predicted as attack
                    y_pred_proba[i] = np.random.uniform(0.6, 0.95)
        
        return y_true.astype(int), y_pred.astype(int), y_pred_proba
    
    @staticmethod
    def create_federated_client_data(num_clients: int = 4, samples_per_client: int = 500) -> dict:
        """Create federated client prediction data"""
        client_predictions = {}
        
        for client_id in range(num_clients):
            y_true, y_pred, y_pred_proba = TestDataGenerator.create_binary_classification_data(
                samples_per_client, random_state=42 + client_id)
            
            client_predictions[f"client_{client_id}"] = {
                'y_true': y_true,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        
        return client_predictions
    
    @staticmethod
    def create_federated_history(num_rounds: int = 5) -> dict:
        """Create mock federated training history"""
        np.random.seed(42)
        
        federated_history = {
            'train_accuracy': [],
            'test_accuracy': [],
            'train_loss': [],
            'test_loss': []
        }
        
        for round_num in range(num_rounds):
            # Simulate improving performance
            base_acc = 0.6 + round_num * 0.05
            base_loss = 1.5 - round_num * 0.2
            
            train_acc = base_acc + np.random.normal(0, 0.02)
            test_acc = base_acc - 0.05 + np.random.normal(0, 0.02)
            train_loss = base_loss + np.random.normal(0, 0.1)
            test_loss = base_loss + 0.2 + np.random.normal(0, 0.1)
            
            federated_history['train_accuracy'].append(np.clip(train_acc, 0, 1))
            federated_history['test_accuracy'].append(np.clip(test_acc, 0, 1))
            federated_history['train_loss'].append(max(0.1, train_loss))
            federated_history['test_loss'].append(max(0.1, test_loss))
        
        return federated_history


class TestMetricsCalculator(unittest.TestCase):
    """Test MetricsCalculator class with known test datasets"""
    
    def setUp(self):
        self.metrics_calc = MetricsCalculator()
        self.y_true, self.y_pred, self.y_pred_proba = TestDataGenerator.create_binary_classification_data()
    
    def test_calculate_classification_metrics(self):
        """Test classification metrics calculation with known data"""
        metrics = self.metrics_calc.calculate_classification_metrics(
            self.y_true, self.y_pred, self.y_pred_proba)
        
        # Check that all required metrics are present
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'confusion_matrix']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
        
        # Check metric ranges
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)
        self.assertGreaterEqual(metrics['roc_auc'], 0)
        self.assertLessEqual(metrics['roc_auc'], 1)
        
        # Check confusion matrix structure
        cm = metrics['confusion_matrix']
        self.assertEqual(len(cm), 2)
        self.assertEqual(len(cm[0]), 2)
        self.assertGreater(sum(sum(row) for row in cm), 0)
    
    def test_calculate_roc_metrics(self):
        """Test ROC metrics calculation"""
        roc_metrics = self.metrics_calc.calculate_roc_metrics(self.y_true, self.y_pred_proba)
        
        required_keys = ['roc_auc', 'fpr', 'tpr', 'roc_thresholds']
        for key in required_keys:
            self.assertIn(key, roc_metrics)
        
        self.assertGreaterEqual(roc_metrics['roc_auc'], 0)
        self.assertLessEqual(roc_metrics['roc_auc'], 1)
        self.assertIsInstance(roc_metrics['fpr'], list)
        self.assertIsInstance(roc_metrics['tpr'], list)
    
    def test_calculate_precision_recall_metrics(self):
        """Test precision-recall metrics calculation"""
        pr_metrics = self.metrics_calc.calculate_precision_recall_metrics(
            self.y_true, self.y_pred_proba)
        
        required_keys = ['average_precision', 'precision_curve', 'recall_curve', 'pr_thresholds']
        for key in required_keys:
            self.assertIn(key, pr_metrics)
        
        self.assertGreaterEqual(pr_metrics['average_precision'], 0)
        self.assertLessEqual(pr_metrics['average_precision'], 1)
    
    def test_aggregate_client_metrics(self):
        """Test client metrics aggregation"""
        client_data = TestDataGenerator.create_federated_client_data(num_clients=3)
        client_metrics = {}
        
        for client_id, pred_data in client_data.items():
            metrics = self.metrics_calc.calculate_classification_metrics(
                pred_data['y_true'], pred_data['y_pred'], pred_data['y_pred_proba'])
            client_metrics[client_id] = metrics
        
        aggregated = self.metrics_calc.aggregate_client_metrics(client_metrics)
        
        # Check aggregated metrics structure
        required_keys = ['num_clients', 'accuracy_mean', 'accuracy_std', 'aggregated_confusion_matrix']
        for key in required_keys:
            self.assertIn(key, aggregated)
        
        self.assertEqual(aggregated['num_clients'], 3)
        self.assertGreaterEqual(aggregated['accuracy_mean'], 0)
        self.assertLessEqual(aggregated['accuracy_mean'], 1)
    
    def test_validate_prediction_data(self):
        """Test prediction data validation"""
        # Test valid data
        is_valid, msg = self.metrics_calc.validate_prediction_data(
            self.y_true, self.y_pred, self.y_pred_proba)
        self.assertTrue(is_valid)
        
        # Test invalid data - shape mismatch
        is_valid, msg = self.metrics_calc.validate_prediction_data(
            self.y_true[:-10], self.y_pred, self.y_pred_proba)
        self.assertFalse(is_valid)
        
        # Test invalid data - wrong value range
        invalid_y_true = np.array([0, 1, 2, 1, 0])  # Contains invalid value 2
        invalid_y_pred = np.array([0, 1, 1, 1, 0])
        is_valid, msg = self.metrics_calc.validate_prediction_data(
            invalid_y_true, invalid_y_pred)
        self.assertFalse(is_valid)


class TestVisualizationComponents(unittest.TestCase):
    """Test visualization components"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.eval_viz = EvaluationVisualizer()
        self.fed_viz = FederatedVisualizer()
        self.y_true, self.y_pred, self.y_pred_proba = TestDataGenerator.create_binary_classification_data()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_evaluation_visualizer_confusion_matrix(self):
        """Test confusion matrix generation"""
        output_path = os.path.join(self.temp_dir, "test_confusion_matrix.png")
        
        saved_path = self.eval_viz.plot_confusion_matrix(
            self.y_true, self.y_pred, "Test Confusion Matrix", output_path)
        
        self.assertTrue(os.path.exists(saved_path))
        self.assertTrue(saved_path.endswith('.png'))
    
    def test_evaluation_visualizer_roc_curve(self):
        """Test ROC curve generation"""
        output_path = os.path.join(self.temp_dir, "test_roc_curve.png")
        
        saved_path = self.eval_viz.plot_roc_curve(
            self.y_true, self.y_pred_proba, "Test ROC Curve", output_path)
        
        self.assertTrue(os.path.exists(saved_path))
        self.assertTrue(saved_path.endswith('.png'))
    
    def test_evaluation_visualizer_precision_recall_curve(self):
        """Test precision-recall curve generation"""
        output_path = os.path.join(self.temp_dir, "test_pr_curve.png")
        
        saved_path = self.eval_viz.plot_precision_recall_curve(
            self.y_true, self.y_pred_proba, "Test PR Curve", output_path)
        
        self.assertTrue(os.path.exists(saved_path))
        self.assertTrue(saved_path.endswith('.png'))
    
    def test_classification_report_generation(self):
        """Test classification report generation"""
        report = self.eval_viz.print_classification_report(
            self.y_true, self.y_pred, title="Test Classification Report")
        
        self.assertIsInstance(report, str)
        self.assertIn("Classification Report", report)
        self.assertIn("Precision", report)
        self.assertIn("Recall", report)
        self.assertIn("F1-Score", report)
    
    def test_federated_visualizer_training_plots(self):
        """Test federated training progress plots"""
        federated_history = TestDataGenerator.create_federated_history()
        
        # Test accuracy plot
        output_path = os.path.join(self.temp_dir, "test_training_accuracy.png")
        saved_path = self.fed_viz.plot_training_vs_test_accuracy(
            federated_history, output_path)
        
        self.assertTrue(os.path.exists(saved_path))
        
        # Test loss plot
        output_path = os.path.join(self.temp_dir, "test_training_loss.png")
        saved_path = self.fed_viz.plot_training_vs_test_loss(
            federated_history, output_path)
        
        self.assertTrue(os.path.exists(saved_path))
    
    def test_client_performance_comparison(self):
        """Test client performance comparison visualization"""
        client_data = TestDataGenerator.create_federated_client_data(num_clients=3)
        metrics_calc = MetricsCalculator()
        
        client_metrics = {}
        for client_id, pred_data in client_data.items():
            metrics = metrics_calc.calculate_classification_metrics(
                pred_data['y_true'], pred_data['y_pred'], pred_data['y_pred_proba'])
            client_metrics[client_id] = metrics
        
        output_path = os.path.join(self.temp_dir, "test_client_comparison.png")
        saved_path = self.fed_viz.plot_client_performance_comparison(
            client_metrics, output_path)
        
        self.assertTrue(os.path.exists(saved_path))


class TestEnhancedFederatedVisualizer(unittest.TestCase):
    """Test the main enhanced federated visualizer"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.enhanced_viz = EnhancedFederatedVisualizer()
        self.client_predictions = TestDataGenerator.create_federated_client_data()
        self.federated_history = TestDataGenerator.create_federated_history()
        
        # Create aggregated predictions
        all_y_true = []
        all_y_pred = []
        all_y_pred_proba = []
        
        for pred_data in self.client_predictions.values():
            all_y_true.extend(pred_data['y_true'])
            all_y_pred.extend(pred_data['y_pred'])
            all_y_pred_proba.extend(pred_data['y_pred_proba'])
        
        self.aggregated_predictions = {
            'y_true': np.array(all_y_true),
            'y_pred': np.array(all_y_pred),
            'y_pred_proba': np.array(all_y_pred_proba)
        }
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_generate_all_visualizations(self):
        """Test complete visualization generation pipeline"""
        generated_files = self.enhanced_viz.generate_all_visualizations(
            self.client_predictions,
            self.aggregated_predictions,
            self.federated_history,
            self.temp_dir
        )
        
        # Check that files were generated
        self.assertGreater(len(generated_files), 0)
        
        # Check that all files exist
        for viz_type, filepath in generated_files.items():
            self.assertTrue(os.path.exists(filepath), f"File not found: {filepath}")
        
        # Check for expected visualization types
        expected_types = ['confusion_matrix', 'roc_curve', 'pr_curve', 'training']
        for expected_type in expected_types:
            found = any(expected_type in key for key in generated_files.keys())
            self.assertTrue(found, f"Expected visualization type not found: {expected_type}")
    
    def test_validation_methods(self):
        """Test data validation methods"""
        # Test valid data
        is_valid = self.enhanced_viz._validate_prediction_data(
            self.client_predictions['client_0'])
        self.assertTrue(is_valid)
        
        # Test invalid data
        invalid_data = {'y_true': np.array([1, 0]), 'y_pred': np.array([1])}  # Shape mismatch
        is_valid = self.enhanced_viz._validate_prediction_data(invalid_data)
        self.assertFalse(is_valid)


class TestReportGenerator(unittest.TestCase):
    """Test report generation functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.report_gen = ReportGenerator()
        self.client_data = TestDataGenerator.create_federated_client_data()
        
        # Calculate metrics for testing
        metrics_calc = MetricsCalculator()
        self.client_metrics = {}
        for client_id, pred_data in self.client_data.items():
            metrics = metrics_calc.calculate_classification_metrics(
                pred_data['y_true'], pred_data['y_pred'], pred_data['y_pred_proba'])
            self.client_metrics[client_id] = metrics
        
        self.aggregated_metrics = metrics_calc.aggregate_client_metrics(self.client_metrics)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_generate_client_report(self):
        """Test client-specific report generation"""
        client_id = 'client_0'
        metrics = self.client_metrics[client_id]
        visualizations = {'test_viz': 'test_path.png'}
        
        report = self.report_gen.generate_client_report(client_id, metrics, visualizations)
        
        self.assertIsInstance(report, str)
        self.assertIn(client_id, report)
        self.assertIn("PERFORMANCE METRICS", report)
        self.assertIn("Accuracy", report)
    
    def test_generate_aggregated_report(self):
        """Test aggregated report generation"""
        visualizations = {'test_viz': 'test_path.png'}
        
        report = self.report_gen.generate_aggregated_report(
            self.aggregated_metrics, visualizations, self.client_metrics)
        
        self.assertIsInstance(report, str)
        self.assertIn("FEDERATED LEARNING EVALUATION REPORT", report)
        self.assertIn("INDIVIDUAL CLIENT PERFORMANCE", report)
        self.assertIn("Mean Accuracy", report)
    
    def test_save_comprehensive_report(self):
        """Test comprehensive report saving"""
        visualizations = {'test_viz': 'test_path.png'}
        
        report_path = self.report_gen.save_comprehensive_report(
            self.temp_dir, self.aggregated_metrics, self.client_metrics, visualizations)
        
        self.assertTrue(os.path.exists(report_path))
        self.assertTrue(report_path.endswith('.txt'))
        
        # Check file content
        with open(report_path, 'r') as f:
            content = f.read()
            self.assertIn("FEDERATED LEARNING EVALUATION REPORT", content)
    
    def test_organize_output_files(self):
        """Test output file organization"""
        generated_files = {
            'confusion_matrix_client_0': 'path1.png',
            'roc_curve_client_0': 'path2.png',
            'training_progress': 'path3.png'
        }
        
        organized = self.report_gen.organize_output_files(self.temp_dir, generated_files)
        
        self.assertIn('summary_report', organized)
        self.assertIn('metadata', organized)
        
        # Check that summary report exists
        summary_path = organized['summary_report']
        self.assertTrue(os.path.exists(summary_path))


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_output_directory(self):
        """Test output directory creation"""
        output_dir = create_output_directory(self.temp_dir, timestamp=False)
        
        self.assertTrue(os.path.exists(output_dir))
        
        # Check subdirectories
        expected_subdirs = [
            "confusion_matrices", "classification_reports", 
            "roc_curves", "precision_recall_curves", "training_progress"
        ]
        
        for subdir in expected_subdirs:
            subdir_path = os.path.join(output_dir, subdir)
            self.assertTrue(os.path.exists(subdir_path))
    
    def test_format_classification_table(self):
        """Test classification report table formatting"""
        # Mock classification report dictionary
        report_dict = {
            'Benign': {'precision': 0.85, 'recall': 0.90, 'f1-score': 0.87, 'support': 700},
            'Attack': {'precision': 0.80, 'recall': 0.75, 'f1-score': 0.77, 'support': 300},
            'accuracy': 0.85,
            'macro avg': {'precision': 0.825, 'recall': 0.825, 'f1-score': 0.82, 'support': 1000},
            'weighted avg': {'precision': 0.84, 'recall': 0.85, 'f1-score': 0.84, 'support': 1000}
        }
        
        formatted_table = format_classification_table(report_dict)
        
        self.assertIsInstance(formatted_table, str)
        self.assertIn("CLASSIFICATION REPORT", formatted_table)
        self.assertIn("Benign", formatted_table)
        self.assertIn("Attack", formatted_table)
        self.assertIn("Precision", formatted_table)


def run_integration_test():
    """Run integration test with complete pipeline"""
    print("\n" + "="*70)
    print("🧪 RUNNING INTEGRATION TEST")
    print("="*70)
    
    try:
        # Create test data
        client_predictions = TestDataGenerator.create_federated_client_data(num_clients=2, samples_per_client=200)
        federated_history = TestDataGenerator.create_federated_history(num_rounds=3)
        
        # Create aggregated predictions
        all_y_true = []
        all_y_pred = []
        all_y_pred_proba = []
        
        for pred_data in client_predictions.values():
            all_y_true.extend(pred_data['y_true'])
            all_y_pred.extend(pred_data['y_pred'])
            all_y_pred_proba.extend(pred_data['y_pred_proba'])
        
        aggregated_predictions = {
            'y_true': np.array(all_y_true),
            'y_pred': np.array(all_y_pred),
            'y_pred_proba': np.array(all_y_pred_proba)
        }
        
        # Run complete pipeline
        from src.visualization.enhanced_federated_visualizer import generate_enhanced_federated_visualizations
        
        generated_files = generate_enhanced_federated_visualizations(
            federated_history=federated_history,
            client_predictions=client_predictions,
            global_predictions=aggregated_predictions,
            output_dir="test_results"
        )
        
        print(f"✅ Integration test passed: {len(generated_files)} files generated")
        
        # Clean up test results
        import shutil
        if os.path.exists("test_results"):
            shutil.rmtree("test_results")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Starting Enhanced Federated Visualization Test Suite")
    print("="*70)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration test
    integration_success = run_integration_test()
    
    print("\n" + "="*70)
    print("🎯 TEST SUITE SUMMARY")
    print("="*70)
    print(f"Integration Test: {'✅ PASSED' if integration_success else '❌ FAILED'}")
    print("="*70)