#!/usr/bin/env python3
"""
Enhanced Federated Visualization Demo
Demonstrates the new comprehensive visualization system for federated learning
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.visualization.enhanced_federated_visualizer import generate_enhanced_federated_visualizations
from src.evaluation.metrics_calculator import MetricsCalculator
from src.evaluation.report_generator import ReportGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_federated_data(num_clients: int = 4, num_rounds: int = 5) -> tuple:
    """
    Create realistic mock federated learning data for demonstration
    
    Args:
        num_clients: Number of federated clients
        num_rounds: Number of communication rounds
        
    Returns:
        Tuple of (client_predictions, aggregated_predictions, federated_history)
    """
    logger.info(f"Creating mock federated data: {num_clients} clients, {num_rounds} rounds")
    
    np.random.seed(42)  # For reproducible results
    
    # Create federated training history
    federated_history = {
        'train_accuracy': [],
        'test_accuracy': [],
        'train_loss': [],
        'test_loss': []
    }
    
    # Simulate improving performance over rounds
    base_train_acc = 0.6
    base_test_acc = 0.4
    base_train_loss = 1.5
    base_test_loss = 2.0
    
    for round_num in range(num_rounds):
        # Simulate gradual improvement with some noise
        improvement = round_num * 0.05
        noise = np.random.normal(0, 0.02)
        
        train_acc = min(0.95, base_train_acc + improvement + noise)
        test_acc = min(0.9, base_test_acc + improvement + noise * 0.5)
        train_loss = max(0.1, base_train_loss - improvement * 2 + abs(noise))
        test_loss = max(0.2, base_test_loss - improvement * 1.5 + abs(noise) * 1.5)
        
        federated_history['train_accuracy'].append(train_acc)
        federated_history['test_accuracy'].append(test_acc)
        federated_history['train_loss'].append(train_loss)
        federated_history['test_loss'].append(test_loss)
    
    # Create client predictions
    client_predictions = {}
    all_y_true = []
    all_y_pred = []
    all_y_pred_proba = []
    
    for client_id in range(num_clients):
        # Generate realistic prediction data for each client
        n_samples = np.random.randint(800, 1200)  # Variable client data sizes
        
        # Create imbalanced dataset (more benign than attack)
        n_benign = int(n_samples * np.random.uniform(0.6, 0.8))
        n_attack = n_samples - n_benign
        
        y_true = np.concatenate([
            np.zeros(n_benign),  # Benign
            np.ones(n_attack)    # Attack
        ])
        
        # Simulate model predictions with realistic performance
        client_accuracy = federated_history['test_accuracy'][-1] + np.random.normal(0, 0.05)
        client_accuracy = np.clip(client_accuracy, 0.3, 0.95)
        
        # Generate predictions based on accuracy
        y_pred = y_true.copy()
        n_errors = int(len(y_true) * (1 - client_accuracy))
        error_indices = np.random.choice(len(y_true), n_errors, replace=False)
        y_pred[error_indices] = 1 - y_pred[error_indices]  # Flip labels for errors
        
        # Generate prediction probabilities
        y_pred_proba = np.random.beta(2, 2, size=len(y_true))  # Base probabilities
        
        # Adjust probabilities based on true labels and predictions
        for i in range(len(y_true)):
            if y_true[i] == y_pred[i]:  # Correct prediction
                if y_true[i] == 1:  # True attack, predicted attack
                    y_pred_proba[i] = np.random.uniform(0.6, 0.95)
                else:  # True benign, predicted benign
                    y_pred_proba[i] = np.random.uniform(0.05, 0.4)
            else:  # Incorrect prediction
                if y_true[i] == 1:  # True attack, predicted benign
                    y_pred_proba[i] = np.random.uniform(0.05, 0.4)
                else:  # True benign, predicted attack
                    y_pred_proba[i] = np.random.uniform(0.6, 0.95)
        
        client_predictions[f"client_{client_id}"] = {
            'y_true': y_true.astype(int),
            'y_pred': y_pred.astype(int),
            'y_pred_proba': y_pred_proba
        }
        
        # Collect for aggregated predictions
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_y_pred_proba.extend(y_pred_proba)
    
    # Create aggregated predictions
    aggregated_predictions = {
        'y_true': np.array(all_y_true, dtype=int),
        'y_pred': np.array(all_y_pred, dtype=int),
        'y_pred_proba': np.array(all_y_pred_proba)
    }
    
    logger.info(f"Generated mock data: {len(client_predictions)} clients, "
                f"{len(all_y_true)} total samples")
    
    return client_predictions, aggregated_predictions, federated_history

def calculate_all_metrics(client_predictions: dict, aggregated_predictions: dict) -> tuple:
    """Calculate metrics for all clients and aggregated data"""
    logger.info("Calculating comprehensive metrics for all clients")
    
    metrics_calc = MetricsCalculator()
    
    # Calculate client metrics
    client_metrics = {}
    for client_id, pred_data in client_predictions.items():
        metrics = metrics_calc.calculate_classification_metrics(
            pred_data['y_true'], pred_data['y_pred'], pred_data['y_pred_proba'])
        client_metrics[client_id] = metrics
        logger.info(f"{client_id}: Accuracy={metrics['accuracy']:.4f}, "
                   f"F1={metrics['f1_score']:.4f}, AUC={metrics['roc_auc']:.4f}")
    
    # Calculate aggregated metrics
    aggregated_metrics = metrics_calc.calculate_classification_metrics(
        aggregated_predictions['y_true'], aggregated_predictions['y_pred'], 
        aggregated_predictions['y_pred_proba'])
    
    # Calculate cross-client aggregated metrics
    cross_client_metrics = metrics_calc.aggregate_client_metrics(client_metrics)
    
    logger.info(f"Aggregated: Accuracy={aggregated_metrics['accuracy']:.4f}, "
               f"Mean Client Accuracy={cross_client_metrics['accuracy_mean']:.4f}")
    
    return client_metrics, aggregated_metrics, cross_client_metrics

def demonstrate_enhanced_visualizations():
    """Main demonstration function"""
    logger.info("🚀 Starting Enhanced Federated Visualization Demo")
    logger.info("=" * 70)
    
    try:
        # 1. Create mock federated learning data
        logger.info("📊 Step 1: Creating mock federated learning data...")
        client_predictions, aggregated_predictions, federated_history = create_mock_federated_data(
            num_clients=4, num_rounds=5)
        
        # 2. Calculate comprehensive metrics
        logger.info("🧮 Step 2: Calculating comprehensive metrics...")
        client_metrics, agg_metrics, cross_client_metrics = calculate_all_metrics(
            client_predictions, aggregated_predictions)
        
        # 3. Generate all enhanced visualizations
        logger.info("🎨 Step 3: Generating enhanced visualizations...")
        generated_files = generate_enhanced_federated_visualizations(
            federated_history=federated_history,
            client_predictions=client_predictions,
            global_predictions=aggregated_predictions,
            output_dir="results/validations"
        )
        
        # 4. Generate comprehensive reports
        logger.info("📄 Step 4: Generating comprehensive reports...")
        report_gen = ReportGenerator()
        
        # Get the output directory from generated files
        if generated_files:
            sample_path = list(generated_files.values())[0]
            output_base = os.path.dirname(os.path.dirname(sample_path))
            
            # Save comprehensive report
            report_path = report_gen.save_comprehensive_report(
                output_base, cross_client_metrics, client_metrics, generated_files)
            
            # Organize output files
            organized_files = report_gen.organize_output_files(output_base, generated_files)
            
            logger.info(f"📁 Comprehensive report saved: {report_path}")
            logger.info(f"📋 Summary report saved: {organized_files.get('summary_report', 'N/A')}")
        
        # 5. Display summary
        logger.info("=" * 70)
        logger.info("✅ ENHANCED VISUALIZATION DEMO COMPLETE!")
        logger.info("=" * 70)
        
        print("\n" + "=" * 70)
        print("🎯 FEDERATED LEARNING VISUALIZATION SUMMARY")
        print("=" * 70)
        
        print(f"📊 Generated {len(generated_files)} visualization files:")
        
        # Group and display files by type
        file_types = {}
        for key, path in generated_files.items():
            file_type = key.split('_')[0] if '_' in key else key
            if file_type not in file_types:
                file_types[file_type] = []
            file_types[file_type].append(os.path.basename(path))
        
        for file_type, files in file_types.items():
            print(f"\n{file_type.replace('_', ' ').title()}:")
            for file in files:
                print(f"  • {file}")
        
        print(f"\n📈 Performance Summary:")
        print(f"  • Mean Client Accuracy: {cross_client_metrics['accuracy_mean']:.4f} ± {cross_client_metrics['accuracy_std']:.4f}")
        print(f"  • Mean Client F1-Score: {cross_client_metrics['f1_score_mean']:.4f} ± {cross_client_metrics['f1_score_std']:.4f}")
        print(f"  • Mean Client ROC-AUC: {cross_client_metrics['roc_auc_mean']:.4f} ± {cross_client_metrics['roc_auc_std']:.4f}")
        print(f"  • Number of Clients: {cross_client_metrics['num_clients']}")
        
        print(f"\n📁 All files saved in: results/validations/")
        print("=" * 70)
        
        return generated_files
        
    except Exception as e:
        logger.error(f"❌ Error in demonstration: {e}")
        raise

def integrate_with_existing_pipeline():
    """Show how to integrate with existing federated training pipeline"""
    logger.info("🔗 Integration Example with Existing Pipeline")
    
    integration_code = '''
# Example integration with federated_training.py

def run_federated_simulation_with_enhanced_viz(num_rounds: int = 10, num_clients: int = 4):
    """Enhanced version of federated training with comprehensive visualizations"""
    
    # ... existing federated training code ...
    
    history = fl.simulation.start_simulation(
        client_fn=lambda cid: client_fns[cid](),
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.25},
    )
    
    # NEW: Extract client predictions after training
    client_predictions = {}
    for client_id in range(num_clients):
        X_train, y_train, X_test, y_test = load_client_data(client_id)
        
        # Get final model predictions
        final_model = get_global_model()  # Implement this function
        y_pred_proba = final_model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        client_predictions[f"client_{client_id}"] = {
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    # NEW: Create aggregated predictions
    all_y_true = np.concatenate([cp['y_true'] for cp in client_predictions.values()])
    all_y_pred = np.concatenate([cp['y_pred'] for cp in client_predictions.values()])
    all_y_pred_proba = np.concatenate([cp['y_pred_proba'] for cp in client_predictions.values()])
    
    aggregated_predictions = {
        'y_true': all_y_true,
        'y_pred': all_y_pred,
        'y_pred_proba': all_y_pred_proba
    }
    
    # NEW: Convert history to required format
    federated_history = {
        'train_accuracy': extract_train_accuracy_from_history(history),
        'test_accuracy': extract_test_accuracy_from_history(history),
        'train_loss': extract_train_loss_from_history(history),
        'test_loss': extract_test_loss_from_history(history)
    }
    
    # NEW: Generate enhanced visualizations
    from src.visualization.enhanced_federated_visualizer import generate_enhanced_federated_visualizations
    
    generated_files = generate_enhanced_federated_visualizations(
        federated_history=federated_history,
        client_predictions=client_predictions,
        global_predictions=aggregated_predictions,
        output_dir="results/validations"
    )
    
    logger.info(f"Generated {len(generated_files)} enhanced visualization files")
    
    return history, generated_files
'''
    
    print("\n" + "=" * 70)
    print("🔗 INTEGRATION WITH EXISTING PIPELINE")
    print("=" * 70)
    print(integration_code)
    print("=" * 70)

if __name__ == "__main__":
    try:
        # Run the demonstration
        generated_files = demonstrate_enhanced_visualizations()
        
        # Show integration example
        integrate_with_existing_pipeline()
        
        logger.info("🎉 Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        sys.exit(1)