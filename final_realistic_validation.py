import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_realistic_data():
    """Load the challenging realistic dataset we created"""
    try:
        logger.info("📂 Loading challenging realistic dataset...")
        data = pd.read_csv('data/optimized/challenging_realistic_dataset.csv')
        logger.info(f"✅ Loaded challenging realistic dataset: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"❌ Error loading challenging realistic dataset: {e}")
        # Fallback to original realistic dataset
        try:
            logger.info("🔄 Falling back to original realistic dataset...")
            data = pd.read_csv('data/optimized/realistic_balanced_dataset.csv')
            logger.info(f"✅ Loaded original realistic dataset: {data.shape}")
            return data
        except Exception as e2:
            logger.error(f"❌ Error loading any realistic dataset: {e2}")
            return None


def comprehensive_model_test(X_train, X_test, y_train, y_test):
    """Test our trained model and multiple baselines on realistic data"""

    # Scale the data
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # Load our trained model
    try:
        logger.info("🤖 Loading our trained enhanced model...")
        model = tf.keras.models.load_model('results/best_enhanced_model.keras')

        # Predict with our model
        y_pred_proba = model.predict(X_test_scaled, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # Calculate metrics for our model
        results['Enhanced_CNN'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        logger.info(
            f"✅ Enhanced CNN - Accuracy: {results['Enhanced_CNN']['accuracy']:.4f}")

    except Exception as e:
        logger.error(f"❌ Error loading enhanced model: {e}")
        results['Enhanced_CNN'] = None

    # Test baseline algorithms
    algorithms = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True)
    }

    for name, alg in algorithms.items():
        logger.info(f"🧪 Testing {name}...")
        try:
            alg.fit(X_train_scaled, y_train)
            y_pred = alg.predict(X_test_scaled)
            y_pred_proba = alg.predict_proba(X_test_scaled)[:, 1] if hasattr(
                alg, 'predict_proba') else y_pred

            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted'),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            logger.info(
                f"✅ {name} - Accuracy: {results[name]['accuracy']:.4f}")

        except Exception as e:
            logger.error(f"❌ Error with {name}: {e}")
            results[name] = None

    return results


def analyze_data_complexity(X, y):
    """Analyze the complexity of the realistic dataset"""
    logger.info("🔬 Analyzing data complexity...")

    # Check linear separability
    lr = LogisticRegression(random_state=42, max_iter=1000)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validation scores
    cv_scores = cross_val_score(lr, X_scaled, y, cv=5, scoring='accuracy')

    # Feature correlations
    correlations = np.corrcoef(X.T)
    max_correlation = np.max(
        np.abs(correlations[np.triu_indices_from(correlations, k=1)]))

    complexity = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'max_feature_correlation': max_correlation,
        'separability_assessment': 'HIGH' if cv_scores.mean() > 0.95 else 'MODERATE' if cv_scores.mean() > 0.8 else 'LOW'
    }

    logger.info(f"📊 Complexity Analysis:")
    logger.info(
        f"   CV Mean: {complexity['cv_mean']:.4f} ± {complexity['cv_std']:.4f}")
    logger.info(
        f"   Max Feature Correlation: {complexity['max_feature_correlation']:.4f}")
    logger.info(f"   Separability: {complexity['separability_assessment']}")

    return complexity


def create_performance_visualization(results, save_path):
    """Create comprehensive performance visualization"""
    logger.info("📊 Creating performance visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Realistic Dataset Model Performance Analysis',
                 fontsize=16, fontweight='bold')

    # Extract data for visualization
    algorithms = []
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    data = {metric: [] for metric in metrics}

    for alg_name, alg_results in results.items():
        if alg_results is not None:
            algorithms.append(alg_name)
            for metric in metrics:
                data[metric].append(alg_results[metric])

    # Plot 1: Accuracy comparison
    axes[0, 0].bar(algorithms, data['accuracy'], color=[
                   '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[0, 0].set_title('Model Accuracy Comparison', fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for i, v in enumerate(data['accuracy']):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

    # Plot 2: All metrics comparison
    x = np.arange(len(algorithms))
    width = 0.15

    for i, metric in enumerate(['precision', 'recall', 'f1', 'roc_auc']):
        axes[0, 1].bar(x + i*width, data[metric], width,
                       label=metric.replace('_', ' ').title())

    axes[0, 1].set_title('All Metrics Comparison', fontweight='bold')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_xticks(x + width * 1.5)
    axes[0, 1].set_xticklabels(algorithms, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].set_ylim(0, 1)

    # Plot 3: Performance heatmap
    heatmap_data = pd.DataFrame(data, index=algorithms)
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlBu_r',
                ax=axes[1, 0], cbar_kws={'label': 'Score'})
    axes[1, 0].set_title('Performance Heatmap', fontweight='bold')

    # Plot 4: Performance distribution
    all_scores = []
    for metric in metrics:
        all_scores.extend(data[metric])

    axes[1, 1].hist(all_scores, bins=20, alpha=0.7,
                    color='#45B7D1', edgecolor='black')
    axes[1, 1].axvline(np.mean(all_scores), color='red', linestyle='--',
                       label=f'Mean: {np.mean(all_scores):.3f}')
    axes[1, 1].set_title('Score Distribution Analysis', fontweight='bold')
    axes[1, 1].set_xlabel('Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✅ Visualization saved to: {save_path}")


def generate_final_report(results, complexity, data_shape):
    """Generate final comprehensive validation report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Calculate summary statistics
    valid_results = {k: v for k, v in results.items() if v is not None}

    if valid_results:
        avg_accuracy = np.mean([r['accuracy'] for r in valid_results.values()])
        std_accuracy = np.std([r['accuracy'] for r in valid_results.values()])
        best_model = max(valid_results.keys(),
                         key=lambda k: valid_results[k]['accuracy'])
        best_accuracy = valid_results[best_model]['accuracy']
    else:
        avg_accuracy = std_accuracy = best_accuracy = 0
        best_model = "None"

    # Determine if results are realistic
    is_realistic = avg_accuracy < 0.95 and std_accuracy > 0.01

    report = f"""
# Final Realistic Model Validation Report
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📊 Dataset Information
- **Total Samples:** {data_shape[0]:,}
- **Features:** {data_shape[1]}
- **Data Source:** Realistic synthetic dataset with noise and complexity

## 🤖 Model Performance Summary
- **Average Accuracy:** {avg_accuracy:.4f} ± {std_accuracy:.4f}
- **Best Performing Model:** {best_model} ({best_accuracy:.4f})
- **Performance Variance:** {std_accuracy:.4f}

## 🔬 Data Complexity Assessment
- **Cross-Validation Score:** {complexity['cv_mean']:.4f} ± {complexity['cv_std']:.4f}
- **Separability:** {complexity['separability_assessment']}
- **Max Feature Correlation:** {complexity['max_feature_correlation']:.4f}

## 📈 Individual Model Results
"""

    for model_name, model_results in results.items():
        if model_results is not None:
            report += f"""
### {model_name}
- **Accuracy:** {model_results['accuracy']:.4f}
- **Precision:** {model_results['precision']:.4f}
- **Recall:** {model_results['recall']:.4f}
- **F1-Score:** {model_results['f1']:.4f}
- **ROC-AUC:** {model_results['roc_auc']:.4f}
"""
        else:
            report += f"""
### {model_name}
- **Status:** ❌ Failed to evaluate
"""

    report += f"""
## 🎯 Validation Assessment

{'✅ **REALISTIC PERFORMANCE DETECTED**' if is_realistic else '⚠️ **POTENTIALLY UNREALISTIC PERFORMANCE**'}

### Key Findings:
- **Performance Realism:** {'GOOD' if is_realistic else 'QUESTIONABLE'}
- **Model Diversity:** {'GOOD' if std_accuracy > 0.01 else 'LOW'}
- **Data Complexity:** {complexity['separability_assessment']}

### Recommendations:
{'✅ The model shows realistic performance on challenging data.' if is_realistic else
        '⚠️ Consider increasing data complexity or noise for more realistic evaluation.'}

---
*Report generated by Final Realistic Validation Pipeline*
"""

    # Save report
    report_path = f"results/final_realistic_validation_{timestamp}.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    # Save JSON results
    json_results = {
        'timestamp': timestamp,
        'dataset_info': {'samples': data_shape[0], 'features': data_shape[1]},
        'performance_summary': {
            'avg_accuracy': avg_accuracy,
            'std_accuracy': std_accuracy,
            'best_model': best_model,
            'best_accuracy': best_accuracy,
            'is_realistic': is_realistic
        },
        'complexity': complexity,
        'individual_results': results
    }

    json_path = f"results/final_realistic_validation_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)

    logger.info(f"📄 Report saved to: {report_path}")
    logger.info(f"📊 JSON results saved to: {json_path}")

    return report_path, json_path


def main():
    """Main validation pipeline"""
    logger.info("🚀 Starting Final Realistic Model Validation")

    # Load realistic data
    data = load_realistic_data()
    if data is None:
        logger.error("❌ Failed to load realistic dataset")
        return

    # Prepare data
    logger.info("🔄 Preparing data for validation...")

    # Find the target column
    target_columns = ['Label', 'Binary_Label', 'label', 'target']
    target_col = None
    for col in target_columns:
        if col in data.columns:
            target_col = col
            break

    if target_col is None:
        logger.error("❌ No target column found in dataset")
        return

    logger.info(f"📊 Using target column: {target_col}")
    X = data.drop(target_col, axis=1)
    y = data[target_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    logger.info(f"📊 Train set: {X_train.shape}, Test set: {X_test.shape}")

    # Analyze data complexity
    complexity = analyze_data_complexity(X, y)

    # Test all models
    logger.info("🧪 Testing all models on realistic data...")
    results = comprehensive_model_test(X_train, X_test, y_train, y_test)

    # Create visualization
    viz_path = "results/final_realistic_validation_analysis.png"
    create_performance_visualization(results, viz_path)

    # Generate final report
    report_path, json_path = generate_final_report(
        results, complexity, X.shape)

    logger.info("✅ Final realistic validation complete!")
    logger.info(f"📄 Check the report: {report_path}")

    # Print summary
    print("\n" + "="*60)
    print("🎯 FINAL REALISTIC VALIDATION SUMMARY")
    print("="*60)

    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        for model_name, model_results in valid_results.items():
            print(
                f"📊 {model_name:20} Accuracy: {model_results['accuracy']:.4f}")

        avg_acc = np.mean([r['accuracy'] for r in valid_results.values()])
        print(f"\n📈 Average Accuracy: {avg_acc:.4f}")
        print(f"🔬 Data Complexity: {complexity['separability_assessment']}")
        print(
            f"✅ Realistic Performance: {'YES' if avg_acc < 0.95 else 'QUESTIONABLE'}")

    print("="*60)


if __name__ == "__main__":
    main()
