# Federated DDoS Detection

## 🚀 **OPTIMIZED 30-FEATURE FEDERATED LEARNING PIPELINE**

Advanced DDoS detection system with **optimized 30-feature federated learning** using 1D-CNN and robust Multi-Krum aggregation. Features intelligent feature selection, automatic visualization generation, and comprehensive model analysis.

## 🎯 **Major Updates & Optimizations**

### ✅ **30-Feature Optimization Pipeline**
- **Intelligent Feature Selection**: Automated top-30 feature selection using variance filtering, correlation pruning, and mutual information ranking
- **Unified Schema Enforcement**: All clients use exactly the same 30 features in the same order
- **Performance Boost**: Reduced from 78+ features to optimized 30 features for faster training and better generalization
- **Auto-Detection**: System automatically detects and applies optimized feature schema when `selected_features.json` is present

### ✅ **Enhanced Visualization System**
- **Essential Federated Analysis**: 3 key visualizations focusing on client-specific performance
  - Client Performance Metrics (Training vs Testing Accuracy & Loss as line graphs)
  - Client Confusion Matrices (CNN-based per client)
  - Client ROC Curves (Client-based CNN performance)
- **Line Graph Format**: Training/testing metrics displayed as comparative line graphs instead of separate bar charts
- **Robust Model Compatibility**: Automatic model rebuilding for visualization if input shapes mismatch

### ✅ **Robust Federated Architecture**
- **Multi-Krum Aggregation**: Byzantine-fault tolerant aggregation with client selection based on mutual distances
- **Dynamic Feature Detection**: Server and clients automatically align to optimized feature schema
- **Shape-Safe Operations**: Intelligent handling of model input/output shape mismatches
- **Enhanced Logging**: Comprehensive feature count validation and optimization status logging

## Key Features

- **Optimized 30-Feature Pipeline**: Intelligent feature selection and unified schema enforcement
- **Robust Federated Learning**: Multi-Krum + FedAvg with Byzantine fault tolerance
- **Advanced Visualization**: Essential client-focused analysis with line graph comparisons
- **Shape-Robust Operations**: Automatic model reconstruction for compatibility
- **Comprehensive Analysis**: Post-training evaluation and model recommendations
- **Reproducible Results**: All artifacts saved with detailed metrics and visualizations

## 🚀 Quickstart (Optimized 30-Feature Mode)

### Prerequisites
1. **Activate Virtual Environment**
```powershell
.\.venv\Scripts\Activate.ps1
```

2. **Install Dependencies**
```powershell
pip install -r requirements.txt
```

3. **Verify Optimized Data** (already provided)
- Data location: `data/optimized/clean_partitions/`
- Feature schema: `data/optimized/clean_partitions/selected_features.json` (30 features)
- Client data: `client_0..3_{train,test}.csv` (30 features + Binary_Label + Label)

### 🎯 **Optimized Federated Learning (RECOMMENDED)**

The system now defaults to **30-feature optimized mode** for maximum performance.

**Start Server:**
```powershell
python server.py --rounds 10 --address 127.0.0.1:8080
```

**Start Clients** (in separate terminals):
```powershell
python client.py --cid 0 --epochs 5
python client.py --cid 1 --epochs 5  
python client.py --cid 2 --epochs 5
python client.py --cid 3 --epochs 5
```

**Expected Output:**
- Client logs: `"Using optimized 30-feature schema."`
- Server logs: `"Using optimized feature list for global evaluation from ... (30 features)"`
- Model input shape: `(None, 30, 1)`
- Generated visualizations:
  - `results/federated_analysis/01_client_performance_metrics.png`
  - `results/federated_analysis/02_client_confusion_matrices.png`
  - `results/federated_analysis/03_client_roc_curves.png`

### 🔧 **Feature Optimization Management**

**Regenerate 30-Feature Schema:**
```powershell
python scripts/prepare_optimized_federated_dataset.py --input-dir data/optimized/clean_partitions --output-dir data/optimized/clean_partitions --clients 4 --k 30 --label-col Binary_Label
```

**Verify Feature Count:**
```python
import pandas as pd
df = pd.read_csv("data/optimized/clean_partitions/client_0_train.csv")
print(f"Total columns: {len(df.columns)}")  # Should be 32 (30 features + Binary_Label + Label)
print(f"Feature columns: {len(df.columns) - 2}")  # Should be 30
```

### 📊 **Legacy Options (Non-Optimized)**

**Centralized Training:**
```powershell
python train_centralized.py --data_dir data/optimized/clean_partitions --epochs 25
```

**Enhanced Training:**
```powershell
python train_enhanced.py
```

**Federated Simulation:**
```powershell
python federated_training.py
```

## 📊 **Automatic Visualization & Analysis**

### **Essential Federated Visualizations** (Auto-Generated)
The system generates **3 essential client-focused visualizations** after each federated training session:

1. **Client Performance Metrics** (`01_client_performance_metrics.png`)
   - **Training vs Testing Accuracy**: Comparative line graphs showing performance across all clients
   - **Training vs Testing Loss**: Comparative line graphs showing loss trends across all clients
   - **Format**: Side-by-side line graphs with value labels and legends

2. **Client Confusion Matrices** (`02_client_confusion_matrices.png`)
   - **Per-Client CNN Performance**: 2x2 grid showing confusion matrix for each of 4 clients
   - **Heatmap Format**: Color-coded matrices with count annotations
   - **Binary Classification**: Benign vs Attack classification results

3. **Client ROC Curves** (`03_client_roc_curves.png`)
   - **ROC Analysis**: ROC curves for each client's CNN performance
   - **AUC Scores**: Area Under Curve values for performance comparison
   - **Multi-Client Comparison**: All 4 client ROC curves on single plot

### **Visualization Features**
- **Line Graph Format**: Training vs testing metrics displayed as comparative line graphs (not separate bar charts)
- **Auto-Saved**: All plots automatically saved to `results/federated_analysis/`
- **High Resolution**: 300 DPI PNG format for publication quality
- **Comprehensive Logging**: Detailed generation status and file paths logged

### **Legacy Visualizations** (Enhanced Training)
- Learning curves (loss, accuracy, recall)
- ROC & Precision-Recall curves with AUC annotations
- Threshold analysis and optimization curves

## 🏗️ **Architecture & Technical Details**

### **30-Feature Optimization Pipeline**
```
Raw Dataset (78+ features) 
    ↓
Variance Filtering (remove zero-variance features)
    ↓
Correlation Pruning (remove highly correlated features >0.95)
    ↓
Mutual Information Ranking (rank by relevance to Binary_Label)
    ↓
Top-30 Feature Selection
    ↓
Schema Enforcement (exact column order, zero-fill missing, drop extras)
    ↓
Client Training (30 features + Binary_Label)
```

### **Robust Federated Aggregation**
```
Client Updates → Multi-Krum Distance Calculation → Client Selection → FedAvg → Global Model
```

- **Multi-Krum Selection**: Selects subset of mutually closest client updates
- **Byzantine Tolerance**: Filters out potential malicious/outlier updates
- **Fallback**: Automatic fallback to FedAvg when insufficient clients
- **Enhanced Logging**: Detailed aggregation method and client selection logging

### **Shape-Robust Model Operations**
- **Auto-Detection**: Automatic detection of input feature mismatches
- **Model Rebuilding**: Intelligent reconstruction of compatible models for visualization
- **Weight Transfer**: Safe weight copying between models with compatible architectures
- **Error Handling**: Graceful handling of shape incompatibilities

## 📁 **Repository Structure & Scripts**

| Script/Module                                    | Purpose                                                          | Status      |
|------------------------------------------------|------------------------------------------------------------------|-------------|
| **🎯 OPTIMIZED PIPELINE**                      |                                                                  |             |
| `client.py`                                    | **Optimized federated client** (auto-detects 30-feature schema) | ✅ Updated  |
| `server.py`                                    | **Robust federated server** (Multi-Krum + shape-safe model saving) | ✅ Updated  |
| `scripts/prepare_optimized_federated_dataset.py` | **30-feature optimization script** (generates selected_features.json) | ✅ New      |
| `src/visualization/training_visualizer.py`     | **Enhanced visualization system** (3 essential plots, line graphs) | ✅ Updated  |
| **CORE TRAINING MODULES**                      |                                                                  |             |
| `train_centralized.py`                        | Baseline centralized training                                    | ✅ Stable   |
| `train_enhanced.py`                            | Enhanced architecture with focal loss & threshold optimization  | ✅ Stable   |
| `federated_training.py`                       | Flower simulation driver (single process)                       | ✅ Stable   |
| **ANALYSIS & VALIDATION**                      |                                                                  |             |
| `model_analysis.py`                           | Deep post-training analysis & recommendations                    | ✅ Stable   |
| `final_realistic_validation.py`               | Final evaluation & comprehensive reporting                       | ✅ Stable   |
| `validate_test_set.py`                        | Data integrity & distribution validation                         | ✅ Stable   |
| `prepare_federated_partitions.py`             | Legacy client partition generator                                | ✅ Legacy   |

## 📦 **Data Structure (Optimized)**

```
data/optimized/clean_partitions/
├── selected_features.json              # 30-feature schema definition
├── client_0_train.csv                  # Client 0 training data (30 features + labels)
├── client_0_test.csv                   # Client 0 test data (30 features + labels)
├── client_1_train.csv                  # Client 1 training data
├── client_1_test.csv                   # Client 1 test data
├── client_2_train.csv                  # Client 2 training data
├── client_2_test.csv                   # Client 2 test data
├── client_3_train.csv                  # Client 3 training data
├── client_3_test.csv                   # Client 3 test data
├── partition_summary.json              # Partition statistics
└── clean_build_report.json             # Build process report

data/optimized/
├── realistic_test.csv                  # Global test dataset (for server evaluation)
├── realistic_train.csv                 # Global training dataset reference
└── OPTIMIZATION_SUMMARY.txt            # Optimization process summary
```

## 🎯 **Artifacts (results/)**

### **Optimized Federated Artifacts**
- **Models**: `federated_global_model.keras` (30-feature compatible)
- **Metrics**: `federated_metrics_history.json` (enhanced with aggregation logs)
- **Visualizations**: 
  - `federated_analysis/01_client_performance_metrics.png` (line graph comparisons)
  - `federated_analysis/02_client_confusion_matrices.png` (client-specific matrices)
  - `federated_analysis/03_client_roc_curves.png` (client ROC comparison)

### **Legacy Training Artifacts**
- **Models**: `best_enhanced_model.keras`, `centralized_model.keras`
- **Training JSON**: `enhanced_training_results_<timestamp>.json`, `centralized_training_results.json`
- **Analysis**: `comprehensive_model_analysis.json`, `advanced_model_analysis.png`
- **Validation**: `final_realistic_validation_<timestamp>.md/json`

## 🛡️ **Robust Aggregation (Multi-Krum + FedAvg)**

### **Enhanced Byzantine Fault Tolerance**
The system implements **Enhanced Multi-Krum** aggregation for robust federated learning:

**Algorithm:**
1. **Distance Calculation**: Compute pairwise squared distances between all client weight updates
2. **Krum Scoring**: For each client, sum distances to k-nearest neighbors (k = n - f - 2)
3. **Client Selection**: Select m clients with lowest Krum scores (m = n - f - 2)
4. **Historical Weighting**: Adjust scores based on client performance history
5. **FedAvg**: Perform weighted averaging over selected client updates

**Parameters:**
- `f`: Maximum number of Byzantine (malicious) clients to tolerate
- `m`: Number of selected updates for averaging
- **Auto-Fallback**: Falls back to standard FedAvg when client count insufficient (n < 2f + 3)

**Enhanced Features:**
- **Client Performance History**: Tracks historical accuracy for scoring adjustment
- **Detailed Logging**: Comprehensive aggregation method and client selection logs
- **Aggregation Analytics**: Stores round-wise aggregation decisions in metrics history

## 🔧 **Advanced Configuration**

### **Feature Optimization Settings**
```python
# scripts/prepare_optimized_federated_dataset.py parameters
--k 30                    # Number of features to select
--input-dir path          # Source directory with client CSVs  
--output-dir path         # Output directory (can be same as input)
--clients 4               # Number of clients
--label-col Binary_Label  # Target column name
```

### **Server Configuration**
```python
# server.py parameters
--rounds 10               # Number of federated rounds
--address 127.0.0.1:8080  # Server binding address
--f 1                     # Byzantine fault tolerance parameter
--log                     # Enable detailed logging
```

### **Client Configuration**
```python
# client.py parameters  
--cid 0                   # Client ID (0-3)
--epochs 5                # Local training epochs per round
--batch 32                # Local batch size
--data_dir path           # Data directory (defaults to optimized clean partitions)
```

## 🚨 **Troubleshooting**

### **Common Issues & Solutions**

**❌ Error: "expected shape=(None, 78, 1), found shape=(32, 30)"**
- **Cause**: Model saved with old 78-feature schema but data uses 30 features
- **Solution**: Re-run federated training to generate new 30-feature compatible model

**❌ Warning: "Active feature count is X, expected 30 in optimized mode"**
- **Cause**: `selected_features.json` missing or data not optimized
- **Solution**: Run optimization script to generate 30-feature schema

**❌ Error: "Missing client_*_train.csv"**  
- **Cause**: Data directory path incorrect
- **Solution**: Verify `--data_dir data/optimized/clean_partitions` path

**❌ Flower server connection issues**
- **Cause**: Network binding or client connection problems
- **Solution**: Use `127.0.0.1:8080` for local testing, ensure server starts before clients

### **Validation Commands**
```powershell
# Check feature count
python -c "import pandas as pd; df=pd.read_csv('data/optimized/clean_partitions/client_0_train.csv'); print(f'Features: {len(df.columns)-2}')"

# Verify selected_features.json
python -c "import json; f=open('data/optimized/clean_partitions/selected_features.json'); print(f'Selected: {len(json.load(f)[\"features\"])} features')"

# Test model compatibility
python -c "import tensorflow as tf; m=tf.keras.models.load_model('results/federated_global_model.keras'); print(f'Model input: {m.input_shape}')"
```

## 📚 **Documentation & References**

- **Master Documentation**: `docs/Master_Documentation.md` (comprehensive technical details)
- **Presentation**: `docs/Professional_Presentation_Document.md` (executive summary)
- **Research Context**: Federated learning for DDoS detection with Byzantine fault tolerance
- **Framework**: Flower federated learning framework with custom aggregation strategies

## 🆕 **Recent Changelog**

### **Version 2.0 - Optimized 30-Feature Pipeline**
- ✅ **30-Feature Optimization**: Intelligent feature selection and unified schema
- ✅ **Enhanced Visualizations**: 3 essential client-focused plots with line graph comparisons  
- ✅ **Shape-Robust Operations**: Automatic model reconstruction for compatibility
- ✅ **Multi-Krum Enhancement**: Byzantine fault tolerance with historical client performance
- ✅ **Auto-Detection**: Automatic optimized mode detection and application
- ✅ **Comprehensive Logging**: Feature count validation and optimization status tracking

### **Previous Versions**
- **v1.x**: Basic federated learning with FedAvg and visualization pipeline
- **v0.x**: Centralized baseline and enhanced training implementations

---

## 🎯 **Quick Verification Checklist**

After running optimized federated learning, verify:

- [ ] Client logs show: `"Using optimized 30-feature schema."`
- [ ] Server logs show: `"Using optimized feature list...30 features"`  
- [ ] Model input shape: `(None, 30, 1)`
- [ ] Generated files in `results/federated_analysis/`:
  - [ ] `01_client_performance_metrics.png` (line graph comparisons)
  - [ ] `02_client_confusion_matrices.png` (2x2 client matrices)
  - [ ] `03_client_roc_curves.png` (multi-client ROC curves)
- [ ] No shape compatibility errors in visualization generation
- [ ] `federated_metrics_history.json` contains aggregation logs

**For detailed methodology, architecture rationale, and research context, see `docs/Master_Documentation.md`**
