# 🔥 Production DDoS Detection System# Federated DDoS Detection



**REAL DATA ONLY** - No simulations, no synthetic data, no demos.## 🚀 **OPTIMIZED 30-FEATURE FEDERATED LEARNING PIPELINE**



## ✅ What WorksAdvanced DDoS detection system with **optimized 30-feature federated learning** using 1D-CNN and robust Multi-Krum aggregation. Features intelligent feature selection, automatic visualization generation, and comprehensive model analysis.



- ✅ Production CNN model for DDoS detection  ## 🎯 **Major Updates & Optimizations**

- ✅ Trains on REAL data from all 4 clients

- ✅ Binary classification: Benign vs Attack### ✅ **30-Feature Optimization Pipeline**

- ✅ Real model inference and evaluation- **Intelligent Feature Selection**: Automated top-30 feature selection using variance filtering, correlation pruning, and mutual information ranking

- ✅ Performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC)- **Unified Schema Enforcement**: All clients use exactly the same 30 features in the same order

- **Performance Boost**: Reduced from 78+ features to optimized 30 features for faster training and better generalization

## 🚀 Quick Start- **Auto-Detection**: System automatically detects and applies optimized feature schema when `selected_features.json` is present



### 1. Activate Virtual Environment### ✅ **Enhanced Visualization System**

- **Essential Federated Analysis**: 3 key visualizations focusing on client-specific performance

```powershell  - Client Performance Metrics (Training vs Testing Accuracy & Loss as line graphs)

.\.venv\Scripts\Activate.ps1  - Client Confusion Matrices (CNN-based per client)

```  - Client ROC Curves (Client-based CNN performance)

- **Line Graph Format**: Training/testing metrics displayed as comparative line graphs instead of separate bar charts

### 2. Train Model on Real Data- **Robust Model Compatibility**: Automatic model rebuilding for visualization if input shapes mismatch



```powershell### ✅ **Robust Federated Architecture**

python train.py- **Multi-Krum Aggregation**: Byzantine-fault tolerant aggregation with client selection based on mutual distances

```- **Dynamic Feature Detection**: Server and clients automatically align to optimized feature schema

- **Shape-Safe Operations**: Intelligent handling of model input/output shape mismatches

**Output:**- **Enhanced Logging**: Comprehensive feature count validation and optimization status logging

- ✅ Model: `results/ddos_model.h5`

- ✅ Scaler: `results/scaler.pkl`## Key Features

- ✅ Metrics: `results/metrics.json`

- ✅ Plot: `results/training_results.png`- **Optimized 30-Feature Pipeline**: Intelligent feature selection and unified schema enforcement

- **Robust Federated Learning**: Multi-Krum + FedAvg with Byzantine fault tolerance

### 3. Test on Real Data- **Advanced Visualization**: Essential client-focused analysis with line graph comparisons

- **Shape-Robust Operations**: Automatic model reconstruction for compatibility

```powershell- **Comprehensive Analysis**: Post-training evaluation and model recommendations

python inference.py- **Reproducible Results**: All artifacts saved with detailed metrics and visualizations

```

## 🚀 Quickstart (Optimized 30-Feature Mode)

**Output:**

- ✅ Inference results: `results/inference_results.json`### Prerequisites

- ✅ Detailed metrics for each client1. **Activate Virtual Environment**

- ✅ Overall performance summary```powershell

.\.venv\Scripts\Activate.ps1

## 📊 Expected Performance```



| Metric | Value |2. **Install Dependencies**

|--------|-------|```powershell

| Accuracy | ~87-90% |pip install -r requirements.txt

| Precision | ~88-91% |```

| Recall | ~85-90% |

| F1-Score | ~0.87-0.90 |3. **Verify Optimized Data** (already provided)

| ROC-AUC | ~0.94-0.96 |- Data location: `data/optimized/clean_partitions/`

- Feature schema: `data/optimized/clean_partitions/selected_features.json` (30 features)

## 📁 Data Structure- Client data: `client_0..3_{train,test}.csv` (30 features + Binary_Label + Label)



```### 🎯 **Optimized Federated Learning (RECOMMENDED)**

data/optimized/clean_partitions/

├── client_0_train.csv    ← Real training data (1000 samples)The system now defaults to **30-feature optimized mode** for maximum performance.

├── client_0_test.csv     ← Real test data (400 samples)

├── client_1_train.csv**Start Server:**

├── client_1_test.csv```powershell

├── client_2_train.csvpython server.py --rounds 10 --address 127.0.0.1:8080

├── client_2_test.csv```

├── client_3_train.csv

└── client_3_test.csv     ← Total: 4000 train + 1600 test samples**Start Clients** (in separate terminals):

``````powershell

python client.py --cid 0 --epochs 5

## 🔍 Model Architecturepython client.py --cid 1 --epochs 5  

python client.py --cid 2 --epochs 5

```python client.py --cid 3 --epochs 5

Input (30 features) ```

  ↓

Conv1D(64) → BatchNorm → Dropout → MaxPool**Expected Output:**

  ↓- Client logs: `"Using optimized 30-feature schema."`

Conv1D(128) → BatchNorm → Dropout → MaxPool- Server logs: `"Using optimized feature list for global evaluation from ... (30 features)"`

  ↓- Model input shape: `(None, 30, 1)`

Conv1D(256) → BatchNorm → Dropout → GlobalAvgPool- Generated visualizations:

  ↓  - `results/federated_analysis/01_client_performance_metrics.png`

Dense(128) → Dropout  - `results/federated_analysis/02_client_confusion_matrices.png`

  ↓  - `results/federated_analysis/03_client_roc_curves.png`

Dense(64) → Dropout

  ↓### 🔧 **Feature Optimization Management**

Dense(1, sigmoid) → Output (0=Benign, 1=Attack)

```**Regenerate 30-Feature Schema:**

```powershell

## 🎯 How to Usepython scripts/prepare_optimized_federated_dataset.py --input-dir data/optimized/clean_partitions --output-dir data/optimized/clean_partitions --clients 4 --k 30 --label-col Binary_Label

```

### Training

```powershell**Verify Feature Count:**

python train.py```python

```import pandas as pd

df = pd.read_csv("data/optimized/clean_partitions/client_0_train.csv")

### Inferenceprint(f"Total columns: {len(df.columns)}")  # Should be 32 (30 features + Binary_Label + Label)

```powershellprint(f"Feature columns: {len(df.columns) - 2}")  # Should be 30

python inference.py```

```

### 📊 **Legacy Options (Non-Optimized)**

### Custom Inference (Python)

```python**Centralized Training:**

import tensorflow as tf```powershell

import joblibpython train_centralized.py --data_dir data/optimized/clean_partitions --epochs 25

```

model = tf.keras.models.load_model('results/ddos_model.h5')

scaler = joblib.load('results/scaler.pkl')**Enhanced Training:**

```powershell

# Your datapython train_enhanced.py

X = your_data.values  # Shape: (n_samples, 30)```



# Predict**Federated Simulation:**

X_scaled = scaler.transform(X).reshape(-1, 30, 1)```powershell

predictions = model.predict(X_scaled)python federated_training.py

``````



## ⚙️ Configuration## 📊 **Automatic Visualization & Analysis**



Edit `train.py` for:### **Essential Federated Visualizations** (Auto-Generated)

- Batch size: Line ~195 `batch_size=32`The system generates **3 essential client-focused visualizations** after each federated training session:

- Epochs: Line ~192 `epochs=100`

- Learning rate: Line ~125 `learning_rate=0.001`1. **Client Performance Metrics** (`01_client_performance_metrics.png`)

- Dropout rates: Lines ~106, 110, 114, 128, 131   - **Training vs Testing Accuracy**: Comparative line graphs showing performance across all clients

   - **Training vs Testing Loss**: Comparative line graphs showing loss trends across all clients

## 📊 Results Location   - **Format**: Side-by-side line graphs with value labels and legends



```2. **Client Confusion Matrices** (`02_client_confusion_matrices.png`)

results/   - **Per-Client CNN Performance**: 2x2 grid showing confusion matrix for each of 4 clients

├── ddos_model.h5           ← Trained model   - **Heatmap Format**: Color-coded matrices with count annotations

├── scaler.pkl              ← Feature scaler   - **Binary Classification**: Benign vs Attack classification results

├── metrics.json            ← Training metrics

├── training_results.png    ← Performance plots3. **Client ROC Curves** (`03_client_roc_curves.png`)

├── inference_results.json  ← Test results   - **ROC Analysis**: ROC curves for each client's CNN performance

└── ...   - **AUC Scores**: Area Under Curve values for performance comparison

```   - **Multi-Client Comparison**: All 4 client ROC curves on single plot



## ✅ Verification### **Visualization Features**

- **Line Graph Format**: Training vs testing metrics displayed as comparative line graphs (not separate bar charts)

Check if everything works:- **Auto-Saved**: All plots automatically saved to `results/federated_analysis/`

- **High Resolution**: 300 DPI PNG format for publication quality

```powershell- **Comprehensive Logging**: Detailed generation status and file paths logged

# Verify data

python -c "### **Legacy Visualizations** (Enhanced Training)

import os- Learning curves (loss, accuracy, recall)

for i in range(4):- ROC & Precision-Recall curves with AUC annotations

    train = f'data/optimized/clean_partitions/client_{i}_train.csv'- Threshold analysis and optimization curves

    test = f'data/optimized/clean_partitions/client_{i}_test.csv'

    if os.path.exists(train) and os.path.exists(test):## 🏗️ **Architecture & Technical Details**

        print(f'✅ Client {i} data OK')

    else:### **30-Feature Optimization Pipeline**

        print(f'❌ Client {i} data MISSING')```

"Raw Dataset (78+ features) 

```    ↓

Variance Filtering (remove zero-variance features)

## 🔒 Federated Learning (Optional)    ↓

Correlation Pruning (remove highly correlated features >0.95)

If you want federated training:    ↓

Mutual Information Ranking (rank by relevance to Binary_Label)

```powershell    ↓

# Terminal 1Top-30 Feature Selection

python server.py --rounds 5    ↓

Schema Enforcement (exact column order, zero-fill missing, drop extras)

# Terminal 2-5 (separate terminals)    ↓

python client.py --cid 0Client Training (30 features + Binary_Label)

python client.py --cid 1```

python client.py --cid 2

python client.py --cid 3### **Robust Federated Aggregation**

``````

Client Updates → Multi-Krum Distance Calculation → Client Selection → FedAvg → Global Model

## 📝 Files```



| File | Purpose |- **Multi-Krum Selection**: Selects subset of mutually closest client updates

|------|---------|- **Byzantine Tolerance**: Filters out potential malicious/outlier updates

| `train.py` | Production training on real data |- **Fallback**: Automatic fallback to FedAvg when insufficient clients

| `inference.py` | Test model on real data |- **Enhanced Logging**: Detailed aggregation method and client selection logging

| `server.py` | Federated server (optional) |

| `client.py` | Federated client (optional) |### **Shape-Robust Model Operations**

| `src/models/cnn_model.py` | CNN model definition |- **Auto-Detection**: Automatic detection of input feature mismatches

| `requirements.txt` | Python dependencies |- **Model Rebuilding**: Intelligent reconstruction of compatible models for visualization

- **Weight Transfer**: Safe weight copying between models with compatible architectures

## ❌ What's Removed- **Error Handling**: Graceful handling of shape incompatibilities



- ❌ All demo scripts## 📁 **Repository Structure & Scripts**

- ❌ All simulation scripts

- ❌ Synthetic data generators| Script/Module                                    | Purpose                                                          | Status      |

- ❌ Training visualization demos|------------------------------------------------|------------------------------------------------------------------|-------------|

- ❌ Centralized training (use federated or direct training)| **🎯 OPTIMIZED PIPELINE**                      |                                                                  |             |

| `client.py`                                    | **Optimized federated client** (auto-detects 30-feature schema) | ✅ Updated  |

## 🚀 Next Steps| `server.py`                                    | **Robust federated server** (Multi-Krum + shape-safe model saving) | ✅ Updated  |

| `scripts/prepare_optimized_federated_dataset.py` | **30-feature optimization script** (generates selected_features.json) | ✅ New      |

1. Run `python train.py` to train the model| `src/visualization/training_visualizer.py`     | **Enhanced visualization system** (3 essential plots, line graphs) | ✅ Updated  |

2. Run `python inference.py` to test it| **CORE TRAINING MODULES**                      |                                                                  |             |

3. Check `results/` folder for outputs| `train_centralized.py`                        | Baseline centralized training                                    | ✅ Stable   |

4. Integrate model into your system| `train_enhanced.py`                            | Enhanced architecture with focal loss & threshold optimization  | ✅ Stable   |

| `federated_training.py`                       | Flower simulation driver (single process)                       | ✅ Stable   |

**Done! Pure production code.** 🎯| **ANALYSIS & VALIDATION**                      |                                                                  |             |

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
