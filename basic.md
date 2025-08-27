# Federated DDoS Detection Project - Complete Workflow Guide

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Information](#dataset-information)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Architecture](#model-architecture)
5. [Training Process](#training-process)
6. [Results and Performance](#results-and-performance)
7. [Testing the Model](#testing-the-model)
8. [Federated Learning](#federated-learning)
9. [File Structure](#file-structure)

---

## 1. Project Overview

This project implements a **Federated DDoS Detection System** using deep learning to classify network traffic as either benign or malicious (DDoS attack). The system uses a 1D Convolutional Neural Network (CNN) trained on the CICIDDOS2019 dataset.

**Key Features:**
- Binary classification (Benign vs Attack)
- Federated learning implementation using Flower framework
- Real-time inference capabilities
- Comprehensive data analysis and visualization

---

## 2. Dataset Information

### 2.1 Original Dataset: CICIDDOS2019
- **Source**: Canadian Institute for Cybersecurity (CIC)
- **Total Original Samples**: ~12 million network flow records
- **Attack Types Included**:
  - DrDoS (Distributed Reflection DoS): DNS, LDAP, MSSQL, NetBIOS, NTP, SNMP, SSDP, UDP
  - Syn Flood attacks
  - TFTP attacks
  - UDP Lag attacks

### 2.2 Dataset Features (30 total)
The dataset contains network flow features extracted from packet captures:

**Flow-based Features:**
- `Flow Duration`: Duration of the network flow
- `Total Fwd Packets`: Total forward packets
- `Total Backward Packets`: Total backward packets
- `Total Length of Fwd Packets`: Total bytes in forward direction
- `Total Length of Bwd Packets`: Total bytes in backward direction

**Packet Size Features:**
- `Fwd Packet Length Max/Min/Mean/Std`: Forward packet size statistics
- `Bwd Packet Length Max/Min/Mean/Std`: Backward packet size statistics

**Timing Features:**
- `Flow Bytes/s`: Flow bytes per second
- `Flow Packets/s`: Flow packets per second
- `Flow IAT Mean/Std/Max/Min`: Inter-arrival time statistics

**TCP Flag Features:**
- `FIN Flag Count`, `SYN Flag Count`, `RST Flag Count`, etc.

**Advanced Features:**
- `Packet Length Mean/Std/Variance`: Packet size statistics
- `Down/Up Ratio`: Download to upload ratio
- `Average Packet Size`: Mean packet size
- `Subflow Fwd/Bwd Packets/Bytes`: Subflow statistics

### 2.3 Processed Dataset
**Location**: `data/optimized/balanced_dataset.csv`
- **Total Samples**: 50,000 (balanced)
- **Benign Samples**: 25,000 (50%)
- **Attack Samples**: 25,000 (50%)
- **Features**: 29 (after preprocessing)
- **Labels**: Binary (0 = Benign, 1 = Attack)

---

## 3. Data Preprocessing

### 3.1 Preprocessing Pipeline
**Script**: `create_balanced_dataset.py`

#### Step 1: Data Loading and Initial Cleaning
```python
# Load multiple CSV files from 01-12/ directory
# Remove duplicate records
# Handle missing values
```

#### Step 2: Feature Engineering
- **Label Mapping**: Convert multi-class labels to binary (Benign vs Attack)
- **Feature Selection**: Remove irrelevant features
- **Data Type Optimization**: Convert to appropriate data types

#### Step 3: Balancing Strategy
**Problem**: Original dataset was heavily imbalanced (100% attack samples)
**Solution**: 
- Sample equal numbers of benign and attack samples
- Create stratified splits to maintain attack type diversity
- Result: 50-50 balanced dataset

#### Step 4: Normalization and Scaling
**Method**: StandardScaler from scikit-learn
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

#### Step 5: Data Splitting
- **Training Set**: 80% of data
- **Testing Set**: 20% of data
- **Federated Splits**: Data divided among 4 clients for federated learning

### 3.2 Feature Extraction Algorithms
1. **Statistical Features**: Mean, standard deviation, min, max values
2. **Flow-based Features**: Packet counts, byte counts, flow duration
3. **Timing Features**: Inter-arrival times, flow rates
4. **Protocol Features**: TCP flag counts, packet size distributions

### 3.3 Storage Locations
- **Raw Data**: `01-12/` directory (original CSV files)
- **Balanced Dataset**: `data/optimized/balanced_dataset.csv`
- **Federated Data**: `data/optimized/client_*_train.csv` and `client_*_test.csv`
- **Preprocessing Reports**: `data/optimized/BALANCED_DATASET_SUMMARY.txt`

---

## 4. Model Architecture

### 4.1 1D CNN Architecture
**Script**: `src/models/cnn_model.py`

```
Input Layer (29 features) 
    ↓
Reshape Layer (29, 1)
    ↓
Conv1D Layer (64 filters, kernel=3) + ReLU
    ↓
MaxPooling1D (pool_size=2)
    ↓
Conv1D Layer (128 filters, kernel=3) + ReLU
    ↓
MaxPooling1D (pool_size=2)
    ↓
Conv1D Layer (256 filters, kernel=3) + ReLU
    ↓
GlobalAveragePooling1D
    ↓
Dense Layer (128 units) + ReLU
    ↓
Dropout (0.5)
    ↓
Dense Layer (64 units) + ReLU
    ↓
Dropout (0.3)
    ↓
Output Layer (1 unit) + Sigmoid
```

### 4.2 Model Specifications
- **Total Parameters**: ~99,000
- **Input Shape**: (29, 1) for 29 features
- **Output**: Single probability value (0-1)
- **Activation Functions**: ReLU (hidden), Sigmoid (output)
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy, Precision, Recall, F1-Score

---

## 5. Training Process

### 5.1 Centralized Training
**Script**: `train_balanced.py`

**Training Configuration:**
- **Epochs**: 50
- **Batch Size**: 32
- **Learning Rate**: 0.001 (Adam optimizer)
- **Validation Split**: 20%
- **Early Stopping**: Patience of 10 epochs

**Training Pipeline:**
1. Load balanced dataset
2. Split into train/validation/test sets
3. Apply feature scaling
4. Initialize CNN model
5. Train with early stopping
6. Save best model based on validation accuracy

### 5.2 Training Results
**Model Location**: `results/balanced_centralized_model.h5`
**Training Logs**: `results/balanced_training_results.json`

---

## 6. Results and Performance

### 6.1 Model Performance Metrics
**Test Results** (on 50,000 samples):
- **Accuracy**: 88.7%
- **Precision**: 89.0%
- **Recall**: 88.3%
- **F1-Score**: 88.7%
- **ROC-AUC**: 96.0%
- **PR-AUC**: 95.7%

### 6.2 Confusion Matrix Analysis
- **True Negatives**: 22,267 (Correctly identified benign)
- **False Positives**: 2,733 (Benign classified as attack)
- **False Negatives**: 2,916 (Attacks missed)
- **True Positives**: 22,084 (Correctly identified attacks)

### 6.3 Visualizations
**Location**: `results/`
- `advanced_model_analysis.png`: Comprehensive performance analysis
- `training_results_visualization.png`: Training curves and metrics

---

## 7. Testing the Model

### 7.1 Real-time Testing
**Script**: `model_demo.py`

**How to test:**
```bash
# Activate environment
.venv\Scripts\activate

# Run the demo
python model_demo.py
```

**Demo Features:**
- Interactive mode for testing individual samples
- Batch prediction mode
- Confidence score display
- Performance metrics calculation

### 7.2 Custom Testing
```python
# Load the trained model
import tensorflow as tf
model = tf.keras.models.load_model('results/balanced_centralized_model.h5')

# Prepare your data (same preprocessing as training)
# Make predictions
predictions = model.predict(your_data)
```

### 7.3 Performance Analysis
**Script**: `model_analysis.py`

**Analysis Features:**
- ROC curves
- Precision-Recall curves
- Feature importance analysis
- Prediction confidence distributions
- Error analysis

---

## 8. Federated Learning

### 8.1 Federated Setup
**Script**: `federated_training.py`

**Architecture:**
- **Framework**: Flower (FL framework)
- **Clients**: 4 federated clients
- **Data Distribution**: Non-IID (different attack types per client)
- **Aggregation**: FedAvg algorithm

### 8.2 Client Data Distribution
- **Client 0**: Mix of all attack types
- **Client 1**: Primarily DrDoS attacks
- **Client 2**: Primarily Syn attacks
- **Client 3**: Primarily TFTP/UDP attacks

### 8.3 Running Federated Learning
```bash
# Terminal 1: Start server
python federated_training.py --mode server

# Terminal 2-5: Start clients
python federated_training.py --mode client --client-id 0
python federated_training.py --mode client --client-id 1
python federated_training.py --mode client --client-id 2
python federated_training.py --mode client --client-id 3
```

---

## 9. File Structure

### 9.1 Key Scripts
- `create_balanced_dataset.py`: Data preprocessing and balancing
- `train_balanced.py`: Centralized model training
- `model_analysis.py`: Performance analysis and visualization
- `model_demo.py`: Real-time inference and testing
- `federated_training.py`: Federated learning implementation

### 9.2 Source Code (`src/`)
- `models/cnn_model.py`: CNN architecture definition
- `models/trainer.py`: Training utilities
- `data/preprocessing.py`: Data preprocessing functions
- `data/data_loader.py`: Data loading utilities
- `federated/flower_client.py`: Federated client implementation
- `federated/flower_server.py`: Federated server implementation

### 9.3 Data Storage (`data/`)
- `optimized/balanced_dataset.csv`: Main balanced dataset
- `optimized/client_*_train.csv`: Federated training data
- `optimized/client_*_test.csv`: Federated testing data

### 9.4 Results (`results/`)
- `balanced_centralized_model.h5`: Trained model
- `comprehensive_model_analysis.json`: Detailed metrics
- `*.png`: Visualization results

---

## Quick Start Guide

### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preprocessing
```bash
python create_balanced_dataset.py
```

### 3. Model Training
```bash
python train_balanced.py
```

### 4. Model Testing
```bash
python model_demo.py
```

### 5. Performance Analysis
```bash
python model_analysis.py
```

### 6. Federated Learning (Optional)
```bash
python federated_training.py --mode server
python federated_training.py --mode client --client-id 0
```

---

## Dependencies

- **TensorFlow**: Deep learning framework
- **Flower**: Federated learning framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning utilities
- **Matplotlib/Seaborn**: Visualization

For complete dependency list, see `requirements.txt`.
