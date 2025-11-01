# 🎯 Quick Reference: Commands & Usage

## 🚀 Quick Start (5 Minutes)

### 1️⃣ Start Federated Training
```bash
# Terminal 1: Start Server
python server.py --rounds 3 --address 127.0.0.1:8080

# Terminal 2-5: Start Clients (one in each)
python client.py --cid 0 --epochs 3
python client.py --cid 1 --epochs 3
python client.py --cid 2 --epochs 3
python client.py --cid 3 --epochs 3
```

### 2️⃣ Run Simulation Demo
```bash
python demo_simulation.py
```

### 3️⃣ Analyze Detection
```bash
# View demo results
open results/demo_simulation_analysis.png
```

---

## 📊 Understanding the Data Flow

```
Raw Dataset (CICDDoS2019: 78 features)
    ↓ [Feature Optimization]
Optimized Dataset (30 features + Binary_Label)
    ↓ [Split into 4 clients]
4 × [client_i_train.csv + client_i_test.csv]
    ↓ [Federated Training]
Global Model (federated_global_model.keras)
    ↓ [Inference]
Detection Score (0-1 probability)
```

---

## 🎯 What Each Component Does

| Component | File | Purpose | Input | Output |
|-----------|------|---------|-------|--------|
| **Server** | `server.py` | Coordinates training, aggregates weights | Clients | Global model |
| **Client** | `client.py` | Trains locally, sends weights | Client data | Updated weights |
| **Model** | `src/models/cnn_model.py` | 1D CNN architecture | 30 features | Binary classification |
| **Demo** | `demo_simulation.py` | Interactive training guide | Model + data | Visualizations |
| **Trainer** | `src/models/trainer.py` | Training utility | Model + data | Trained weights |

---

## 🔍 Feature Names (Top 10)

```
1. Flow Duration
2. Total Fwd Packets
3. Total Backward Packets
4. Total Length of Fwd Packets
5. Total Length of Bwd Packets
6. Fwd Packet Length Max
7. Fwd Packet Length Min
8. Fwd Packet Length Mean
9. Fwd Packet Length Std
10. Bwd Packet Length Max
... (20 more)
```

---

## 📈 Model Architecture

```
Input (30 features)
  ↓
Conv1D(32) + BatchNorm + MaxPool + Dropout
  ↓
Conv1D(64) + BatchNorm + MaxPool + Dropout
  ↓
Conv1D(128) + BatchNorm + GlobalMaxPool
  ↓
Dense(256) + BatchNorm + Dropout
  ↓
Dense(128) + Dropout
  ↓
Dense(1, sigmoid) → Output [0-1]
```

---

## 🎮 Detection Examples

```
BENIGN TRAFFIC:
Feature Set: Normal patterns
Model Output: 0.15
Decision: ✅ BENIGN

DDoS UDP FLOOD:
Feature Set: Extreme packet rate, one-way
Model Output: 0.94
Decision: 🚨 ATTACK

DDoS TCP SYN FLOOD:
Feature Set: Only SYN packets, high rate
Model Output: 0.89
Decision: 🚨 ATTACK

STEALTH ATTACK:
Feature Set: Balanced but anomalous
Model Output: 0.62
Decision: 🚨 ATTACK (borderline)
```

---

## 📊 Performance Targets

| Metric | Target | Typical | Excellent |
|--------|--------|---------|-----------|
| Accuracy | > 85% | 87.8% | > 92% |
| Precision | > 85% | 88.9% | > 90% |
| Recall | > 82% | 87.2% | > 90% |
| F1-Score | > 0.85 | 0.880 | > 0.90 |
| ROC-AUC | > 0.92 | 0.948 | > 0.95 |

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Model not found" | Run: `python server.py` + 4× `python client.py` |
| "Port in use" | Change port: `--address 127.0.0.1:8081` |
| "Feature mismatch" | Check: `len(client_0_train.csv columns) == 32` |
| "Low accuracy" | Increase epochs: `--epochs 5` |
| "Connection refused" | Wait for server to start, use `127.0.0.1` not `0.0.0.0` |

---

## 📁 Important Files

```
results/
├── federated_global_model.keras       ← Trained model
├── federated_metrics_history.json     ← Training metrics
└── federated_analysis/
    ├── 01_client_performance_metrics.png
    ├── 02_client_confusion_matrices.png
    └── 03_client_roc_curves.png

data/optimized/clean_partitions/
├── selected_features.json             ← Feature schema
├── client_0_train.csv
├── client_0_test.csv
... (and for clients 1-3)
```

---

## 🚀 Advanced Usage

### Custom Configuration
```bash
# More rounds, Byzantine tolerance
python server.py --rounds 10 --f 1 --min_fit 3

# Custom learning
python client.py --cid 0 --epochs 5 --batch 16
```

### Analyze Specific Client
```python
import pandas as pd
import tensorflow as tf

# Load specific client test data
df = pd.read_csv("data/optimized/clean_partitions/client_0_test.csv")

# Load model
model = tf.keras.models.load_model("results/federated_global_model.keras")

# Analyze...
```

### Export Results
```bash
# Metrics summary
cat results/federated_metrics_history.json | jq

# Training logs (from server output)
python server.py 2>&1 | tee server.log
```

---

## 🎯 Key Concepts

**Federated Learning**: Train on distributed data without centralizing
**Multi-Krum**: Byzantine-robust aggregation method
**Binary Classification**: Benign (0) vs Attack (1)
**1D CNN**: Learns patterns from sequential feature values
**Normalization**: (X - mean) / std for fair comparison

---

## 📚 Learning Path

1. **Understand**
   - Read: `SIMULATION_GUIDE.md`
   - Understand DDoS attacks and detection concepts

2. **Run**
   - Train: `python server.py` + `python client.py`
   - Detect: `demo_simulation.py`

3. **Analyze**
   - View: `results/` visualizations
   - Metrics: `federated_metrics_history.json`

4. **Deploy**
   - Use: `model_demo.py` for production
   - Monitor: Continuously track metrics
   - Retrain: Periodically with new data

---

## 💡 Pro Tips

✅ **Always start server before clients**
✅ **Use separate terminals for each client**
✅ **Check for port conflicts** if connection fails
✅ **Monitor GPU memory** during training
✅ **Save visualizations** for documentation
✅ **Backup models** before retraining
✅ **Validate feature schema** after data updates

---

**Happy detecting! 🛡️🔍**
