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

## 8. Federated Learning (Updated Standalone Scripts)

We now use two standalone scripts instead of a single multi‑mode script:
- `server.py` (Flower server + Multi-Krum style robust aggregation)
- `client.py` (Flower NumPyClient for each data partition)

### 8.1 Core Concepts
- Start the server first; it listens on a TCP host:port (default `0.0.0.0:8080`).
- Each client connects to the server, loads its own partition (`client_<cid>_train.csv` / `client_<cid>_test.csv`).
- Only model parameters are exchanged (no raw data leaves a client).
- Robust aggregation (Multi-Krum variant) can down‑weight anomalous / poisoned updates.

### 8.2 Minimal Local Run (No Containers)
Open 5 terminals (Kali or Windows PowerShell). In terminal 1:
```bash
python server.py --rounds 5 --address 127.0.0.1:8080 --f 0
```
Then in the other 4 terminals (clients 0‑3):
```bash
python client.py --cid 0 --server 127.0.0.1:8080 --data_dir data/optimized
python client.py --cid 1 --server 127.0.0.1:8080 --data_dir data/optimized
python client.py --cid 2 --server 127.0.0.1:8080 --data_dir data/optimized
python client.py --cid 3 --server 127.0.0.1:8080 --data_dir data/optimized
```

Optional warm start (use previously trained centralized model weights):
```bash
python server.py --rounds 5 --address 127.0.0.1:8080 --initial_model results/balanced_centralized_model.h5
```

### 8.3 Interpreting CLI Output
- Server log lines are prefixed with `[SERVER]`.
- Client log lines show `[CLIENT <cid>]` and include training loss/accuracy per round.
- After each round, server prints which clients were selected by Multi‑Krum.
- Persistent history saved to `results/federated_metrics_history.json` (arrays of train vs test accuracy per round).

### 8.4 Typical Issues & Fixes
| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Client hangs on connect | Server not started / wrong address | Verify server listening; use same host:port |
| FileNotFoundError for partition | Wrong `--data_dir` | Point to directory containing `client_<cid>_train.csv` |
| Shape mismatch error | Different feature count across partitions | Ensure preprocessing identical for all clients |
| Train accuracy 1.0 suspicious | Data leakage / duplicates | Rebuild clean partitions (deduplicate & split globally first) |

---
## 10. Integration Plan: Docker & Kubernetes (Documentation Only)
This section documents HOW to containerize and orchestrate the current code base. It does NOT add Docker/K8s artifacts to the repo (another teammate will implement). Use these instructions to build reproducible deployments on a Kali Linux host (CLI environment).

### 10.1 Objectives
1. Encapsulate server and client runtime environments.
2. Launch N clients (scalable) against one server.
3. Preserve dataset partitions (either bake into images or mount volumes).
4. Support warm start from a mounted model artifact.
5. Allow Kubernetes to scale clients horizontally and roll updates safely.

### 10.2 Containerization Strategy
Separate images:
- Image A (Server): runs `server.py`.
- Image B (Client): runs `client.py` with dynamic client id from ENV.

Shared layers: Python base + `requirements.txt` + source tree.

#### Suggested Dockerfile Templates (Do NOT commit yet)
Server (pseudo‑Dockerfile):
```Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
ENTRYPOINT ["python", "server.py", "--address", "0.0.0.0:8080", "--rounds", "5"]
```
Client (pseudo‑Dockerfile):
```Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV CLIENT_ID=0 \
        SERVER_ADDRESS=fl-server:8080 \
        DATA_DIR=data/optimized
ENTRYPOINT ["sh", "-c", "python client.py --cid $CLIENT_ID --server $SERVER_ADDRESS --data_dir $DATA_DIR"]
```

### 10.3 Building & Running on Kali (Example Commands)
Install Docker (if not already):
```bash
sudo apt update
sudo apt install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER  # re-login after
```
Build images (run inside project root):
```bash
docker build -t fed-ddos-server -f Dockerfile.server .
docker build -t fed-ddos-client -f Dockerfile.client .
```
Run containers (network created automatically by Docker default bridge):
```bash
docker run -d --name fl-server -p 8080:8080 fed-ddos-server
docker run -d --name fl-client0 -e CLIENT_ID=0 -e SERVER_ADDRESS=host.docker.internal:8080 fed-ddos-client
docker run -d --name fl-client1 -e CLIENT_ID=1 -e SERVER_ADDRESS=host.docker.internal:8080 fed-ddos-client
docker run -d --name fl-client2 -e CLIENT_ID=2 -e SERVER_ADDRESS=host.docker.internal:8080 fed-ddos-client
docker run -d --name fl-client3 -e CLIENT_ID=3 -e SERVER_ADDRESS=host.docker.internal:8080 fed-ddos-client
```
If server also runs in a container network with clients (recommended):
```bash
docker network create flnet
docker run -d --name fl-server --network flnet fed-ddos-server
for i in 0 1 2 3; do docker run -d --name fl-client$i --network flnet -e CLIENT_ID=$i -e SERVER_ADDRESS=fl-server:8080 fed-ddos-client; done
```
View logs (CLI output expected on Kali):
```bash
docker logs -f fl-server
docker logs -f fl-client0
```

### 10.4 Volume & Data Handling
Options:
1. Bake CSV partitions into image (simplest, static).
2. Bind mount host directory (for iterative data changes):
```bash
docker run -d --name fl-client0 --network flnet \
    -v $(pwd)/data/optimized:/app/data/optimized \
    -e CLIENT_ID=0 -e SERVER_ADDRESS=fl-server:8080 fed-ddos-client
```
3. Use a Docker volume if partitions are generated inside container.

### 10.5 Warm Start in Container
Mount model file read‑only:
```bash
docker run -d --name fl-server --network flnet \
    -v $(pwd)/results/balanced_centralized_model.h5:/app/results/balanced_centralized_model.h5:ro \
    fed-ddos-server python server.py --address 0.0.0.0:8080 --rounds 5 --initial_model results/balanced_centralized_model.h5
```

### 10.6 Compose (Optional Outline)
Although not adding the file now, a `docker-compose.yml` would define one service for server and four replicated client services (or a single service scaled N times with distinct CLIENT_ID). Distinct IDs can be achieved via separate service blocks or an entrypoint script reading an injected ordinal.

### 10.7 Kubernetes Architecture (Planned)
Goal: One Deployment for server + Service (ClusterIP) and a scalable set of clients.

Components:
1. Deployment `fl-server` (replicas=1) exposing containerPort 8080.
2. Service `fl-server` (ClusterIP) at `fl-server:8080` inside cluster.
3. StatefulSet (or Deployment) `fl-client` with N replicas. Each pod derives its client id from its ordinal (0..N-1).
4. (Optional) PersistentVolumeClaim for shared partitions or mount ConfigMap if small synthetic data.
5. (Optional) Secret for credentials if pulling private base images.

StatefulSet client command pattern:
```bash
ORD=${HOSTNAME##*-}; python client.py --cid $ORD --server fl-server:8080 --data_dir data/optimized
```

### 10.8 Example (Illustrative, Not Committed)
Server manifest (conceptual):
```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: fl-server }
spec:
    replicas: 1
    selector: { matchLabels: { app: fl-server } }
    template:
        metadata: { labels: { app: fl-server } }
        spec:
            containers:
                - name: server
                    image: yourrepo/fed-ddos-server:latest
                    args: ["--address", "0.0.0.0:8080", "--rounds", "5"]
                    ports: [{ containerPort: 8080 }]
---
apiVersion: v1
kind: Service
metadata: { name: fl-server }
spec:
    selector: { app: fl-server }
    ports: [{ port: 8080, targetPort: 8080 }]
```
Clients manifest sketch (StatefulSet):
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata: { name: fl-client }
spec:
    serviceName: fl-client-headless
    replicas: 4
    selector: { matchLabels: { app: fl-client } }
    template:
        metadata: { labels: { app: fl-client } }
        spec:
            containers:
                - name: client
                    image: yourrepo/fed-ddos-client:latest
                    env:
                        - name: SERVER_ADDRESS
                            value: fl-server:8080
                    command: ["sh", "-c"]
                    args:
                        - ORD=${HOSTNAME##*-}; python client.py --cid $ORD --server $SERVER_ADDRESS --data_dir data/optimized
```

### 10.9 Scaling & Operations
Scale clients:
```bash
kubectl scale statefulset/fl-client --replicas=8
```
Check logs:
```bash
kubectl logs fl-client-0
kubectl logs deploy/fl-server
```
Rolling update (new image tag):
```bash
kubectl set image deployment/fl-server server=yourrepo/fed-ddos-server:NEW_TAG
kubectl set image statefulset/fl-client client=yourrepo/fed-ddos-client:NEW_TAG
```

### 10.10 Security & Hardening Considerations
- Restrict Service to ClusterIP (no external exposure) unless remote clients required.
- Add NetworkPolicies limiting ingress to server pod from client label only.
- Drop capabilities & run as non‑root in Dockerfiles (USER directive).
- Resource requests/limits: ensure server has enough CPU/RAM for aggregation.
- Add liveness/readiness probes for server (simple TCP or HTTP health if wrapper added).

### 10.11 CLI Output Expectations in Kali
Sample server line:
```
[SERVER] 2025-08-29 12:00:01 INFO: [Round 1] Multi-Krum selected clients: [0,1,2,3]
```
Sample client line:
```
[CLIENT 2] 2025-08-29 12:00:02 INFO: fit done loss=0.4123 acc=0.8571
```
Troubleshooting high latency: confirm cluster DNS resolves `fl-server`; test with `kubectl exec fl-client-0 -- nc -vz fl-server 8080`.

### 10.12 Troubleshooting Matrix (Containers / K8s)
| Issue | Container Context | K8s Context | Resolution |
|-------|------------------|-------------|------------|
| Connection refused | Wrong SERVER_ADDRESS | Service name typo | Confirm port 8080 open; check logs |
| File missing | Volume not mounted | PVC not bound | Mount correct host path / ensure PVC status Bound |
| CrashLoopBackOff | Entry command error | Args mis-specified | Describe pod; verify command syntax |
| Stale model weights | Old image cached | DaemonSet cache | Pull with new tag; use `imagePullPolicy: Always` temporarily |

---
## 11. Summary of Current vs Planned
| Aspect | Current (Repo) | Planned (Infra) |
|--------|----------------|-----------------|
| FL Scripts | `server.py`, `client.py` | Containerized entries |
| Robust Aggregation | Multi-Krum variant | Additional (Trimmed Mean, etc.) |
| Deployment | Manual local terminals | Docker images + K8s orchestrated |
| Scaling | Fixed 4 clients | Horizontal scale via StatefulSet |
| Warm Start | `--initial_model` flag | Mounted model artifact in container |

This document now serves as a handoff specification for the teammate implementing Docker and Kubernetes artifacts. No container files were added by this change.

---

## 9. File Structure

### 9.1 Key Scripts (Trimmed Set)
- `create_balanced_dataset.py`: (Historical) preprocessing script (kept for reference)
- `train_centralized.py`: Centralized binary CNN baseline training
- `server.py`: Standalone Flower federated server (Multi-Krum FedAvg)
- `client.py`: Standalone Flower federated client
- `model_analysis.py`: Performance analysis & visualization

### 9.2 Source Code (`src/`)
- `models/cnn_model.py`: CNN architecture definition
- `models/trainer.py`: Training utilities
- `data/preprocessing.py`: Data preprocessing functions
- `data/data_loader.py`: Data loading utilities
- (Legacy federated module retained for reference: `federated/`)

### 9.3 Data Storage (`data/`)
- `optimized/balanced_dataset.csv`: Original balanced dataset (legacy)
- `optimized/clean_partitions/client_*_train.csv`: Clean client train partitions
- `optimized/clean_partitions/client_*_test.csv`: Clean client test partitions

### 9.4 Results (`results/`)
- `balanced_centralized_model.h5`: Centralized baseline model weights
- `balanced_training_results.json`: Centralized baseline metrics
- `federated_metrics_history.json`: Federated round history (train/test accuracy)
- `*.png`: Visualizations

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
python server.py --rounds 5 --address 127.0.0.1:8080
python client.py --cid 0 --data_dir data/optimized/clean_partitions
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
