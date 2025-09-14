# 🛡️ Decentralized Federated DDoS Detection System

A complete **decentralized** federated learning implementation for DDoS attack detection using 1D CNN and the CICDDoS2019 dataset.

## 📋 Project Overview

This system implements:

- **1D CNN Model** for network traffic classification (29 features → Binary: Benign vs DDoS Attack)
- **Decentralized Federated Learning** using Flower framework with 4 distributed nodes
- **Non-IID Data Distribution** simulating real-world decentralized scenarios
- **Comprehensive Evaluation** comparing traditional centralized vs decentralized federated approaches

### 🎯 Supported Attack Types

- **BENIGN**: Normal network traffic
- **DrDoS_DNS**: DNS-based Distributed Reflection DoS
- **DrDoS_LDAP**: LDAP-based Distributed Reflection DoS
- **DrDoS_MSSQL**: MSSQL-based Distributed Reflection DoS
- **DrDoS_NetBIOS**: NetBIOS-based Distributed Reflection DoS

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone or navigate to project directory
cd federated-ddos-detection

# Create virtual environment
python -m venv fl_env

# Activate environment (Windows)
fl_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

The optimized dataset is already prepared in `data/optimized/` with:

### 2. Data Preparation

Clean, deduplicated, globally split client partitions are already provided under:
`data/optimized/clean_partitions/`

Each client has two files: `client_<cid>_train.csv` and `client_<cid>_test.csv` (cid ∈ {0,1,2,3}). These were produced during earlier cleanup (duplicate removal and leakage mitigation). No regeneration scripts are retained in the trimmed repository; treat the partitions as authoritative for experiments.

Use them directly for centralized or federated runs (examples below).

### 3. Run the System

### Federated Setup (Current Implementation)

- **Framework**: Flower (flwr)
- **Strategy**: FedAvg + Multi-Krum variant (robust subset selection) in standalone `server.py`
- **Aggregation Metrics**: Separately records train vs test accuracy per round (`results/federated_metrics_history.json`)
- **Clients**: `client.py` standalone process (NumPyClient) pointing to clean partitions
- **Data Integrity**: New pipeline eliminates train/test leakage and duplicate rows

````

This provides an interactive menu with options for:

- Testing individual components
**Standalone Robust FL (current):**

Terminal 1 (server, e.g. 5 rounds):

```bash
python server.py --rounds 5 --address 127.0.0.1:8080
````

Terminals 2–5 (clients pointing to clean partitions):

```bash
python client.py --cid 0 --data_dir data/optimized/clean_partitions
python client.py --cid 1 --data_dir data/optimized/clean_partitions
python client.py --cid 2 --data_dir data/optimized/clean_partitions
python client.py --cid 3 --data_dir data/optimized/clean_partitions
```

Check persisted metrics:

```bash
type results/federated_metrics_history.json
```

- Running traditional centralized baseline (for comparison)
- Quick decentralized federated demo
- Full system demonstration

#### Option B: Direct Commands

**Test CNN Model:**

```bash
python src/models/trainer.py --test
```

**Run Centralized Binary Baseline (current maintained baseline):**

```bash
python train_centralized.py --data_dir data/optimized/clean_partitions --epochs 25
```

## 📁 Project Structure

```
federated-ddos-detection/
├── 📊 data/
│   └── optimized/              # Decentralized node datasets
│       ├── client_0_train.csv  # Node 0 training data
│       ├── client_0_test.csv   # Node 0 test data
│       └── ...                 # Nodes 1-3 data
├── 🧠 src/
│   ├── models/
│   │   ├── cnn_model.py        # 1D CNN model implementation
│   │   └── trainer.py          # Model training pipeline
│   ├── federated/
│   │   ├── flower_client.py    # Decentralized learning node
│   │   └── flower_server.py    # Coordination server (for aggregation only)
│   ├── data/                   # Data processing modules
│   └── evaluation/             # Evaluation utilities
├── 📝 (scripts removed)         # Legacy helper scripts pruned for minimal core
<!-- notebooks removed in trimmed repository -->
├── server.py                   # Standalone FL server (Multi-Krum FedAvg)
├── client.py                   # Standalone FL client
├── train_centralized.py        # Centralized baseline training
└── 📋 requirements.txt         # Dependencies
```

## 🔧 Technical Details

### Model Architecture

- **Input**: 29 numerical features (network traffic statistics)
- **Architecture**: 1D CNN with 3 convolutional layers + dense layers
- **Output**: Binary sigmoid classification (0 = Benign, 1 = Attack)
- **Optimization**: Adam optimizer with adaptive learning rate

### Decentralized Federated Setup

- **Framework**: Flower (flwr)
- **Strategy**: FedAvg + Multi-Krum subset selection (robustness) in `server.py`
- **Nodes**: 4 clients (can extend)
- **Privacy**: Only model parameters exchanged

### Data Distribution

Each decentralized node has specialized data to simulate real-world scenarios:

- **Node 0**: Balanced across all attack types
- **Node 1**: Specialized in DNS-based attacks
- **Node 2**: Specialized in LDAP/MSSQL attacks
- **Node 3**: Specialized in NetBIOS attacks + benign

## 📊 Performance Metrics

The system tracks:

- **Accuracy**: Overall classification accuracy across decentralized nodes
- **Per-class Accuracy**: Individual attack type detection rates
- **Confusion Matrix**: Detailed classification analysis
- **Federated Rounds**: Training convergence over decentralized rounds
- **Communication Efficiency**: Model parameter exchange between nodes
- **Privacy Preservation**: No raw data exposure during training

## 🧪 Testing

### Tests

Ad-hoc testing scripts were removed. Validate using:

```bash
python train_centralized.py --data_dir data/optimized/clean_partitions --epochs 1
python server.py --rounds 1 --address 127.0.0.1:8080 &
python client.py --cid 0 --data_dir data/optimized/clean_partitions
```

## 📈 Results

Results are saved in `results/` directory:

- **Training History**: Loss and accuracy over epochs/rounds for decentralized training
- **Evaluation Metrics**: Comprehensive performance analysis across nodes
- **Comparison**: Traditional centralized vs decentralized federated performance
- **Visualization**: Training curves and confusion matrices
- **Privacy Analysis**: Data locality and communication efficiency metrics

## 🛠️ Customization

### Modify Model Architecture

Edit `src/models/cnn_model.py`:

```python
def build(self):
    # Modify CNN layers here
    self.model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    # Add your custom layers
```

### Adjust Federated Parameters

Tune via CLI flags when starting the server:

```bash
python server.py --rounds 10 --f 0 --m -1 --min_fit 4 --min_eval 4 --min_available 4
```

Deeper changes: edit strategy initialization in `server.py`.

### Data Distribution

Clean partitions already embedded; no generation script retained. To redesign distribution, recreate a preprocessing script or revert to earlier commit history.

## 🐛 Troubleshooting

### Common Issues

1. Port in use → change `--address` (both server & clients).
2. Missing partition file → verify path `data/optimized/clean_partitions` and file name pattern.
3. All accuracies = 1.0 → confirm using clean partitions (no leakage) & not legacy files.
4. Connection refused → start server first; confirm host/port.

## 📚 Documentation

- **Context Document**: `context.md` - Technical background
- **Development Plan**: `DEVELOPMENT_PLAN.md` - Project roadmap
- **Phase 3 Roadmap**: `PHASE3_ROADMAP.md` - Current implementation
- **Data Methodology**: `DATA_CLEANUP_METHODOLOGY_REPORT.md`
- **Workspace Organization**: `WORKSPACE_ORGANIZATION_REPORT.md`

## 🤝 Contributing

1. Follow the modular architecture
2. Add comprehensive documentation
3. Include tests for new features
4. Update README for new capabilities

## 📄 License

This project is developed for educational and research purposes.

## 🙏 Acknowledgments

- **CICDDoS2019 Dataset**: University of New Brunswick
- **Flower Framework**: Adap GmbH
- **TensorFlow/Keras**: Google
- **Research Community**: Federated learning and cybersecurity researchers

---

**🚀 Ready to detect DDoS attacks with federated learning!**

## 🔄 Iterative Debugging & Change Log (Summary)

This section documents the major iterations performed to reach the current stable pipeline.

1. Dependency Conflict Resolution

   - Issue: `docker-compose` (PyPI) pinned `PyYAML<6` conflicting with security need for `pyyaml>=6`.
   - Fix: Removed `docker-compose` from `requirements.txt`; rely on Docker CLI plugin. Pinned `numpy<2.0` for TF compatibility.

2. Federated Prototype & Multi-Krum Integration

   - Added standalone `server.py` with `MultiKrumFedAvg` strategy (subset selection for robustness) and `client.py` (Flower `NumPyClient`).
   - Initial rounds fell back to FedAvg (insufficient clients for f=1); adjusted default `f=0`.

3. Model Output Shape Mismatch

   - Issue: Server initialized 5-class model vs clients using binary label → weight shape mismatch.
   - Fix: Unified to binary classification (`num_classes=1`, sigmoid + BCE) in `cnn_model.py`, server forces binary init.

4. Evaluation Metrics Misinterpretation

   - Issue: Reported 1.0 "accuracy" was training accuracy only (test evaluation failing silently earlier).
   - Fix: Refactored server strategy to separately aggregate `avg_client_train_accuracy` and `avg_client_test_accuracy` and persist to `results/federated_metrics_history.json`.

5. Dataset Leakage & Perfect Predictors

   - Symptom: Persistent 1.0 test accuracy across clients.
   - Diagnosis: `scripts/diagnose_splits.py` showed high train/test row overlap (Jaccard up to 11%) and many perfect label-mapping features.
   - Cause: Client-level splitting before global dedup + duplicated rows across partitions.

6. Clean Partition Rebuild

   - Implemented `scripts/rebuild_clean_partitions.py`:
     - Deduplicated (50,000 → 30,919 unique rows; 19,081 duplicates removed).
     - Global stratified train/test split before client partition.
     - Stratified per-client partitioning with near-zero overlap (Jaccard ≤0.00156).
   - Re-ran diagnostics: No perfect predictors; moderate feature-label correlations (≤0.465).

7. Baselines after Cleanup

   - Logistic baseline (`scripts/logistic_baseline.py`): ~0.75 accuracy, ROC-AUC ~0.89 → dataset non-trivial.
   - Centralized CNN baseline (`train_centralized.py`): ~0.90 test accuracy (5 epochs example) on clean split.

8. Federated Initialization from Centralized Model

   - Added `--initial_model` to `server.py` to start FL from centralized trained weights for fair comparative convergence.

9. History Persistence & Reporting

   - Server now writes train/test accuracy time-series to `results/federated_metrics_history.json`.

10. Scripts Added
    - `scripts/diagnose_splits.py`: Overlap, correlations, perfect predictors.
    - `scripts/rebuild_clean_partitions.py`: Clean, deduplicate, stratify, repartition.
    - `train_centralized.py`: Aggregated centralized binary baseline.

## 🧪 Centralized Baseline (Clean Partitions)

```bash
python train_centralized.py --data_dir data/optimized/clean_partitions --epochs 25 --batch 64 --lr 0.001
```

Outputs → model + metrics saved under `results/`.

## 🔁 Federated Training Initialized from Centralized Weights

Optional warm start for federated training using the enhanced model weights:

```bash
python server.py --rounds 10 --address 127.0.0.1:8080 \
    --initial_model results/best_enhanced_model.keras

python client.py --cid 0 --data_dir data/optimized/clean_partitions
python client.py --cid 1 --data_dir data/optimized/clean_partitions
python client.py --cid 2 --data_dir data/optimized/clean_partitions
python client.py --cid 3 --data_dir data/optimized/clean_partitions
```

Monitor history:

```bash
type results/federated_metrics_history.json
```

## 🔍 Diagnostics & Data Integrity

Legacy diagnostic scripts (rebuild & overlap checks) were removed after producing stable clean partitions. If regeneration is needed, recreate tooling or restore from version control history.

## 📌 Future Enhancements (Planned)

- Add Trimmed Mean / Coordinate Median aggregators.
- Norm clipping + cosine anomaly scoring before Multi-Krum selection.
- Global ROC-AUC tracking in federated rounds.
- Model export in Keras SavedModel format (`.keras`) + versioned artifacts.
- Docker/Kubernetes orchestration & traffic capture (Wireshark/tshark) integration.

---

## Quick Start

Coming soon...

## Development Status

- [x] Project structure setup
- [x] Environment configuration
- [x] Dataset preparation (realistic datasets + clean partitions)
- [x] CNN model implementation (binary, with results/best_enhanced_model.keras)
- [x] Federated learning setup (server.py, client.py; metrics persisted)
- [ ] Security mechanisms (next: secure aggregation/DP/TLS/mTLS)
- [x] Evaluation pipeline (model_analysis.py, final_realistic_validation.py)

## Contributors

- Hemanth Kumar CS

## Repository

GitHub: https://github.com/HemanthKumar-CS/Fedrated_DDoS_Detection.git

## License

MIT License
