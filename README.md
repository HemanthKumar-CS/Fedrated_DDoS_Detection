# 🛡️ Decentralized Federated DDoS Detection System

A complete **decentralized** federated learning implementation for DDoS attack detection using 1D CNN and the CICDDoS2019 dataset.

## 📋 Project Overview

This system implements:

- **1D CNN Model** for network traffic classification (29 features → 5 attack types)
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

- **50,000 samples** (10K per attack type)
- **4 federated clients** with non-IID distribution
- **29 optimized features** selected for best performance

### 3. Run the System

#### Option A: Interactive Launcher (Recommended)

```bash
python launcher.py
```

This provides an interactive menu with options for:

- Testing individual components
- Running traditional centralized baseline (for comparison)
- Quick decentralized federated demo
- Full system demonstration

#### Option B: Direct Commands

**Test CNN Model:**

```bash
python src/models/trainer.py --test
```

**Run Traditional Centralized Baseline (for comparison):**

```bash
python demo.py --no_federated --centralized_epochs 10
```

**Run Decentralized Federated Learning:**

Terminal 1 (Coordination Server):

```bash
python src/federated/flower_server.py --rounds 10 --clients 4
```

Terminal 2-5 (Decentralized Nodes):

```bash
python src/federated/flower_client.py --client_id 0
python src/federated/flower_client.py --client_id 1
python src/federated/flower_client.py --client_id 2
python src/federated/flower_client.py --client_id 3
```

**Complete Demo:**

```bash
python demo.py
```

## 📁 Project Structure

```
federated-ddos-detection/
├── 📊 data/
│   ├── optimized/              # Decentralized node datasets
│   │   ├── client_0_train.csv  # Node 0 training data
│   │   ├── client_0_test.csv   # Node 0 test data
│   │   └── ...                 # Nodes 1-3 data
│   └── raw/                    # Original dataset archive
├── 🧠 src/
│   ├── models/
│   │   ├── cnn_model.py        # 1D CNN model implementation
│   │   └── trainer.py          # Model training pipeline
│   ├── federated/
│   │   ├── flower_client.py    # Decentralized learning node
│   │   └── flower_server.py    # Coordination server (for aggregation only)
│   ├── data/                   # Data processing modules
│   └── evaluation/             # Evaluation utilities
├── 📝 scripts/
│   ├── data_explorer.py        # Dataset analysis
│   ├── prepare_federated_data.py # Data preparation
│   └── workspace_cleaner.py    # Workspace management
├── 📓 notebooks/
│   └── data_analysis.ipynb     # Jupyter analysis notebook
├── 🚀 launcher.py              # Interactive launcher
├── 🎯 demo.py                  # Complete system demo
└── 📋 requirements.txt         # Dependencies
```

## 🔧 Technical Details

### Model Architecture

- **Input**: 29 numerical features (network traffic statistics)
- **Architecture**: 1D CNN with 3 convolutional layers + dense layers
- **Output**: 5-class softmax classification
- **Optimization**: Adam optimizer with adaptive learning rate

### Decentralized Federated Setup

- **Framework**: Flower (flwr) decentralized federated learning
- **Strategy**: FedAvg (Federated Averaging) - aggregation happens at coordination server
- **Nodes**: 4 decentralized nodes with non-IID data distribution
- **Communication**: gRPC-based peer-to-peer communication with minimal coordination
- **Privacy**: No raw data leaves individual nodes - only model parameters shared

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

### Component Tests

```bash
# Test CNN model
python src/models/trainer.py --test

# Test decentralized node
python src/federated/flower_client.py --test

# Test coordination server
python src/federated/flower_server.py --test
```

### Integration Tests

```bash
# Run complete system test
python demo.py --centralized_epochs 5 --federated_rounds 5
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

### Adjust Decentralized Parameters

Edit `src/federated/flower_server.py`:

```python
strategy = DDoSFederatedStrategy(
    min_fit_clients=2,      # Minimum nodes for training round
    min_evaluate_clients=2, # Minimum nodes for evaluation
    # Customize decentralized strategy parameters
)
```

### Change Data Distribution

Edit `scripts/prepare_federated_data.py` to modify how data is distributed across decentralized nodes.

## 🐛 Troubleshooting

### Common Issues

**1. TensorFlow Import Errors**

```bash
# Ensure virtual environment is activated
fl_env\Scripts\activate
pip install tensorflow
```

**2. Port Already in Use**

```bash
# Change coordination server port
python src/federated/flower_server.py --address localhost:8081
python src/federated/flower_client.py --server localhost:8081
```

**3. Data Files Not Found**

```bash
# Verify data exists
ls data/optimized/client_*_*.csv
# Should show 8 files (4 train + 4 test) for 4 decentralized nodes
```

**4. Node Connection Issues**

- Ensure coordination server is started first
- Check firewall settings for peer-to-peer communication
- Verify correct coordination server address

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

**🚀 Ready to detect DDoS attacks with federated learning!** Run `python launcher.py` to get started.

## Quick Start

Coming soon...

## Development Status

- [x] Project structure setup
- [x] Environment configuration
- [ ] Dataset preparation
- [ ] CNN model implementation
- [ ] Federated learning setup
- [ ] Security mechanisms
- [ ] Evaluation pipeline

## Contributors

- Hemanth Kumar CS

## Repository

GitHub: https://github.com/HemanthKumar-CS/Fedrated_DDoS_Detection.git

## License

MIT License
