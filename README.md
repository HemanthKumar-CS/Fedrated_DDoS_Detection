# Federated Learning for Privacy Protected Cyber-Threat Detection

## Overview

This project implements a privacy-preserving federated learning framework for detecting DDoS attacks in distributed systems using lightweight CNN models.

## Project Structure

```
federated-ddos-detection/
├── src/
│   ├── models/          # CNN model implementations
│   ├── data/            # Data preprocessing and loading
│   ├── federated/       # FL client and server logic
│   ├── security/        # Security and robustness mechanisms
│   └── evaluation/      # Metrics and visualization
├── docker/              # Docker configurations
├── kubernetes/          # Kubernetes deployment files
├── notebooks/           # Jupyter notebooks for analysis
├── tests/              # Unit tests
├── docs/               # Documentation
├── data/               # Dataset storage (not in git)
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv fl_env
# On Windows:
fl_env\Scripts\activate
# On Linux/Mac:
source fl_env/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import flwr as fl; print('Flower version:', fl.__version__)"
```

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
