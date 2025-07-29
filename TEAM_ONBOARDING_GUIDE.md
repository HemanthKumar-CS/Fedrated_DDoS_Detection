# Federated DDoS Detection Project: Step-by-Step Guide

Welcome to the collaborative guide for building and simulating the Decentralized Federated DDoS Detection System! This document will help you and your teammates understand the project structure, setup, and workflow, so you can contribute and run simulations with confidence.

---

## 1. Project Overview

- **Goal:** Detect DDoS attacks using a 1D CNN model and federated learning (FL) with the Flower framework.
- **Dataset:** CICDDoS2019, preprocessed and split for 4 federated clients (non-IID data).
- **Key Features:**
  - No raw data sharing (privacy-preserving)
  - Centralized vs. federated comparison
  - Modular, extensible codebase

---

## 2. Project Structure

```
federated-ddos-detection/
├── data/
│   ├── optimized/              # Preprocessed datasets for each client
│   └── raw/                   # Original dataset (large, zipped)
├── src/
│   ├── models/                # CNN model and training logic
│   ├── federated/             # Flower client/server code
│   ├── data/                  # Data processing modules
│   └── evaluation/            # Metrics and analysis
├── scripts/                   # Data prep and utility scripts
├── notebooks/                 # Jupyter notebooks for analysis
├── launcher.py                # Interactive launcher
├── demo.py                    # Full system demo
├── requirements.txt           # Python dependencies
└── ... (docs, reports, etc.)
```

---

## 3. Environment Setup

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd federated-ddos-detection
   ```
2. **Create and activate virtual environment**
   ```bash
   python -m venv fl_env
   fl_env\Scripts\activate  # On Windows
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **(Optional) Install Docker Desktop**
   - For advanced simulation (multi-node, containerized)
   - Download from [docker.com](https://www.docker.com/)

---

## 4. Data Preparation

- **Optimized data** is already available in `data/optimized/` for quick start.
- To regenerate or customize data splits:
  ```bash
  python scripts/prepare_federated_data.py
  ```
- Each client gets its own train/test CSVs (non-IID distribution).

---

## 5. Running the System

### A. Interactive Launcher (Recommended)

```bash
python launcher.py
```

- Menu-driven interface for testing, training, and demos.

### B. Direct Commands

- **Test CNN Model:**
  ```bash
  python src/models/trainer.py --test
  ```
- **Run Centralized Baseline:**
  ```bash
  python demo.py --no_federated --centralized_epochs 10
  ```
- **Federated Learning Simulation:**
  - Start server:
    ```bash
    python src/federated/flower_server.py --rounds 10 --clients 4
    ```
  - Start clients (in separate terminals):
    ```bash
    python src/federated/flower_client.py --client_id 0
    python src/federated/flower_client.py --client_id 1
    python src/federated/flower_client.py --client_id 2
    python src/federated/flower_client.py --client_id 3
    ```
- **Full Demo:**
  ```bash
  python demo.py
  ```

---

## 6. How Federated Learning Works Here

- **Server** coordinates training rounds and aggregates model weights (FedAvg).
- **Clients** train locally on their own data, send model updates to server.
- **No raw data** leaves any client—only model parameters are shared.
- **Rounds**: Typically 10-20, configurable.

---

## 7. Key Files to Understand/Modify

- `src/models/cnn_model.py` – CNN architecture
- `src/models/trainer.py` – Training pipeline
- `src/federated/flower_client.py` – FL client logic
- `src/federated/flower_server.py` – FL server logic
- `scripts/prepare_federated_data.py` – Data splitting
- `src/data/preprocessing.py` – Data cleaning and feature engineering

---

## 8. Evaluation & Results

- Results (accuracy, confusion matrix, etc.) are saved in `results/`.
- Compare centralized vs. federated performance.
- Use `src/evaluation/` modules for custom analysis.

---

## 9. Troubleshooting

- **TensorFlow errors?**
  - Ensure virtual environment is active: `fl_env\Scripts\activate`
  - Reinstall: `pip install tensorflow`
- **Port in use?**
  - Change server/client port with `--address` or `--server` flags.
- **Data not found?**
  - Check `data/optimized/` for client CSVs.

---

## 10. Contributing as a Team

- Follow modular structure—work on separate modules (model, FL, data, etc.).
- Document your code and changes.
- Use Git for version control.
- Communicate regularly about progress and blockers.

---

## 11. Useful Resources

- [TensorFlow CNN Guide](https://www.tensorflow.org/tutorials/images/cnn)
- [Flower Framework Docs](https://flower.dev/docs/)
- [CICDDoS2019 Dataset](https://www.unb.ca/cic/datasets/ddos-2019.html)

---

**Ready to build and simulate? Start with `python launcher.py` or explore the code modules above!**

For any questions, check the `README.md`, `DEVELOPMENT_PLAN.md`, and `PHASE3_ROADMAP.md` for more details.
