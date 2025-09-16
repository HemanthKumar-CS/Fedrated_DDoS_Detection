# Federated DDoS Detection — Master Documentation

This is the single, authoritative documentation for the project. It consolidates all prior reports and notes into one place. Keep this file updated going forward. The presentation document is intentionally preserved separately for slide-friendly content: see `docs/Professional_Presentation_Document.md`.

## Overview

- Goal: Detect DDoS traffic using a lightweight 1D CNN on tabular network-flow features, supporting both centralized and federated learning.
- Stack: Python 3.11+, TensorFlow/Keras, Flower (flwr), Pandas/NumPy/Scikit-learn, Matplotlib/Seaborn.
- Modes:
  - Centralized baseline: train/evaluate a single model on prepared data.
  - Federated learning: start a server and multiple clients; clients train locally and share only weights.

## Repository structure (key paths)

- `src/models/cnn_model.py` — CNN architecture and builder
- `src/models/trainer.py` — Baseline training/evaluation pipeline
- `src/data/` — Data loading, preprocessing, federated split helpers
- `src/federated/` — Alternative Flower client/server module (reference)
- `server.py` — Standalone Flower server with robust aggregation hooks
- `client.py` — Standalone Flower NumPyClient
- `train_centralized.py` — Centralized training baseline
- `prepare_federated_partitions.py` — Build clean federated client splits (if regeneration is needed)
- `results/` — Saved models, metrics, plots (artifacts)
- `data/optimized/` — Optimized datasets
  - `realistic_balanced_dataset.csv` — Combined dataset (binary: Benign vs Attack)
  - `clean_partitions/` — Authoritative per-client train/test CSVs

## Data pipeline

- Features: 29 numeric network-flow features (e.g., packet counts, sizes, rates, IAT stats) after preprocessing.
- Labels: Binary — 0 = Benign, 1 = Attack.
- Clean partitions are provided under `data/optimized/clean_partitions/`:
  - `client_<cid>_train.csv` and `client_<cid>_test.csv` for cid in {0,1,2,3}.
- Regeneration (optional): Use `prepare_federated_partitions.py` if you need to rebuild client splits from `realistic_balanced_dataset.csv`.

## Environment setup

1. Create a virtual environment (example for Windows PowerShell):
   - python -m venv .venv
   - .\.venv\Scripts\Activate.ps1
2. Install dependencies:
   - pip install -r requirements.txt

Note: TensorFlow may require additional runtime components depending on your hardware.

## How to run

### Centralized baseline

- Train and evaluate a centralized CNN on the prepared dataset/partitions:
  - python train_centralized.py --data_dir data/optimized/clean_partitions --epochs 25
- Artifacts: model and metrics saved under `results/` (for example, `results/best_enhanced_model.keras`).

### Federated learning (local machine)

1. Start the server (choose rounds and address):
   - python server.py --rounds 5 --address 127.0.0.1:8080
2. Start clients in separate terminals (one per client id):
   - python client.py --cid 0 --data_dir data/optimized/clean_partitions
   - python client.py --cid 1 --data_dir data/optimized/clean_partitions
   - python client.py --cid 2 --data_dir data/optimized/clean_partitions
   - python client.py --cid 3 --data_dir data/optimized/clean_partitions
3. Optional warm start from a centralized model:
   - python server.py --rounds 10 --address 127.0.0.1:8080 --initial_model results/best_enhanced_model.keras

Server behavior:

- Aggregates client updates using FedAvg; includes hooks for robust subset selection (Multi-Krum-style) when enough clients are available.
- Persists per-round metrics to `results/federated_metrics_history.json`.

## Model summary

- Input: 29 features reshaped to (29, 1) for Conv1D.
- Backbone: 1D CNN blocks (Conv1D + BatchNorm + ReLU), pooling, global pooling.
- Head: Dense layers with BatchNorm and Dropout; final sigmoid output.
- Loss/optimizer: Binary cross-entropy with Adam; metrics include accuracy (and in scripts, precision/recall where relevant).

## Results and artifacts

- Centralized baseline model: `results/best_enhanced_model.keras`
- Training/evaluation plots: `results/training_results_visualization.png`, `results/enhanced_training_analysis.png`, `results/final_realistic_validation_analysis.png`
- Comprehensive analysis: `results/comprehensive_model_analysis.json`
- Federated history: `results/federated_metrics_history.json`
- Final validation reports: `results/final_realistic_validation_*.md/json`

Interpretation:

- Expect 0.88–0.92 test accuracy for the binary baseline on realistic, balanced data depending on epochs and hyperparameters.
- Federated runs may be slightly lower on small rounds due to non-IID splits and communication overhead; warm starts can help.

## Troubleshooting

1. Client cannot connect — Ensure server is running and `--address` matches on both sides.
2. File not found — Verify `--data_dir` points to `data/optimized/clean_partitions`.
3. Shape/feature mismatch — Confirm partitions share the same preprocessing/columns across clients.
4. Suspicious perfect accuracy — Ensure you’re using the clean partitions (deduplicated, no leakage).
5. Port in use — Change `--address` or stop conflicting processes.

## Next steps and roadmap

- Add additional robust aggregators (Trimmed Mean, Coordinate-wise Median) and norm clipping.
- Track global ROC-AUC during federated rounds.
- Optional: Docker/Kubernetes orchestration and traffic-capture integration for demos.
- Optional: Differential privacy and/or secure aggregation for stronger privacy guarantees.

## Changelog (high level)

- Clean, deduplicated federated partitions were generated and adopted as the authoritative data source.
- Fixed initial weight shape mismatch (server vs client) by standardizing to binary output.
- Server now persists round-wise train vs test metrics for FL.
- Documentation consolidated into this master file; presentation doc preserved.

— End of Master Documentation —
