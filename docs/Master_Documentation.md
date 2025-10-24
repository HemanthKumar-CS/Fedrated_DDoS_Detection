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
- `src/visualization/training_visualizer.py` — Unified plot & dashboard generation (centralized + federated)
- `src/data/` — Data loading, preprocessing, federated split helpers
- `src/federated/` — Alternative Flower client/server module (reference)
- `server.py` — Standalone Flower server with robust FedAvg + Multi-Krum aggregation
- `client.py` — Standalone Flower NumPyClient
- `train_centralized.py` — Centralized training baseline (auto-plots)
- `train_enhanced.py` — Enhanced architecture + strategic training + auto visualizations
- `federated_training.py` — Simulation driver (single process Flower simulation)
- `model_analysis.py` — Advanced post‑hoc analysis & recommendations
- `final_realistic_validation.py` — Final evaluation & reproducible validation report
- `validate_test_set.py` — Integrity & distribution checks on provided test set
- `prepare_federated_partitions.py` — (Re)generate clean federated client splits
- `results/` — Saved models, metrics, plots, validation & analysis reports
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
- Artifacts: baseline model, JSON metrics (if enabled), plots via visualizer (`training_results_visualization.png`).

### Enhanced centralized training

- Optimized architecture, focal loss, class weighting, threshold optimization & extended diagnostics:
  - python train_enhanced.py
- Generates: `best_enhanced_model.keras`, `enhanced_training_results_<timestamp>.json`, `enhanced_training_analysis.png`.

### Federated learning (local machine)

1. Start the server (choose rounds and address):
   - python server.py --rounds 5 --address 127.0.0.1:8080 [--initial_model results/best_enhanced_model.keras]
2. Start clients in separate terminals (one per client id):
   - python client.py --cid 0 --data_dir data/optimized/clean_partitions
   - python client.py --cid 1 --data_dir data/optimized/clean_partitions
   - python client.py --cid 2 --data_dir data/optimized/clean_partitions
   - python client.py --cid 3 --data_dir data/optimized/clean_partitions
3. (Alternative) Single-process simulation:
   - python federated_training.py

Server behavior:

- Aggregates client updates using robust FedAvg + Multi-Krum subset selection (falls back to pure FedAvg when insufficient clients for safety conditions).
- Persists per-round metrics to `results/federated_metrics_history.json` and (in simulation) `results/federated_training_results.json`.
- Differentiates average client train accuracy (fit) vs evaluation accuracy (evaluate) per round.

## Model summary

- Input: 29 features reshaped to (29, 1) for Conv1D.
- Backbone: 1D CNN blocks (Conv1D + BatchNorm + ReLU), pooling, global pooling.
- Head: Dense layers with BatchNorm and Dropout; final sigmoid output.
- Loss/optimizer: Binary cross-entropy with Adam; metrics include accuracy (and in scripts, precision/recall where relevant).

## Results and artifacts

Core generated artifacts (all under `results/`):

| Category          | Files (examples)                                                                  | Description                                     |
| ----------------- | --------------------------------------------------------------------------------- | ----------------------------------------------- |
| Models            | `best_enhanced_model.keras`, `centralized_model.keras`                            | Trained weights for reuse/inference             |
| Centralized Plots | `training_results_visualization.png`, `enhanced_training_analysis.png`            | Learning curves & performance diagnostics       |
| Federated Metrics | `federated_metrics_history.json`, `federated_training_results.json`               | Round-wise aggregated metrics & simulation logs |
| Advanced Analysis | `advanced_model_analysis.png`, `comprehensive_model_analysis.json`                | Post‑hoc deep evaluation & recommendations      |
| Validation        | `final_realistic_validation_*.md/json`, `final_realistic_validation_analysis.png` | Reproducible final evaluation snapshot          |
| Integrity         | `data_integrity_report_*.md/json`                                                 | Dataset integrity and distribution reports      |

Automatic Visualization Pipeline:

- Centralized & enhanced training auto-call unified visualizer to produce standardized figures.
- Federated server appends metrics per round enabling longitudinal plots (accuracy, loss, optionally recall/F1 in future roadmap).

Interpretation:

- Expect 0.88–0.92 test accuracy for the binary baseline on realistic, balanced data depending on epochs and hyperparameters.
- Federated runs may be slightly lower on small rounds due to non-IID splits and communication overhead; warm starts can help.

## Robust aggregation (Multi-Krum + FedAvg)

Rationale:

- Federated learning can be vulnerable to anomalous or adversarial client updates (noise, poisoning, or system glitches).
- Multi-Krum selects a subset of updates with minimal aggregate pairwise distance, discarding statistical outliers prior to averaging.

Mechanics (implemented in `server.py`):

- Let n = number of received client updates; choose f = assumed max Byzantine clients.
- Safety requirement: n ≥ 2f + 3; otherwise revert to plain FedAvg.
- Distance matrix computed in parameter space (flattened weights → concatenated vector per client).
- For each client i: score_i = sum of distances to its closest (n - f - 2) peers.
- Select m lowest-score clients (m auto-chosen or bounded) and average their weights with standard FedAvg weighting.
- Logs selection and fallback events for auditability.

Fallback Behavior:

- If too few clients or numerical instability observed (e.g., NaNs), strategy safely reverts to vanilla FedAvg.

Limitations:

- Distance computation cost scales O(n^2 \* p) where p = total parameters; acceptable for small n (≤10) used here.
- Does not yet combine with norm clipping or DP—see roadmap.

## Automatic Visualization & Reporting

Implemented via `src/visualization/training_visualizer.py` and integrated into training scripts:

Centralized / Enhanced:

- Loss & accuracy curves (train vs validation)
- Recall (and precision when available)
- Confusion matrix
- ROC & PR curves with AUC annotations
- Threshold optimization (enhanced script) recorded in JSON

Federated:

- Round-wise aggregated train vs evaluation accuracy (extendable)
- Potential to add per-round variance, client participation heatmaps (future)

Advanced Analysis (`model_analysis.py`):

- Deep dive metrics, confidence distribution, threshold sweep, architecture summary, recommendations JSON.

## Troubleshooting

1. Client cannot connect — Ensure server is running and `--address` matches on both sides.
2. File not found — Verify `--data_dir` points to `data/optimized/clean_partitions`.
3. Shape/feature mismatch — Confirm partitions share the same preprocessing/columns across clients.
4. Suspicious perfect accuracy — Ensure you’re using the clean partitions (deduplicated, no leakage).
5. Port in use — Change `--address` or stop conflicting processes.

## Next steps and roadmap

- Add additional robust aggregators (Trimmed Mean, Coordinate-wise Median) and per-layer/coordinate median blending.
- Integrate adaptive norm clipping before distance scoring.
- Track federated ROC-AUC & PR-AUC each round (extend metrics history schema).
- Add client participation visualization & anomaly flags.
- Optional: Docker/Kubernetes orchestration + secure aggregation / DP.
- Deploy inference microservice + streaming feature extraction pipeline.

## Changelog (high level)

- Added robust aggregation (Multi-Krum subset selection + FedAvg fallback) in `server.py`.
- Implemented unified visualization & reporting module; automatic plot generation across training modes.
- Added enhanced training script with focal loss, attention components, threshold optimization.
- Added comprehensive model analysis + recommendation engine (`model_analysis.py`).
- Added final realistic validation producing timestamped reproducible reports.
- Expanded artifact taxonomy and updated documentation accordingly.

— End of Master Documentation —
