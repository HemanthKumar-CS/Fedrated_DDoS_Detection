# Federated DDoS Detection

Lightweight + enhanced 1D-CNN pipelines for DDoS detection with both centralized and federated learning (Flower). Robust aggregation (FedAvg + Multi-Krum subset selection) and automatic visualization/reporting are built-in. This README is a quick operational guide—see `docs/Master_Documentation.md` for deep detail.

## Key Features

- Centralized baseline (`train_centralized.py`)
- Enhanced training pipeline with advanced architecture, focal loss & threshold optimization (`train_enhanced.py`)
- Federated learning (Flower) with robust FedAvg + Multi-Krum filtering (`server.py`, `client.py`)
- Automatic visualization & reporting (training curves, ROC, PR, confusion matrix, federated round metrics)
- Advanced model analysis module (`model_analysis.py`) for post‑hoc evaluation & recommendations
- Final realistic validation script (`final_realistic_validation.py`) and test set validator (`validate_test_set.py`)
- Reproducible artifacts (models, JSON metrics, plots) saved under `results/`

## Quickstart (Windows PowerShell)

1. Create & activate virtual environment

- python -m venv .venv
- .\.venv\Scripts\Activate.ps1

2. Install dependencies

- pip install -r requirements.txt

3. (Optional) Verify data exists under `data/optimized/clean_partitions/` (already provided)

### Centralized Baseline

python train_centralized.py --data_dir data/optimized/clean_partitions --epochs 25

### Enhanced Training (recommended)

python train_enhanced.py

Generates:

- `results/best_enhanced_model.keras`
- `results/enhanced_training_results_<timestamp>.json`
- `results/enhanced_training_analysis.png`

### Federated Learning (1 server + 4 clients)

Start server (robust FedAvg + Multi-Krum automatically engaged when client count sufficient):

python server.py --rounds 5 --address 127.0.0.1:8080 [--initial_model results/best_enhanced_model.keras]

Start clients in four separate terminals:

- python client.py --cid 0 --data_dir data/optimized/clean_partitions
- python client.py --cid 1 --data_dir data/optimized/clean_partitions
- python client.py --cid 2 --data_dir data/optimized/clean_partitions
- python client.py --cid 3 --data_dir data/optimized/clean_partitions

Federated artifacts:

- `results/federated_metrics_history.json` (round-wise train/test metrics)
- `results/federated_training_results.json` (if using `federated_training.py` simulation)

### Federated Simulation (single process)

python federated_training.py (runs Flower simulation instead of manual terminals)

### Advanced Model Analysis

python model_analysis.py (requires `results/best_enhanced_model.keras` and dataset)

Outputs:

- `results/advanced_model_analysis.png`
- `results/comprehensive_model_analysis.json`

### Final Realistic Validation

python final_realistic_validation.py --model results/best_enhanced_model.keras

Produces timestamped validation markdown & JSON reports in `results/`.

## Automatic Visualization & Reporting

Training scripts automatically generate:

- Learning curves (loss, accuracy, recall)
- Confusion matrix heatmaps
- ROC & Precision-Recall curves (AUC values annotated)
- Threshold analysis (enhanced training)
- Federated round history (accuracy/loss trajectories) via `training_visualizer.py`

## Repository Scripts Overview

| Script                          | Purpose                                             |
| ------------------------------- | --------------------------------------------------- |
| train_centralized.py            | Baseline centralized training                       |
| train_enhanced.py               | Enhanced architecture & strategies + visualizations |
| server.py / client.py           | Federated execution with robust aggregation         |
| federated_training.py           | Flower simulation driver (no manual terminals)      |
| model_analysis.py               | Deep post-training analysis & recommendations       |
| final_realistic_validation.py   | Final evaluation & report generation                |
| validate_test_set.py            | Integrity / distribution checks on test data        |
| prepare_federated_partitions.py | (Re)generate client CSV partitions                  |

## Artifacts (results/)

- Models: `best_enhanced_model.keras`, `centralized_model.keras`
- Training JSON: `enhanced_training_results_<timestamp>.json`, `centralized_training_results.json` (if produced)
- Federated: `federated_metrics_history.json`, `federated_training_results.json`
- Plots: `enhanced_training_analysis.png`, `advanced_model_analysis.png`, `training_results_visualization.png`, `final_realistic_validation_analysis.png`
- Validation & reports: `final_realistic_validation_<timestamp>.md/json`, `comprehensive_model_analysis.json`

## Robust Aggregation (Multi-Krum + FedAvg)

When enough clients participate, the server selects a subset of mutually closest client updates (Multi-Krum style) before averaging (FedAvg) to mitigate outliers / potential poisoning. Falls back to plain FedAvg automatically if client count too low.

## Documentation

- Master Doc: `docs/Master_Documentation.md`
- Presentation: `docs/Professional_Presentation_Document.md`

## Notes

- Default data path: `data/optimized/clean_partitions/`
- Warm start: pass `--initial_model` to `server.py` for federated bootstrapping
- Ensure `results/` exists (scripts create it automatically if missing)

## Changelog (recent)

- Added Multi-Krum robust aggregation layer on top of FedAvg
- Added automatic visualization/reporting pipeline (`training_visualizer.py`)
- Added enhanced training script and advanced analysis tools
- Added final realistic validation and test set integrity scripts

---

For full methodology, architecture rationale, and roadmap see the Master Documentation.
