# Federated DDoS Detection – Professional Technical Document

Subtitle: Implementation, Model Architecture, Methodology, Workflow, and Conclusion

## 1) Executive Overview

This project delivers a practical, privacy-preserving DDoS detection system that can train both centrally and in a federated setting. The core is a lightweight 1D Convolutional Neural Network (CNN) designed for tabular network-flow features. Federated learning (FL) is implemented with the Flower framework to keep raw data on distributed clients. The server coordinates multiple rounds of local training and aggregates model updates; the current server integrates a Multi-Krum-style robust selection on top of FedAvg to mitigate outlier client updates.

Key goals

- Accurate binary classification of network flows: Benign vs DDoS Attack
- Data minimization and privacy by training where data lives (clients/edges)
- Modular design that separates data processing, modeling, training, and federation
- Clear evaluation and export of artifacts for analysis and presentation

Where to look in the repo

- Model: `src/models/cnn_model.py`
- Training pipeline & evaluation: `src/models/trainer.py`
- Standalone FL client/server: `client.py`, `server.py`
- Alternative FL client wrapper: `src/federated/flower_client.py`
- Data utilities: `src/data/`
- Evaluation artifacts: `results/`

## 2) Implementation (What is built and how it is organized)

### 2.1 High-level system design

- Data layer (ingest and preprocessing)
  - Loads CSV partitions (client-specific train/test files) and selects numeric features.
  - Applies standardization per training split (centralized) or per client (federated).
  - Handles non-numeric features on clients by stable factorization to integers prior to scaling.
- Model layer (learnable representation)
  - 1D CNN that treats the feature vector as a short “signal” to capture local interactions among engineered traffic features.
  - Architecture is intentionally compact for fast iterations at the edge.
- Training layer (orchestration and evaluation)
  - Centralized: trains with Keras fit/evaluate loops, early stopping, learning-rate reduction on plateau.
  - Produces structured metrics (loss, accuracy, precision, recall), classification report, and confusion matrix.
  - Saves model artifacts (Keras format) and plots to `results/` for reporting.
- Federated layer (coordination across clients)
  - Clients perform local epochs on their own partitions and share only model weights/metrics.
  - Server coordinates rounds and aggregates weights via FedAvg augmented with Multi-Krum selection.
  - Server persists round-wise metrics to `results/federated_metrics_history.json` for visualization.

### 2.2 Key components and responsibilities

- `src/models/cnn_model.py`
  - Encapsulates CNN definition, compile-time configuration, input reshaping, and save/load utilities.
  - Exposes a small API surface to be reused by centralized and federated training flows.
- `src/models/trainer.py`
  - End-to-end training and evaluation for centralized experiments: load → split → normalize → train → evaluate → report.
  - Implements early stopping and learning-rate scheduling callbacks.
  - Generates confusion matrix and classification report for decision-maker-friendly outputs.
- `client.py`
  - Standalone Flower NumPyClient for federated training.
  - Loads a client’s partition, performs per-client preprocessing and local training, returns updated weights.
- `server.py`
  - Standalone Flower server with a Multi-Krum FedAvg strategy.
  - Separately tracks average client train accuracy (during fit) and test accuracy (during evaluate) per round.
- `src/data/`
  - Data helpers (loader, preprocessing, and federated split logic used by scripts in the repository).

Design choices

- Strict separation of concerns: data processing, model definition, and training orchestration are independent.
- Reuse of the same CNN and normalization principles across centralized and federated flows to maintain parity.
- Lightweight model to support quick rounds on modest client hardware.

## 3) Model Architecture (What the network looks like and why)

Conceptual input and output

- Input: A feature vector per flow with F numeric features (commonly 29–31 depending on dataset variant).
- Output: Binary probability p(Attack) via a sigmoid unit; a threshold (typically 0.5) maps probability to class.

Backbone (1D CNN)

- Input reshape: (samples, F) → (samples, F, 1) to enable Conv1D.
- Convolutional blocks (feature extractor):
  - Conv1D (filters: 32 → 64 → 128, kernel size 3) with ReLU to learn local feature interactions.
  - Batch Normalization for stable, faster training and robustness to scale shifts.
  - MaxPooling in early blocks to reduce dimensionality and overfitting risk.
- Global aggregation:
  - Global MaxPooling to summarize the strongest activations across the feature axis, reducing parameters vs Flatten.
- Classifier head:
  - Dense 256 (ReLU) + BatchNorm + Dropout for regularization.
  - Dense 128 (ReLU) + Dropout.
  - Output Dense 1 with sigmoid for binary classification.

Optimization and metrics

- Loss: Binary Cross-Entropy for binary classification.
- Optimizer: Adam with a moderate learning rate for stable convergence.
- Metrics: Accuracy, precision, and recall are tracked during training/evaluation.

Why this architecture for DDoS features

- Local correlations: Engineered statistics (rates, counts, sizes) often co-vary; Conv1D captures short-range patterns.
- Regularization: BatchNorm and Dropout mitigate overfitting in non-stationary network traffic.
- Efficiency: Global pooling reduces parameters, enabling faster client-side training in FL.

## 4) Methodology (How the system is used end-to-end)

### 4.1 Data methodology

- Partitioning
  - Centralized: Data is split into train/validation/test using stratification to preserve label balance.
  - Federated: Each client has disjoint train/test CSV partitions to simulate realistic, potentially non-IID distributions.
- Normalization
  - Centralized: Compute per-feature mean/std on training split; apply to validation/test.
  - Federated: Compute per-client mean/std on client’s training data; apply to that client’s test data.
- Handling non-numeric columns (federated)
  - Stable factorization to integers based on the client’s training categories; unseen test categories map to a fallback.

### 4.2 Centralized training methodology

- Create CNN with the correct input dimensionality based on data columns.
- Reshape features for 1D CNN and train with callbacks:
  - EarlyStopping (patience) restores best weights.
  - ReduceLROnPlateau halves the learning rate if validation loss stalls.
- Evaluate on a held-out test set:
  - Probability outputs are thresholded to labels; confusion matrix and classification report are generated.
- Persist artifacts to `results/` (metrics, plots, and model weights) for auditability and reuse.

### 4.3 Federated learning methodology

- Protocol
  - Round 0: Server initializes aggregation (optionally from a known good model); clients receive weights.
  - Local training: Each client trains for E epochs on its partition (no raw data leaves the client).
  - Update: Clients send model weights and metrics to the server.
  - Aggregation: The server applies FedAvg; if configured, a Multi-Krum subset selection filters outliers before averaging.
  - Repeat for R rounds.
- Robust aggregation (Multi-Krum-style)
  - Idea: Prefer updates that are mutually close in parameter space; treat distant updates as potential outliers or poisoned.
  - Parameter f: maximum assumed Byzantine (malicious) clients. Multi-Krum requires n ≥ 2f + 3 to operate; otherwise it falls back to FedAvg.
  - Selection m: number of “closest” updates to average (auto-derived if not provided).
- Observability
  - The server distinguishes average client train accuracy (fit phase) and average client test accuracy (evaluate phase) per round.
  - History persists to `results/federated_metrics_history.json` for downstream visualization.

### 4.4 Evaluation methodology (both modes)

- Thresholding: Sigmoid output → label via a configurable threshold (0.5 by default).
- Metrics reported
  - Accuracy: overall correctness.
  - Precision: proportion of predicted attacks that are true attacks.
  - Recall: proportion of actual attacks that are detected.
  - F1: harmonic mean of precision and recall (in detailed reports).
- Diagnostic views
  - Confusion matrix: True/False Positives/Negatives by class.
  - Learning curves: training/validation loss and accuracy progress (centralized) or round-wise FL metrics (federated).

## 5) System Workflow (From data to decision)

Centralized (baseline)

1. Load CSV dataset and identify label column (binary).
2. Split into train/validation/test with stratification.
3. Compute normalization statistics on train; transform val/test.
4. Create CNN with input dimensionality matching feature count.
5. Train with callbacks; monitor validation metrics.
6. Evaluate on test; export metrics and confusion matrix to `results/`.
7. Save trained model for reuse or deployment.

Federated (decentralized)

1. Prepare per-client CSV partitions (train/test per client).
2. Start FL server (aggregation strategy configured) and multiple clients.
3. For each round: server broadcasts, clients train locally, server aggregates.
4. Monitor train/test accuracy per round (history file in `results/`).
5. Optionally warm-start federation from a previously trained centralized model to improve convergence speed.

Artifacts and figures

- Figures and plots for inclusion in slides can be sourced from:
  - `Figure_1.png` (project-specific overview image)
  - `results/advanced_model_analysis.png`
  - `results/enhanced_training_analysis.png`
  - `results/final_realistic_validation_analysis.png`
  - `results/training_results_visualization.png`
- Time-series FL metrics: `results/federated_metrics_history.json`

## 6) Conclusion (What this delivers and why it matters)

Strengths

- Practical privacy-by-design training: raw traffic features remain local to each client.
- Lightweight 1D CNN tailored to tabular network-flow features with strong regularization.
- Robustness-ready aggregation via Multi-Krum subset selection to reduce the impact of anomalous updates.
- Clear separation of layers (data, model, trainer, federation) for maintainability and extensibility.

Limitations and risks

- Non-IID client distributions can challenge simple aggregation and slow convergence.
- Without additional defenses, federated systems can remain vulnerable to subtle poisoning attacks.
- Performance depends on consistent feature engineering and normalization across clients.

Future enhancements

- Add more robust aggregators (Trimmed Mean, Coordinate-wise Median) and adaptive norm clipping.
- Introduce privacy layers such as secure aggregation and/or differential privacy.
- Improve calibration (threshold tuning) and track ROC-AUC during federated rounds.
- Package the model with an inference-friendly export and a small CLI for live traffic scoring.

## 7) Appendix (Glossary and quick references)

- Federated Learning (FL): A training paradigm where clients compute updates locally and share model parameters or gradients with a server; raw data never leaves the clients.
- FedAvg: Baseline FL aggregation where client updates are averaged, typically weighted by client sample counts.
- Multi-Krum: A robust aggregation technique that selects updates based on minimal pairwise distances, reducing the influence of outliers.
- Precision/Recall/F1: Standard classification metrics; recall is particularly important for ensuring attacks aren’t missed, while precision limits false alarms.

Repository anchors (no code shown)

- Centralized training: `src/models/trainer.py`
- Model definition: `src/models/cnn_model.py`
- Federated client/server: `client.py`, `server.py`, and `src/federated/flower_client.py`
- Artifacts for slides: `results/` images and JSON history
