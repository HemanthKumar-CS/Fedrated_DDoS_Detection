# Federated DDoS Detection — Master Documentation

Authoritative technical guide (concise). Aligns to real data and current code.

## Overview
- Binary classification: Benign (0) vs Attack (1)
- Input: 30 optimized tabular features (see `selected_features.json`)
- Modes: centralized training, federated learning (Flower)
- Robust aggregation: Multi-Krum subset selection with FedAvg

## Key files
- Model: `src/models/cnn_model.py`
- Data: `src/data/data_loader.py`, `src/data/preprocessing.py`
- Federated: `server.py`, `client.py`
- Training/Inference: `train.py`, `inference.py`
- Extras: `quantization_v2.py`, `async_federated.py`, `monitoring_dashboard.py`

## Data pipeline
- Location: `data/optimized/clean_partitions/`
- Files: `client_{0..3}_{train,test}.csv`, `selected_features.json`
- Enforced 30-feature order; label column: `Binary_Label`

## Run
Centralized:
```powershell
python train.py
python inference.py
```
Federated:
```powershell
python server.py --rounds 10 --address 127.0.0.1:8080
python client.py --cid 0 --epochs 5
python client.py --cid 1 --epochs 5
python client.py --cid 2 --epochs 5
python client.py --cid 3 --epochs 5
```

## Model
- Input shape: (None, 30, 1)
- Backbone: Conv1D blocks → pooling → global pooling
- Head: Dense(128→64) + Dropout → Dense(1, sigmoid)
- Loss/opt: BCE + Adam; Metrics: accuracy (others computed in scripts)

## Robust aggregation (server)
- Multi-Krum: compute pairwise distances of updates, score by sum to nearest neighbors, select subset
- Safety: fallback to FedAvg if n < 2f + 3 or instability detected
- Logs selection/fallback events; integrates with round metrics

## Artifacts (`results/`)
- Centralized: `ddos_model.h5`, `scaler.pkl`, `metrics.json`, `training_results.png`
- Inference: `inference_results.json`
- Production extras: `ddos_model_quantized_int8.tflite`, `quantization_report.json`, `async_federation_report.json`, `monitoring_dashboard.json`

## Docker
- Compose: `docker-compose.yml`
- Guide: `DOCKER_DEPLOYMENT_GUIDE.md`

## Notes and troubleshooting
- Ensure 30-feature schema alignment across clients
- Start server before clients; adjust `--address` when port busy
- CPU-only TF works; GPU optional

## Roadmap (short)
- Add DP-SGD and secure aggregation
- Additional robust aggregators (Trimmed Mean, Median)
- Extended round metrics and monitoring hooks
