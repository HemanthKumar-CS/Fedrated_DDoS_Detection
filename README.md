# Federated DDoS Detection (Production)

Lightweight 1D-CNN for DDoS detection on real data with federated learning (Flower) and robust aggregation (Multi-Krum + FedAvg). Focused, production-ready, minimal docs.

## What’s included
- Real data only (30-feature schema)
- Centralized training and inference
- Federated server/client with Multi-Krum + FedAvg
- 8-bit quantization artifact (TFLite)
- Async update simulation and monitoring dashboard
- Docker compose for multi-container runs

## Setup
```powershell
pip install -r requirements.txt
```

## Data layout
```
data/optimized/clean_partitions/
  selected_features.json          # 30 features
  client_{0..3}_{train,test}.csv  # 30 features + Binary_Label (+ Label)
```

## Train on real data
```powershell
python train.py
```
Artifacts in `results/`: `ddos_model.h5`, `scaler.pkl`, `metrics.json`, `training_results.png`.

## Inference on real data
```powershell
python inference.py
```
Outputs `results/inference_results.json`.

## Federated learning (local)
Server (choose rounds/address):
```powershell
python server.py --rounds 10 --address 127.0.0.1:8080
```
Clients (4 terminals):
```powershell
python client.py --cid 0 --epochs 5
python client.py --cid 1 --epochs 5
python client.py --cid 2 --epochs 5
python client.py --cid 3 --epochs 5
```

## Production extras
- Quantization (dynamic-range INT8): `python quantization_v2.py` → `results/ddos_model_quantized_int8.tflite` + `results/quantization_report.json`
- Async federation simulation: `python async_federated.py` → `results/async_federation_report.json`
- Monitoring snapshot: `python monitoring_dashboard.py` → `results/monitoring_dashboard.json`

## Docker (optional)
- Compose file: `docker-compose.yml`
- Guide: `DOCKER_DEPLOYMENT_GUIDE.md`

## Pointers
- Deep dive: `docs/Master_Documentation.md`
- Quick commands: `docs/QUICK_REFERENCE.md`
- Diagrams: `docs/ARCHITECTURE_DIAGRAMS.md`

## Troubleshooting
- Port busy → change `--address`
- Feature mismatch → ensure `selected_features.json` and CSVs use 30 features
- TensorFlow CPU only → OK; GPU optional

License: MIT
