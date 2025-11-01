# Quick Reference

Minimal, production-focused commands (Windows PowerShell).

## Setup
```powershell
pip install -r requirements.txt
```

## Centralized
```powershell
python train.py
python inference.py
```

## Federated (Flower)
Server:
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
```powershell
# Quantize model to INT8 (TFLite)
python quantization_v2.py

# Async federation simulation
python async_federated.py

# Monitoring snapshot
python monitoring_dashboard.py
```

## Data
- Folder: `data/optimized/clean_partitions/`
- Files: `client_{0..3}_{train,test}.csv`, `selected_features.json`
- Label: `Binary_Label` (0=Benign, 1=Attack)

## Troubleshooting
- Port busy → change `--address`
- Feature mismatch → ensure 30-feature schema from `selected_features.json`
- Low accuracy → increase `--rounds` (server) and `--epochs` (client)
