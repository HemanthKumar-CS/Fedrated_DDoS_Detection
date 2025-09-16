# Federated DDoS Detection

Lightweight 1D-CNN for DDoS detection with both centralized and federated training using Flower. This README is intentionally concise—see the Master Documentation for full details.

## Quickstart (Windows PowerShell)

1. Create and activate env

- python -m venv .venv
- .\.venv\Scripts\Activate.ps1

2. Install

- pip install -r requirements.txt

3. Centralized baseline

- python train_centralized.py --data_dir data/optimized/clean_partitions --epochs 25

4. Federated (1 server + 4 clients)

- Server: python server.py --rounds 5 --address 127.0.0.1:8080
- Clients (open 4 terminals):
  - python client.py --cid 0 --data_dir data/optimized/clean_partitions
  - python client.py --cid 1 --data_dir data/optimized/clean_partitions
  - python client.py --cid 2 --data_dir data/optimized/clean_partitions
  - python client.py --cid 3 --data_dir data/optimized/clean_partitions

Artifacts and metrics are saved to `results/`.

## Documentation

- Master Doc: docs/Master_Documentation.md
- Presentation: docs/Professional_Presentation_Document.md

## Notes

- Data used by default: `data/optimized/clean_partitions/`
- Optional warm start: `--initial_model results/best_enhanced_model.keras`

# Install dependencies
