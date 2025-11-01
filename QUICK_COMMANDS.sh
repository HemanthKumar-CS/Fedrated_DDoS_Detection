#!/bin/bash
# Quick Commands for DDoS Detection System

# 1. SETUP
python -m venv venv
venv\Scripts\Activate
pip install -r requirements_prod.txt

# 2. TRAIN MODEL (production, real data only)
venv\Scripts\python train.py

# 3. RUN INFERENCE (test on real data)
venv\Scripts\python inference.py

# 4. CHECK RESULTS
cat results\metrics.json          # Training metrics
cat results\inference_results.json # Per-client inference

# 5. VIEW TRAINING VISUALIZATION
# Open results\training_results.png in image viewer

# 6. OPTIONAL: RUN FEDERATED LEARNING
# Terminal 1:
venv\Scripts\python server.py --rounds 5

# Terminal 2-5 (separate terminals):
venv\Scripts\python client.py --cid 0
venv\Scripts\python client.py --cid 1
venv\Scripts\python client.py --cid 2
venv\Scripts\python client.py --cid 3
