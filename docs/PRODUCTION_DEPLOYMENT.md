# Production Deployment Guide

**Last Updated:** November 2, 2025  
**System:** Federated DDoS Detection with Production Service API  
**Status:** 🟢 PRODUCTION READY

---

## 1. Quick Start

### Local Setup
```bash
# Setup Python environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
pip install flask flask-cors

# Train model (optional, model already exists)
python train.py --federated-rounds 50

# Start inference API
python api_service.py
```

### Test API Locally
```bash
# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, -0.2, 0.3, ..., 0.5]}'  # 30 features

# Batch predictions
curl -X POST http://localhost:5000/batch \
  -H "Content-Type: application/json" \
  -d '{"samples": [{"features": [...]}, {"features": [...]}]}'

# Health check
curl http://localhost:5000/health
```

---

## 2. Docker Deployment

### Build & Run Multi-Container System

```bash
# Build image
docker build -t ddos-detection:latest .

# Start all services (1 FL server + 4 clients + 1 API)
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs -f api-service  # Watch API logs
docker-compose logs -f fl-server    # Watch server logs

# Stop services
docker-compose down
```

### Container Architecture
```
1 FL Server (port 8080)
├─ Client 0 (FL client)
├─ Client 1 (FL client)
├─ Client 2 (FL client)
└─ Client 3 (FL client)

+ API Service (port 5000)
  ├─ /predict endpoint
  ├─ /batch endpoint
  └─ /health endpoint
```

---

## 3. API Endpoints Reference

### `/health` - GET
Health check (always available).
```json
{
  "status": "healthy",
  "timestamp": "2025-11-02T...",
  "models_loaded": true,
  "uptime_seconds": 125.4
}
```

### `/predict` - POST
Single sample prediction.

**Request:**
```json
{
  "features": [f1, f2, ..., f30],
  "model": "standard"
}
```

**Response:**
```json
{
  "prediction": "Attack",
  "confidence": 0.87,
  "risk_score": 0.87,
  "processing_time_ms": 45.2,
  "timestamp": "2025-11-02T..."
}
```

### `/batch` - POST
Multiple sample predictions (efficient for bulk analysis).

**Request:**
```json
{
  "samples": [
    {"features": [f1, f2, ..., f30]},
    {"features": [f1, f2, ..., f30]}
  ],
  "model": "standard"
}
```

**Response:**
```json
{
  "predictions": [
    {"prediction": "Attack", "confidence": 0.87, "risk_score": 0.87},
    {"prediction": "Benign", "confidence": 0.12, "risk_score": 0.12}
  ],
  "total_samples": 2,
  "attacks_detected": 1,
  "attack_rate": 0.5,
  "processing_time_ms": 62.3,
  "timestamp": "2025-11-02T..."
}
```

### `/metrics` - GET
API performance metrics.
```json
{
  "requests_total": 1250,
  "uptime_seconds": 3600,
  "models_available": {
    "standard": true,
    "quantized": true
  },
  "features_count": 30
}
```

### `/info` - GET
Model and system information.
```json
{
  "service": "DDoS Detection API",
  "version": "1.0.0",
  "status": "production",
  "models": {"standard": true, "quantized": true},
  "model_loaded_at": "2025-11-02T..."
}
```

---

## 4. Model Selection

### Standard Model (Recommended)
- **Accuracy:** 76.99%
- **ROC-AUC:** 85.10%
- **Latency:** ~45ms per prediction
- **Size:** 640 KB
- **Use Case:** Production inference, high accuracy

### Quantized Model (8-bit INT8)
- **Accuracy:** 75.49% (1.5% loss)
- **Compression:** 3.55x (0.18 MB)
- **Latency:** ~30ms per prediction
- **Use Case:** Edge devices, low-bandwidth environments

```bash
# Switch to quantized model in API
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [...], "model": "quantized"}'
```

---

## 5. Production Features

### ✅ Federated Learning
- **Clients:** 4 distributed data sources
- **Aggregation:** Multi-Krum (Byzantine-robust)
- **Communication:** gRPC over TLS
- **Architecture:** Flower Framework (industry-standard)

### ✅ Async Updates
- **Staleness Weighting:** Adaptive aggregation
- **Failure Tolerance:** Client dropout handling
- **Convergence:** Tested over 50 rounds

### ✅ Monitoring Dashboard
- System health metrics
- Model performance tracking
- Threat detection alerts
- Real-time statistics

```bash
# Generate monitoring snapshot
python monitoring_dashboard.py
# Output: results/monitoring_dashboard.json
```

### ✅ Privacy Audit
- Information leakage assessment
- Differential privacy metrics
- Model inversion resistance

```bash
# Run privacy audit
python privacy_audit.py
# Output: results/privacy_audit_report.json
```

---

## 6. Performance Baseline

**50-Round Federated Training Results:**
- Accuracy: **76.99%**
- Precision: 77.37%
- Recall: 76.21%
- F1-Score: 76.79%
- ROC-AUC: **85.10%**
- Convergence: STABLE (no degradation)

**Deployment Optimization:**
- Quantization: **3.55x compression** (1.5% accuracy trade-off)
- Async Federation: **0.77s average staleness** (validated)
- API Latency: **~45ms per prediction** (standard model)

---

## 7. Integration Example

### Python Client
```python
import requests
import json

API_URL = "http://localhost:5000"

# Single prediction
features = [0.1, -0.2, 0.3, ..., 0.5]  # 30 features
response = requests.post(
    f"{API_URL}/predict",
    json={"features": features, "model": "standard"}
)
result = response.json()
print(f"Threat Level: {result['prediction']} (confidence: {result['confidence']:.2f})")

# Batch predictions
samples = [{"features": [...]} for _ in range(100)]
response = requests.post(
    f"{API_URL}/batch",
    json={"samples": samples}
)
batch_result = response.json()
print(f"Attacks detected: {batch_result['attacks_detected']}/{batch_result['total_samples']}")
```

### HTTP Integration
```bash
# Check health before operations
curl http://localhost:5000/health | jq .

# Get metrics for monitoring
curl http://localhost:5000/metrics | jq .

# Real-time prediction with timeout
curl --max-time 5 -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [...]}'
```

---

## 8. Files & Artifacts

### Core System Files
- `train.py` - Centralized/federated training
- `server.py` - Flower federation server
- `client.py` - Flower federation client
- `api_service.py` - Production inference API

### Models
- `results/ddos_model.h5` - Standard trained model (76.99% accuracy)
- `results/ddos_model_quantized_int8.tflite` - Quantized TFLite model

### Reports
- `results/federated_training_convergence.json` - 50-round convergence data
- `results/quantization_report.json` - Compression metrics
- `results/async_federation_report.json` - Staleness analysis
- `results/monitoring_dashboard.json` - System metrics snapshot

### Docker Files
- `Dockerfile` - Container build specification
- `docker-compose.yml` - Multi-container orchestration
- `docker_commands.sh` - Docker CLI cheat sheet

---

## 9. Troubleshooting

### API won't start
```bash
# Check port 5000 is available
netstat -an | grep 5000

# Check model files exist
ls -la results/ddos_model.h5
ls -la data/optimized/clean_partitions/selected_features.json

# Run with verbose logging
python api_service.py --verbose
```

### Docker container fails to start
```bash
# Check image built successfully
docker images | grep ddos-detection

# View container logs
docker logs container_name

# Rebuild without cache
docker build --no-cache -t ddos-detection:latest .
```

### API returns 500 errors
```bash
# Check model can be loaded
python test_api.py

# Check feature count matches
# Model expects: 30 features
# API validates: len(features) == 30
```

---

## 10. Next Steps

### Immediate (Ready Now)
- ✅ Deploy API service locally or in Docker
- ✅ Integrate with threat detection pipeline
- ✅ Setup continuous monitoring
- ✅ Configure alerting for high-risk predictions

### Short-term (2-4 weeks)
- Implement continuous model retraining
- Add model versioning & rollback
- Setup centralized logging (ELK/Splunk)
- Performance benchmarking at scale

### Future Enhancements (Optional)
- Differential Privacy (DP-SGD) for formal privacy guarantees
- Model explainability (SHAP/LIME)
- AutoML for hyperparameter optimization
- Multi-GPU federation support

---

## Support & Documentation

- **Architecture Details:** `docs/ARCHITECTURE_DIAGRAMS.md`
- **Technical Deep Dive:** `docs/Master_Documentation.md`
- **Quick Commands:** `docs/QUICK_REFERENCE.md`
- **README:** `README.md`

---

**System Status:** 🟢 PRODUCTION READY  
**Last Validation:** 50-round federated training, 76.99% accuracy, API tested ✅
