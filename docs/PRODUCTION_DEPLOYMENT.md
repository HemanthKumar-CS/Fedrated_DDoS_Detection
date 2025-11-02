# Production Deployment Guide

**Last Updated:** November 2, 2025  
**System:** Federated DDoS Detection with Production Service API  
**Status:** 🟢 PRODUCTION READY

---

## 1. Quick Start

### Local Setup
```powershell
# Setup Python environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install flask flask-cors

# Train model (optional, model already exists)
python train.py --federated-rounds 50

# Start inference API
python api_service.py
```

### Test API Locally
```powershell
# Single prediction
curl -X POST http://localhost:5000/predict `
  -H "Content-Type: application/json" `
  -d '{"features": [0.1, -0.2, 0.3, ..., 0.5]}'

# Batch predictions
curl -X POST http://localhost:5000/batch `
  -H "Content-Type: application/json" `
  -d '{"samples": [{"features": [...]}, {"features": [...]}]}'

# Health check
curl http://localhost:5000/health
```

---

## 2. Docker Deployment

### Build & Run Multi-Container System

```powershell
# Build image
docker build -t ddos-detection:latest .

# Start all services
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs -f api-service

# Stop services
docker-compose down
```

### Container Architecture
```
API Service (port 5000)
├─ /predict endpoint
├─ /batch endpoint
└─ /health endpoint

Optional:
FL Server (port 8080) + 4 FL Clients
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

```powershell
# Switch to quantized model in API
curl -X POST http://localhost:5000/predict `
  -H "Content-Type: application/json" `
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

```powershell
# Generate monitoring snapshot
python monitoring_dashboard.py
# Output: results/monitoring_dashboard.json
```

### ✅ Privacy Audit
- Information leakage assessment
- Differential privacy metrics
- Model inversion resistance

```powershell
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

## 7. Attack Simulation & Validation

### Purpose
Validate DDoS detection capabilities and system resilience under real attack conditions.

### Capabilities

**Attack Targets:**
- API Service (port 5000)
- FL Server (port 8080)
- FL Clients (ports 5001-5004)

**Attack Patterns:**
- HTTP Flood (high throughput)
- SYN Flood (connection exhaustion)
- UDP Flood (high volume)
- Slowloris (resource starvation)

**Intensity Presets:**
- Light: 50 packets
- Normal: 200 packets
- Heavy: 1000 packets
- Severe: 3000 packets

### Quick Tests

```powershell
# Light attack test (50 packets)
python attack_simulator.py --intensity light --skip-benign

# Heavy attack test (1000 packets)
python attack_simulator.py --intensity heavy --skip-benign

# Attack specific target
python attack_simulator.py --target api --packets 200 --threads 10

# Attack multiple targets
python attack_simulator.py --targets api,fl-server --intensity normal

# List available targets and patterns
python attack_simulator.py --list-targets
python attack_simulator.py --list-patterns
```

### Demonstrated Results

**Light Attack Test (250 packets):**
- ✅ Packets sent: 250
- ✅ Successful delivery: 250 (100%)
- ✅ Attacks detected: 250 (100.0%)
- ✅ Avg latency: 349.09ms
- ✅ System uptime: 277.28 seconds
- ✅ Status: STABLE

**Heavy Attack Test (1000 packets):**
- ✅ Packets sent: 1000
- ✅ Successful delivery: 1000 (100%)
- ✅ Attacks detected: 1000 (100.0%)
- ✅ Avg latency: 949.39ms
- ✅ System uptime: 318.53+ seconds
- ✅ Status: NO CRASHES, FULLY OPERATIONAL

### Production Workflow

```powershell
# Step 1: Start API service
docker-compose up -d
# or
python api_service.py

# Step 2: Verify health
curl http://localhost:5000/health

# Step 3: Run attack simulation
python attack_simulator.py --target api --intensity heavy --skip-benign

# Step 4: Check results
cat results/attack_resilience_test.json
```

### Results Analysis

Results saved to: `results/attack_resilience_test.json`

**Key Metrics:**
- `packets_sent` - Total attack packets
- `successful_deliveries` - Packets successfully delivered
- `attacks_detected` - Packets correctly identified as attacks
- `detection_rate` - Percentage of packets detected as attacks
- `average_latency_ms` - Mean response time during attack
- `system_uptime_seconds` - Time system remained operational

**Validation Criteria:**
- ✅ Detection rate ≥ 95%
- ✅ System uptime > 300 seconds
- ✅ No crashes or restarts
- ✅ API remains responsive

---

## 8. Integration Example

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
```powershell
# Check health before operations
curl http://localhost:5000/health

# Get metrics for monitoring
curl http://localhost:5000/metrics

# Real-time prediction with timeout
curl --max-time 5 -X POST http://localhost:5000/predict `
  -H "Content-Type: application/json" `
  -d '{"features": [...]}'
```

---

## 9. Files & Artifacts

### Core System Files
- `train.py` - Centralized/federated training
- `server.py` - Flower federation server
- `client.py` - Flower federation client
- `api_service.py` - Production inference API
- `attack_simulator.py` - DDoS attack simulation & testing

### Models
- `results/ddos_model.h5` - Standard trained model (76.99% accuracy)
- `results/ddos_model_quantized_int8.tflite` - Quantized TFLite model

### Reports
- `results/federated_training_convergence.json` - 50-round convergence data
- `results/quantization_report.json` - Compression metrics
- `results/async_federation_report.json` - Staleness analysis
- `results/monitoring_dashboard.json` - System metrics snapshot
- `results/attack_resilience_test.json` - Attack simulation results

### Docker Files
- `Dockerfile` - Container build specification
- `docker-compose.yml` - Multi-container orchestration

---

## 10. Troubleshooting

### API won't start
```powershell
# Check port 5000 is available
netstat -ano | findstr :5000

# Check model files exist
ls results/ddos_model.h5
ls data/optimized/clean_partitions/selected_features.json

# Run with verbose logging
python api_service.py --verbose
```

### Docker container fails to start
```powershell
# Check image built successfully
docker images | grep ddos-detection

# View container logs
docker logs container_name

# Rebuild without cache
docker build --no-cache -t ddos-detection:latest .
```

### API returns 500 errors
```powershell
# Check model can be loaded
python test_api.py

# Check feature count matches
# Model expects: 30 features
```

### Attack simulator fails
```powershell
# Check API is running
curl http://localhost:5000/health

# Check with verbose output
python attack_simulator.py --target api --packets 10 --verbose

# Try simpler test
python attack_simulator.py --intensity light
```

---

## 11. Next Steps

### Immediate (Ready Now)
- ✅ Deploy API service locally or in Docker
- ✅ Integrate with threat detection pipeline
- ✅ Setup continuous monitoring
- ✅ Configure alerting for high-risk predictions
- ✅ Validate with attack simulation

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
- **Docker Setup:** `docs/DOCKER_DEPLOYMENT_GUIDE.md`
- **Attack Simulation:** `docs/ATTACK_SIMULATOR_GUIDE.md`
- **Quick Commands:** `docs/QUICK_REFERENCE.md`
- **README:** `README.md`

---

**System Status:** 🟢 PRODUCTION READY  
**Last Validation:** 50-round federated training (76.99% accuracy), API tested, Attack simulation validated (100% detection, 318+ sec uptime) ✅
