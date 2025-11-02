# Docker Deployment Guide

## Overview

This guide explains the Docker setup for the DDoS detection system. The current deployment focuses on **API Service** (production inference) with **optional** federated learning server.

## Current Architecture

```
┌────────────────────────────────────────────────────────┐
│                 Docker Container                       │
│  ┌─────────────────────────────────────────────────┐  │
│  │  API Service (Flask)                             │  │
│  │  ├─ DDoS Detection Model (TensorFlow)            │  │
│  │  ├─ Health Endpoint: GET /health                 │  │
│  │  ├─ Predict Endpoint: POST /predict              │  │
│  │  └─ Batch Endpoint: POST /batch                  │  │
│  └─────────────────────────────────────────────────┘  │
│                       ↑↓                               │
│              Port 5000 (Mapped from Host)             │
└────────────────────────────────────────────────────────┘

Optional: FL Server for Federated Learning (port 8080)
Optional: FL Clients (ports 5001-5004) for edge training
```

## Prerequisites

1. **Docker installed**
   ```powershell
   docker --version
   docker-compose --version
   ```

2. **Project structure intact**
   - `Dockerfile` - Container image definition
   - `docker-compose.yml` - Multi-container orchestration
   - `requirements_prod.txt` - Production dependencies
   - `api_service.py` - Flask API service
   - `results/ddos_model.h5` - Trained DDoS detection model

## Quick Start - 3 Commands

### Step 1: Build Docker Image
```powershell
docker build -t ddos-detection:latest .
```
*Takes 2-5 minutes first time (installs TensorFlow, Flask, etc.)*

### Step 2: Start Service
```powershell
docker-compose up -d
```
*Launches API service container*

### Step 3: Verify Health
```powershell
curl http://localhost:5000/health
```
*Expected response: `{"status": "healthy", ...}`*

---

## Detailed Usage

### Building the Docker Image

```powershell
# Standard build
docker build -t ddos-detection:latest .

# Build without cache (fresh install)
docker build --no-cache -t ddos-detection:latest .

# Build with custom tag
docker build -t ddos-detection:v1.0 .
```

### Running with Docker-Compose

#### Start All Services
```powershell
docker-compose up -d
```

#### Check Status
```powershell
docker-compose ps
docker ps
```

Output should show:
```
NAME              STATUS      PORTS
api-service       Up 1 min    0.0.0.0:5000->5000/tcp
```

#### View Logs
```powershell
# All logs
docker-compose logs -f

# API service only
docker-compose logs -f api-service

# Last 50 lines
docker-compose logs --tail=50 api-service
```

#### Stop Services
```powershell
# Stop but keep containers
docker-compose stop

# Stop and remove containers
docker-compose down

# Remove containers and volumes
docker-compose down -v
```

---

## API Service Usage

### Health Check
```powershell
curl http://localhost:5000/health
```

### Single Prediction
```powershell
$payload = @{
    features = @(6, 45000, 1500, 500, 1000, 250, 100, 50, 25, 10, 5, 2, 1, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
} | ConvertTo-Json

curl -X POST http://localhost:5000/predict `
  -H "Content-Type: application/json" `
  -Body $payload
```

### Batch Predictions
```powershell
$payload = @{
    samples = @(
        @(6, 45000, 1500, 500, 1000, 250, 100, 50, 25, 10, 5, 2, 1, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        @(6, 45000, 1500, 500, 1000, 250, 100, 50, 25, 10, 5, 2, 1, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    )
} | ConvertTo-Json

curl -X POST http://localhost:5000/batch `
  -H "Content-Type: application/json" `
  -Body $payload
```

---

## Attack Simulation with Docker

Once the API is running in Docker, test it with the attack simulator:

### Simple Attack Test
```powershell
# Verify API is running
python attack_simulator.py --target api --packets 50 --threads 5 --skip-benign
```

### Heavy Load Test
```powershell
# Stress test with 1000 packets
python attack_simulator.py --target api --intensity heavy --skip-benign
```

### Attack Multiple Targets (if FL server enabled)
```powershell
# Attack both API and FL server
python attack_simulator.py --targets api,fl-server --intensity normal
```

Results saved to: `results/attack_resilience_test.json`

---

## Optional: Federated Learning Setup

The current system is simplified to focus on **API inference**. However, FL capabilities are available.

### Enable Federated Learning

To also run FL server + clients, ensure `docker-compose.yml` includes:

```yaml
  fl-server:
    build: .
    container_name: fl-server
    command: python server.py --rounds 10 --address 0.0.0.0:8080
    ports:
      - "8080:8080"
    volumes:
      - ./data:/app/data
      - ./results:/app/results
    networks:
      - federated_network

  fl-client-0:
    build: .
    container_name: fl-client-0
    command: python client.py --cid 0 --epochs 5
    depends_on:
      - fl-server
    volumes:
      - ./data:/app/data
      - ./results:/app/results
    networks:
      - federated_network
```

Then start all:
```powershell
docker-compose up -d
```

---

## Volume Management

### Docker Volumes

```powershell
# List volumes
docker volume ls

# Inspect specific volume
docker volume inspect federated_ddos_detection_data

# Remove unused volumes
docker volume prune
```

### Host Mount Points

The system mounts from host filesystem:

- `./data` → Container `/app/data`
- `./results` → Container `/app/results`

Verify mounts:
```powershell
docker inspect api-service | findstr -A 10 "Mounts"
```

---

## Network Configuration

### Port Mapping

| Service | Container Port | Host Port | Purpose |
|---------|---|---|---|
| API Service | 5000 | 5000 | REST API inference |
| FL Server | 8080 | 8080 | Federated learning aggregation |
| FL Client 0 | 5001 | (internal) | Edge training |
| FL Client 1 | 5002 | (internal) | Edge training |
| FL Client 2 | 5003 | (internal) | Edge training |
| FL Client 3 | 5004 | (internal) | Edge training |

### Custom Port Mapping

To run API on different port:

```yaml
  api-service:
    ports:
      - "8000:5000"  # Host port 8000 → Container port 5000
```

Then access at: `http://localhost:8000`

---

## Monitoring & Debugging

### Check Container Status
```powershell
docker-compose ps
```

### View Resource Usage
```powershell
# All containers
docker stats

# Specific container
docker stats api-service
```

### Connect to Running Container
```powershell
# Open shell
docker exec -it api-service powershell

# Or bash (if available)
docker exec -it api-service bash

# Run Python commands
docker exec api-service python -c "import tensorflow; print(tensorflow.__version__)"
```

### Test API from Container
```powershell
docker exec api-service curl http://localhost:5000/health
```

---

## Troubleshooting

### Issue: Container fails to start

```powershell
# View error logs
docker-compose logs api-service

# Try rebuilding
docker-compose down
docker build --no-cache -t ddos-detection:latest .
docker-compose up
```

### Issue: Port already in use

```powershell
# Find what's using port 5000
netstat -ano | findstr :5000

# Change port in docker-compose.yml or stop conflicting container
```

### Issue: API not responding

```powershell
# Check container is running
docker ps | findstr api-service

# Test connectivity
curl http://localhost:5000/health

# View logs
docker logs api-service
```

### Issue: Model not found

```powershell
# Verify results directory
ls results/

# Ensure ddos_model.h5 exists
docker exec api-service ls -la /app/results/ddos_model.h5
```

### Issue: Out of disk space

```powershell
# Clean up stopped containers
docker container prune

# Remove unused images
docker image prune

# Full cleanup (dangerous - removes all unused resources)
docker system prune -a
```

---

## Performance Optimization

### Resource Limits

Add to `docker-compose.yml`:

```yaml
  api-service:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

### Environment Variables

```yaml
  api-service:
    environment:
      - TF_CPP_MIN_LOG_LEVEL=3  # Reduce TensorFlow logging
      - FLASK_ENV=production
      - THREADS=4               # API worker threads
```

---

## Production Deployment

### Multi-Stage Build (Optimized)

```dockerfile
# Build stage
FROM python:3.12-slim as builder
WORKDIR /app
COPY requirements_prod.txt .
RUN pip install --user -r requirements_prod.txt

# Runtime stage
FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
EXPOSE 5000
CMD ["python", "api_service.py"]
```

### Docker Swarm Deployment

```bash
# Initialize swarm
docker swarm init

# Deploy service
docker service create --name api-service \
  --publish 5000:5000 \
  --replicas 3 \
  ddos-detection:latest python api_service.py
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-service
  template:
    metadata:
      labels:
        app: api-service
    spec:
      containers:
      - name: api-service
        image: ddos-detection:latest
        ports:
        - containerPort: 5000
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 10
          periodSeconds: 5
```

---

## Summary

| Aspect | Local | Docker | K8s |
|--------|-------|--------|-----|
| **Setup Time** | 5 min | 2 min | 20 min |
| **Scalability** | Limited | Easy | Excellent |
| **Isolation** | None | Full | Full + Orchestration |
| **Resource Management** | Manual | Automatic | Advanced |
| **Monitoring** | Basic | Good | Excellent |

---

## Next Steps

1. **Build:** `docker build -t ddos-detection:latest .`
2. **Start:** `docker-compose up -d`
3. **Test:** `curl http://localhost:5000/health`
4. **Simulate:** `python attack_simulator.py --target api --intensity heavy`
5. **Monitor:** `docker-compose logs -f`
6. **Stop:** `docker-compose down`

---

*For more details on attack simulation, see `ATTACK_SIMULATOR_GUIDE.md`*
*For deployment troubleshooting, check Docker logs with `docker-compose logs`*
