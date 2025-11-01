# Docker Distributed Deployment Guide

## Overview
This guide explains how to run your federated learning system as a truly distributed system using Docker containers.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Network (bridge)                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Client 0 │  │ Client 1 │  │ Client 2 │  │ Client 3 │    │
│  │Container │  │Container │  │Container │  │Container │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
│       │              │              │              │          │
│       └──────────────┼──────────────┼──────────────┘          │
│                      │              │                         │
│                 ┌────▼──────────────▼────┐                   │
│                 │   FL-Server Container  │                   │
│                 │  (Aggregation Engine)  │                   │
│                 └────────────────────────┘                   │
│                                                               │
│ Shared Volumes: /data, /results                             │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **Docker installed** (version 20.10+)
   ```bash
   docker --version
   docker-compose --version
   ```

2. **Project files in place**
   - `Dockerfile` - Container image definition
   - `docker-compose.yml` - Multi-container orchestration
   - `requirements_prod.txt` - Python dependencies
   - Data files in `data/optimized/clean_partitions/`

## Quick Start - 3 Commands

### Step 1: Build Docker Image
```bash
docker build -t federated-ddos:latest .
```
*Takes ~2-5 minutes first time (installs all dependencies)*

### Step 2: Start Distributed System
```bash
docker-compose up -d
```
*Launches 1 server + 4 client containers*

### Step 3: Monitor Training
```bash
docker-compose logs -f fl-server
```

---

## Detailed Usage

### Building the Docker Image

```bash
# Build with progress output
docker build -t federated-ddos:latest .

# Build without cache (fresh install)
docker build --no-cache -t federated-ddos:latest .

# Build with custom tag
docker build -t federated-ddos:v1.0 .
```

### Running Distributed Training

#### Option 1: Using docker-compose (Recommended)

**Start all containers:**
```bash
docker-compose up -d
```

**View logs:**
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f fl-server
docker-compose logs -f fl-client-0

# Last 50 lines
docker-compose logs --tail=50 fl-server
```

**Stop all containers:**
```bash
docker-compose down
```

**Remove containers and volumes:**
```bash
docker-compose down -v
```

---

#### Option 2: Manual Docker Run (For Testing)

**Start Server:**
```bash
docker run -d \
  --name fl-server \
  --network federated_network \
  -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  federated-ddos:latest \
  python server.py
```

**Start Client 0:**
```bash
docker run -d \
  --name fl-client-0 \
  --network federated_network \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  federated-ddos:latest \
  python client.py --client-id 0
```

---

## Understanding the Distributed Setup

### What Happens With Docker-Compose?

1. **Network Creation** (`federated_network`)
   - Isolated Docker network bridge
   - All containers can communicate by service name
   - Server: `fl-server:8080`
   - Client 0: `fl-client-0:8080` (internal only)

2. **Volume Sharing**
   - `/app/data` → Shared from `./data` (host)
   - `/app/results` → Shared from `./results` (host)
   - All containers see the same data files

3. **Container Startup Sequence**
   ```
   fl-server starts first
   ↓ (health check passes)
   fl-client-0 connects to server
   fl-client-1 connects to server
   fl-client-2 connects to server
   fl-client-3 connects to server
   ↓
   Federated training begins
   ```

4. **Data Distribution**
   - Each client loads its own data partition:
     - Client 0: `data/optimized/clean_partitions/client_0_train.csv`
     - Client 1: `data/optimized/clean_partitions/client_1_train.csv`
     - Client 2: `data/optimized/clean_partitions/client_2_train.csv`
     - Client 3: `data/optimized/clean_partitions/client_3_train.csv`
   
   - Server aggregates updates via Multi-Krum

---

## Production Configuration

### Environment Variables

Add to `docker-compose.yml` environment section:

```yaml
environment:
  - SERVER_ADDRESS=fl-server:8080
  - LOG_LEVEL=INFO           # DEBUG, INFO, WARNING, ERROR
  - CLIENTS_PER_ROUND=4      # How many clients per round
  - NUM_ROUNDS=10            # Total training rounds
  - BATCH_SIZE=32            # Training batch size
  - EPOCHS=5                 # Local epochs per round
```

### Resource Limits

Add to each service in `docker-compose.yml`:

```yaml
  fl-server:
    # ... existing config ...
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### Persistent Storage

For production, use named volumes:

```yaml
volumes:
  fl_data:
    driver: local
  fl_results:
    driver: local

services:
  fl-server:
    volumes:
      - fl_data:/app/data
      - fl_results:/app/results
```

---

## Monitoring & Debugging

### Check Container Status
```bash
docker-compose ps
```

Output:
```
NAME              STATUS      PORTS
fl-server         Up 2 min    0.0.0.0:8080->8080/tcp
fl-client-0       Up 1 min    
fl-client-1       Up 1 min    
fl-client-2       Up 1 min    
fl-client-3       Up 1 min    
```

### View Container Logs
```bash
# Server logs
docker logs fl-server

# Client logs
docker logs fl-client-0

# Real-time logs
docker logs -f fl-server

# Last N lines
docker logs --tail 100 fl-server
```

### Connect to Running Container
```bash
docker exec -it fl-server bash

# Inside container, run Python
python
>>> import flwr
>>> print(flwr.__version__)
```

### Check Network Connectivity
```bash
# From one container, ping another
docker exec fl-client-0 ping fl-server

# Expected output: PONG (connection OK)
```

---

## Network Simulation in Docker

Docker containers naturally introduce real network effects:

1. **Latency** - Inter-container communication overhead
2. **Packet Loss** - Can be simulated with tc (traffic control)
3. **Bandwidth Limits** - Can be set per container

### Simulate Network Conditions

To add latency to Client 3 (simulating slow edge device):

```bash
docker exec fl-client-3 tc qdisc add dev eth0 root netem delay 150ms jitter 15ms
```

To simulate 2% packet loss:

```bash
docker exec fl-client-3 tc qdisc add dev eth0 root netem loss 2%
```

---

## Troubleshooting

### Issue: Containers won't start

```bash
# Check if port 8080 is already in use
lsof -i :8080

# Check Docker daemon
docker ps

# View detailed error
docker-compose logs fl-server
```

### Issue: Clients can't connect to server

```bash
# Test network connectivity
docker-compose exec fl-client-0 ping fl-server

# Check server is listening
docker-compose exec fl-server netstat -tlnp | grep 8080
```

### Issue: Data files not found

```bash
# Verify volumes are mounted
docker inspect fl-server | grep Mounts

# Check file exists
ls -la data/optimized/clean_partitions/
```

### Issue: Out of disk space

```bash
# Clean up stopped containers
docker container prune

# Remove unused images
docker image prune

# Remove all unused data
docker system prune -a
```

---

## Performance Comparison

### Single Machine (Current)
```
Training Time: 2-3 hours
Communication: All internal (fast)
Bottleneck: Single CPU/GPU
Network Latency: 0.1ms
```

### Docker Distributed (This Setup)
```
Training Time: Similar (can parallelize)
Communication: Inter-container (1-5ms)
Bottleneck: Largest client or server
Network Latency: 1-5ms (realistic)
Scalability: Add more clients easily
```

### Multi-Machine (Production)
```
Training Time: Similar or faster
Communication: Network (10-150ms)
Bottleneck: Slowest network link
Network Latency: 10-150ms (real-world)
Scalability: Geographic distribution
```

---

## Scaling Beyond 4 Clients

To add more clients (e.g., Client 4, 5, 6), add to `docker-compose.yml`:

```yaml
  fl-client-4:
    build: .
    container_name: fl-client-4
    command: python client.py --client-id 4
    depends_on:
      fl-server:
        condition: service_healthy
    volumes:
      - ./data:/app/data
      - ./results:/app/results
    networks:
      - federated_network
    environment:
      - CLIENT_ID=4
      - SERVER_ADDRESS=fl-server:8080
```

Then ensure data files exist:
```bash
# Copy client 4 data (or generate)
cp data/optimized/clean_partitions/client_0_*.csv \
   data/optimized/clean_partitions/client_4_*.csv
```

---

## Production Deployment

### Kubernetes (K8s)

For production, consider Kubernetes:

1. Create StatefulSet for clients
2. Create Deployment for server
3. Use Services for communication
4. Monitor with Prometheus/Grafana

Example Kubernetes manifest:
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: fl-clients
spec:
  serviceName: fl-clients
  replicas: 4
  selector:
    matchLabels:
      app: fl-client
  template:
    metadata:
      labels:
        app: fl-client
    spec:
      containers:
      - name: fl-client
        image: federated-ddos:latest
        env:
        - name: SERVER_ADDRESS
          value: fl-server:8080
```

---

## Summary

| Aspect | Single Machine | Docker | Production |
|--------|---|---|---|
| **Setup Time** | 10 min | 5 min | 1-2 hours |
| **Clients** | 1 process | 4 containers | 100+ nodes |
| **Communication** | Memory (~0.1ms) | Network (~1-5ms) | Real network (10-150ms) |
| **Scalability** | Limited | Easy | Excellent |
| **Monitoring** | Basic | docker logs | Prometheus |
| **Cost** | Your machine | Shared resources | Cloud provider |

---

## Next Steps

1. **Build image:** `docker build -t federated-ddos:latest .`
2. **Start system:** `docker-compose up -d`
3. **Monitor:** `docker-compose logs -f`
4. **Check results:** `ls results/`
5. **Stop system:** `docker-compose down`

---

*For questions or issues, check Docker logs and verify data files exist in `data/optimized/clean_partitions/`*
