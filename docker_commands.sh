#!/bin/bash
# Docker Quick Commands for Federated Learning System

# ============================================================================
# BUILDING
# ============================================================================

# Build Docker image
docker build -t federated-ddos:latest .

# Build without cache (fresh install)
docker build --no-cache -t federated-ddos:latest .

# ============================================================================
# RUNNING - DOCKER COMPOSE (RECOMMENDED)
# ============================================================================

# Start all containers (server + 4 clients)
docker-compose up -d

# Start with verbose output
docker-compose up

# Stop all containers
docker-compose down

# Stop and remove volumes
docker-compose down -v

# View status of all containers
docker-compose ps

# ============================================================================
# LOGGING & MONITORING
# ============================================================================

# View all logs
docker-compose logs

# View server logs
docker-compose logs fl-server

# View client 0 logs
docker-compose logs fl-client-0

# Real-time logs (tail -f style)
docker-compose logs -f fl-server

# Last 50 lines
docker-compose logs --tail=50 fl-server

# Follow logs for multiple services
docker-compose logs -f fl-server fl-client-0

# ============================================================================
# DEBUGGING & INSPECTION
# ============================================================================

# Connect to server container bash
docker-compose exec fl-server bash

# Connect to client 0 container bash
docker-compose exec fl-client-0 bash

# Run Python in server container
docker-compose exec fl-server python -c "import flwr; print(flwr.__version__)"

# Check if server is healthy
docker-compose exec fl-server curl http://localhost:8080/health

# Test network connectivity between containers
docker-compose exec fl-client-0 ping fl-server

# ============================================================================
# RESOURCE MONITORING
# ============================================================================

# View container resource usage (CPU, memory)
docker stats

# View container resource usage for specific service
docker stats fl-server

# View container processes
docker-compose top fl-server

# ============================================================================
# FILE OPERATIONS
# ============================================================================

# Copy file from container to host
docker cp fl-server:/app/results/metrics.json ./local_metrics.json

# Copy file from host to container
docker cp ./config.json fl-server:/app/config.json

# List files in container
docker-compose exec fl-server ls -la /app/results

# ============================================================================
# CLEANUP
# ============================================================================

# Remove stopped containers
docker container prune

# Remove unused images
docker image prune

# Remove all unused objects (containers, images, volumes, networks)
docker system prune -a

# Remove specific container
docker rm fl-server

# ============================================================================
# NETWORK DEBUGGING
# ============================================================================

# List Docker networks
docker network ls

# Inspect federated network
docker network inspect federated_ddos_detection_federated_network

# View network settings of a container
docker inspect fl-server | grep -A 10 NetworkSettings

# ============================================================================
# VOLUME OPERATIONS
# ============================================================================

# List volumes
docker volume ls

# Inspect volume
docker volume inspect federated_ddos_detection_data

# Remove unused volumes
docker volume prune

# ============================================================================
# ADVANCED - NETWORK SIMULATION
# ============================================================================

# Add 150ms latency to client 3
docker exec fl-client-3 tc qdisc add dev eth0 root netem delay 150ms

# Add 2% packet loss to client 3
docker exec fl-client-3 tc qdisc add dev eth0 root netem loss 2%

# Remove network simulation
docker exec fl-client-3 tc qdisc del dev eth0 root

# View current network conditions
docker exec fl-client-3 tc qdisc show dev eth0

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

# Check if port is in use
lsof -i :8080

# Force remove container
docker rm -f fl-server

# Remove all containers (cleanup)
docker container prune -f

# View Docker system info
docker system info

# Check Docker daemon logs
docker logs daemon  # On systemd systems

# ============================================================================
# USEFUL DOCKER INSPECT QUERIES
# ============================================================================

# Get container IP address
docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' fl-server

# Get container environment variables
docker inspect -f '{{json .Config.Env}}' fl-server

# Get container mounts/volumes
docker inspect -f '{{json .Mounts}}' fl-server

# Get container status
docker inspect -f '{{.State.Status}}' fl-server

# ============================================================================
# ONE-LINERS FOR QUICK OPERATIONS
# ============================================================================

# Start training and follow logs
docker-compose up -d && docker-compose logs -f fl-server

# Check all client logs for errors
for i in {0..3}; do echo "=== Client $i ==="; docker-compose logs fl-client-$i | grep -i error; done

# Get summary of all containers
docker-compose ps | tail -5

# Stop all and cleanup
docker-compose down -v && docker system prune -f

# ============================================================================
# ENVIRONMENT SETUP VERIFICATION
# ============================================================================

# Verify Docker is installed
docker --version

# Verify docker-compose is installed
docker-compose --version

# Verify Docker daemon is running
docker ps

# ============================================================================
# BUILD OPTIMIZATION
# ============================================================================

# Build with progress output
docker build --progress=plain -t federated-ddos:latest .

# Build and see build cache usage
docker build --verbose -t federated-ddos:latest .

# View image layers
docker history federated-ddos:latest

# ============================================================================
# PRODUCTION CHECKS
# ============================================================================

# Health check status
docker-compose ps | grep "Up"

# Verify data volumes are mounted
docker inspect fl-server | grep -A 5 Mounts

# Check if all clients are connected to server
docker-compose logs fl-server | grep "connected"

# View aggregation statistics
docker-compose logs fl-server | tail -20

# ============================================================================
# SCALING OPERATIONS
# ============================================================================

# Scale to multiple replicas (if using Swarm or K8s)
docker service scale fl-client=10

# Update service (for rolling updates)
docker-compose up -d --scale fl-client=6

# ============================================================================
# DATA VALIDATION
# ============================================================================

# Verify data files are accessible from container
docker-compose exec fl-server ls -la /app/data/optimized/clean_partitions/

# Check data file sizes
docker-compose exec fl-server du -h /app/data/

# Verify results are being written
docker-compose exec fl-server ls -lat /app/results/ | head -10

# ============================================================================
# PERFORMANCE ANALYSIS
# ============================================================================

# Measure container startup time
time docker-compose up -d

# Measure training throughput
docker-compose logs fl-server | grep "Round" | wc -l

# Measure communication overhead
docker stats --no-stream | awk '{print $1, $3, $4}'

# ============================================================================
# SECURITY OPERATIONS
# ============================================================================

# Run container with read-only root filesystem
docker-compose run --read-only fl-server bash

# Scan image for vulnerabilities
docker scan federated-ddos:latest

# View running processes in container
docker top fl-server

# ============================================================================
