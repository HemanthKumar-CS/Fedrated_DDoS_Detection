# Production DDoS Detection API - Docker Image
# Optimized for inference with Flask REST API

FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy production requirements
COPY requirements_prod.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_prod.txt

# Copy application code
COPY api_service.py .
COPY dashboard.html .

# Copy trained model and data
COPY results/ ./results/
COPY data/ ./data/
COPY src/ ./src/

# Expose API port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=10s --timeout=5s --retries=3 --start-period=10s \
    CMD curl -f http://localhost:5000/health || exit 1

# Run API service
CMD ["python", "api_service.py"]
