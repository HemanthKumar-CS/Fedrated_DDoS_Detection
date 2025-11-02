FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir flask flask-cors

# Copy application code
COPY . .

# Create results directory if not exists
RUN mkdir -p results data

# Expose ports
EXPOSE 8080 5000

# Default command
CMD ["python", "server.py"]
