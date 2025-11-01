# ===============================
# Base Image
# ===============================
FROM ubuntu:latest

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# ===============================
# System Dependencies
# ===============================
RUN apt update && \
    apt install -y \
        iputils-ping \
        netcat-openbsd \
        python3 \
        python3-pip \
        curl \
        iproute2 \
        tcpdump && \
    pip3 install --break-system-packages scapy && \
    rm -rf /var/lib/apt/lists/*

# ===============================
# Working Directory
# ===============================
WORKDIR /app

# ===============================
# Python Dependencies
# ===============================
# Copy requirements file into the container
COPY requirements.txt .

# Install all Python dependencies (with override for Ubuntu-managed Python)
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# ===============================
# Default Command
# ===============================
CMD ["/bin/bash"]
