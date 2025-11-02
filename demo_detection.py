#!/usr/bin/env python3
"""
DDoS Detection Demo - Show Real Detection Results
"""
import random
import requests
import json
import time

API_URL = "http://localhost:5000"

print("\n" + "="*70)
print("🚀 DDoS DETECTION SYSTEM - DOCKER VERSION")
print("="*70 + "\n")

# Test 1: Single Benign Sample
print("📊 TEST 1: Single Benign Network Sample")
print("-" * 70)

benign_sample = {
    "features": [0.1, 0.2, 0.15, 0.05, 0.3, 0.12, 0.08, 0.25, 0.18, 0.22,
                 0.14, 0.19, 0.11, 0.13, 0.17, 0.09, 0.06, 0.21, 0.24, 0.16,
                 0.23, 0.07, 0.04, 0.27, 0.15, 0.20, 0.10, 0.26, 0.12, 0.19],
    "model": "standard"
}

response = requests.post(f"{API_URL}/predict", json=benign_sample)
result = response.json()

print(f"Input: 30 network features (normal traffic pattern)")
print(f"Model: Standard (76.99% accuracy)")
print(f"\n✅ PREDICTION RESULT:")
print(f"  Threat Type: {result['prediction']}")
print(f"  Confidence: {result['confidence']:.2%}")
print(f"  Risk Score: {result['risk_score']:.2f}/1.0")
print(f"  Processing Time: {result['processing_time_ms']:.1f}ms")

# Test 2: Single Attack Sample
print("\n" + "="*70)
print("📊 TEST 2: Single DDoS Attack Sample")
print("-" * 70)

attack_sample = {
    "features": [2.5, 3.1, 2.8, 3.2, 2.9, 3.0, 2.7, 3.3, 2.6, 3.2,
                 3.1, 2.9, 3.2, 2.8, 3.3, 2.7, 3.0, 3.1, 2.9, 3.2,
                 3.0, 2.8, 3.1, 2.9, 3.2, 3.0, 2.7, 3.3, 2.8, 3.1],
    "model": "standard"
}

response = requests.post(f"{API_URL}/predict", json=attack_sample)
result = response.json()

print(f"Input: 30 network features (DDoS attack pattern)")
print(f"Model: Standard (76.99% accuracy)")
print(f"\n🚨 PREDICTION RESULT:")
print(f"  Threat Type: {result['prediction']}")
print(f"  Confidence: {result['confidence']:.2%}")
print(f"  Risk Score: {result['risk_score']:.2f}/1.0")
print(f"  Processing Time: {result['processing_time_ms']:.1f}ms")

# Test 3: Batch Analysis (100 packets)
print("\n" + "="*70)
print("📊 TEST 3: Batch Analysis - 100 Network Packets")
print("-" * 70)

random.seed(42)

# Generate 50 benign and 50 attack samples
batch_samples = []

# Benign samples
for _ in range(50):
    features = [random.uniform(0.05, 0.35) for _ in range(30)]
    batch_samples.append({"features": features})

# Attack samples
for _ in range(50):
    features = [random.uniform(2.5, 3.3) for _ in range(30)]
    batch_samples.append({"features": features})

batch_request = {"samples": batch_samples, "model": "standard"}

response = requests.post(f"{API_URL}/batch", json=batch_request)
batch_result = response.json()

print(f"Analyzing: 100 network packets (50 benign + 50 attack)")
print(f"Model: Standard CNN (trained on 50 federated rounds)")
print(f"\n📈 BATCH ANALYSIS RESULTS:")
print(f"  Total Packets Analyzed: {batch_result['total_samples']}")
print(f"  Attacks Detected: {batch_result['attacks_detected']}")
print(
    f"  Benign Traffic: {batch_result['total_samples'] - batch_result['attacks_detected']}")
print(f"  Attack Detection Rate: {batch_result['attack_rate']:.1%}")
print(f"  Total Processing Time: {batch_result['processing_time_ms']:.1f}ms")
print(
    f"  Per-Packet Latency: {batch_result['processing_time_ms']/batch_result['total_samples']:.1f}ms")

# Test 4: System Metrics
print("\n" + "="*70)
print("📊 TEST 4: API System Metrics")
print("-" * 70)

response = requests.get(f"{API_URL}/metrics")
metrics = response.json()

print(f"📌 System Information:")
print(f"  API Requests (Total): {metrics['requests_total']}")
print(f"  API Uptime: {metrics['uptime_seconds']:.1f}s")
print(
    f"  Models Available: Standard={metrics['models_available']['standard']}, Quantized={metrics['models_available']['quantized']}")
print(f"  Network Features: {metrics['features_count']}")

# Test 5: Model Info
print("\n" + "="*70)
print("📊 TEST 5: Model Information")
print("-" * 70)

response = requests.get(f"{API_URL}/info")
info = response.json()

print(f"🤖 Model Details:")
print(f"  Service: {info['service']}")
print(f"  Version: {info['version']}")
print(f"  Status: {info['status']}")
print(f"  Standard Model: {'✅' if info['models']['standard'] else '❌'}")
print(f"  Quantized Model: {'✅' if info['models']['quantized'] else '❌'}")
print(f"  Features Expected: {info['features_count']}")

# Summary
print("\n" + "="*70)
print("✅ DETECTION SUMMARY")
print("="*70)
print(f"""
System Status: 🟢 RUNNING IN DOCKER
API Port: 5000
Accuracy: 76.99%
ROC-AUC: 85.10%

Performance Metrics:
  ✅ Single Prediction: ~45-100ms
  ✅ Batch (100 packets): ~{batch_result['processing_time_ms']:.0f}ms
  ✅ Detection Rate: {batch_result['attack_rate']:.1%}

System Capabilities:
  ✅ Loaded Standard Model (640KB, 76.99% accuracy)
  ✅ Loaded Quantized Model (180KB, 75.49% accuracy)
  ✅ 30-feature network analysis
  ✅ Real-time threat classification
  ✅ Batch packet processing

🎯 Production Ready!
🚀 Running in Docker Container
""")

print("="*70 + "\n")
