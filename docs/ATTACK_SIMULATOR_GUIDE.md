# DDoS Attack Simulator - Command Reference

Advanced attack simulation with multi-target and multi-pattern support.

## Basic Usage

```bash
python attack_simulator.py [options]
```

---

## TARGET SELECTION

### Single Target

```bash
python attack_simulator.py --target api              # Detection API (port 5000)
python attack_simulator.py --target fl-server        # FL Server (port 8080)
python attack_simulator.py --target client-0         # Client 0 (port 5001)
python attack_simulator.py --target client-1         # Client 1 (port 5002)
python attack_simulator.py --target client-2         # Client 2 (port 5003)
python attack_simulator.py --target client-3         # Client 3 (port 5004)
```

### Multiple Targets

```bash
# Sequential (one after another)
python attack_simulator.py --target all

# Specific multiple targets
python attack_simulator.py --targets api,client-0,client-1

# Parallel vs Sequential
python attack_simulator.py --targets api,client-2 --sequential
```

### List Available Targets

```bash
python attack_simulator.py --list-targets
```

Output:
```
AVAILABLE TARGETS
================================================================================
  api            → http://localhost:5000/predict     (DDoS Detection API)
  fl-server      → http://localhost:8080/predict     (Federated Learning Server)
  client-0       → http://localhost:5001/predict     (Federated Client 0)
  client-1       → http://localhost:5002/predict     (Federated Client 1)
  client-2       → http://localhost:5003/predict     (Federated Client 2)
  client-3       → http://localhost:5004/predict     (Federated Client 3)
================================================================================
```

---

## ATTACK PATTERNS

### Available Patterns

```bash
python attack_simulator.py --pattern http-flood      # HTTP Request Flood
python attack_simulator.py --pattern syn-flood       # TCP SYN Flood (simulated)
python attack_simulator.py --pattern udp-flood       # UDP Flood (simulated)
python attack_simulator.py --pattern slowloris       # Slowloris Attack
```

### List Available Patterns

```bash
python attack_simulator.py --list-patterns
```

Output:
```
AVAILABLE ATTACK PATTERNS
================================================================================
  http-flood     → HTTP Request Flood (Intensity: high)
  syn-flood      → TCP SYN Flood (simulated) (Intensity: severe)
  udp-flood      → UDP Flood (simulated) (Intensity: high)
  slowloris      → Slowloris Attack (Intensity: medium)
================================================================================
```

---

## ATTACK PARAMETERS

### Core Parameters

#### `--threads N`
Number of concurrent attack threads
- Default: `10`
- Example: `python attack_simulator.py --threads 20`

#### `--packets N`
Number of packets each thread sends
- Default: `20`
- Example: `python attack_simulator.py --packets 50`
- **Total = threads × packets**

#### `--duration N`
How long attack runs (seconds)
- Default: `15`
- Example: `python attack_simulator.py --duration 30`

#### `--benign N`
Benign baseline packets
- Default: `20`
- Example: `python attack_simulator.py --benign 10`

---

## INTENSITY PRESETS

Quick mode using `--intensity`:

```bash
# LIGHT (50 total packets)
python attack_simulator.py --intensity light

# NORMAL (200 total packets - default)
python attack_simulator.py --intensity normal

# HEAVY (1000 total packets)
python attack_simulator.py --intensity heavy

# SEVERE (3000 total packets)
python attack_simulator.py --intensity severe
```

---

## EXECUTION MODES

### Parallel (Default)
Attack multiple targets simultaneously:
```bash
python attack_simulator.py --targets api,client-0,client-1
# All three attacked at the same time
```

### Sequential
Attack targets one after another:
```bash
python attack_simulator.py --targets api,client-0,client-1 --sequential
# api attacked first, then client-0, then client-1
```

---

## FLAGS

### `--skip-benign`
Skip benign baseline, go straight to attack
```bash
python attack_simulator.py --skip-benign
```

### `--help`
Show all available options
```bash
python attack_simulator.py --help
```

---

## REAL-WORLD EXAMPLES

### Example 1: Basic API Test
```bash
python attack_simulator.py
```
**What happens:**
- 20 benign packets baseline
- 10 threads × 20 packets = 200 attack packets
- Results → `results/attack_resilience_test.json`

---

### Example 2: Attack FL Server
```bash
python attack_simulator.py --target fl-server --intensity heavy
```
**What happens:**
- Attacks FL Server (port 8080)
- 20 threads × 50 packets = 1000 total packets
- Shows if FL server can handle attack traffic

---

### Example 3: Multiple Client Attack
```bash
python attack_simulator.py --targets client-0,client-1,client-2 --intensity heavy
```
**What happens:**
- Simultaneously attacks 3 federated clients
- 1000 packets to each client
- Shows distributed resilience

---

### Example 4: Pattern Comparison
```bash
# HTTP Flood attack
python attack_simulator.py --target api --pattern http-flood --intensity heavy

# SYN Flood attack
python attack_simulator.py --target api --pattern syn-flood --intensity heavy

# UDP Flood attack
python attack_simulator.py --target api --pattern udp-flood --intensity heavy
```
**What happens:**
- Compare detection rates for different attack types
- Measure system resilience to each pattern

---

### Example 5: Slowloris Attack on All Targets
```bash
python attack_simulator.py --target all --pattern slowloris --sequential
```
**What happens:**
- Attacks each target with Slowloris pattern
- Sequential (one at a time)
- Reports resilience of each target

---

### Example 6: Custom High-Intensity Attack
```bash
python attack_simulator.py --target client-2 --threads 30 --packets 100 --duration 45
```
**What happens:**
- Attacks Client 2 specifically
- 30 concurrent threads, 100 packets each = 3000 total
- 45 second attack duration
- Shows extreme stress scenario

---

### Example 7: Minimal Baseline Test
```bash
python attack_simulator.py --benign 5 --intensity light
```
**What happens:**
- Only 5 benign packets baseline
- 50 attack packets (light intensity)
- Quick validation test

---

### Example 8: Skip Benign, Direct Attack
```bash
python attack_simulator.py --target fl-server --pattern syn-flood --skip-benign --intensity severe
```
**What happens:**
- No baseline phase
- Immediately attacks FL Server with SYN flood
- 3000 attack packets
- Measures pure attack resilience

---

### Example 9: Sequential All Targets
```bash
python attack_simulator.py --target all --intensity heavy --sequential
```
**What happens:**
- Attacks API first (1000 packets)
- Then attacks FL Server (1000 packets)
- Then attacks Client-0 (1000 packets)
- ... and so on for all clients
- Generates comprehensive resilience report

---

### Example 10: Comprehensive Multi-Pattern Test
```bash
python attack_simulator.py \
  --targets api,fl-server,client-0 \
  --threads 15 \
  --packets 40 \
  --duration 30 \
  --pattern http-flood \
  --sequential
```
**What happens:**
- 600 packets per target (15 × 40)
- 30 second attack duration
- HTTP flood pattern
- Sequential targeting
- Comprehensive test results for all three

---

## ATTACK INTENSITY GUIDE

```
LIGHT ATTACK
  Threads: 5       Packets: 10     Total: 50
  Use: Quick validation, gentle testing

NORMAL ATTACK (DEFAULT)
  Threads: 10      Packets: 20     Total: 200
  Use: Standard resilience test

HEAVY ATTACK
  Threads: 20      Packets: 50     Total: 1000
  Use: Stress testing, performance limits

SEVERE ATTACK
  Threads: 30      Packets: 100    Total: 3000
  Use: Maximum stress, absolute limits
```

---

## OUTPUT & RESULTS

### Console Output Example
```
🔐 FEDERATED DDOS DETECTION - ATTACK RESILIENCE TEST 🔐
================================================================================
Targets: api
Attack Pattern: http-flood
Attack Mode: Parallel
Attack Threads: 20
Packets per Thread: 50
Total Attack Packets: 1000
Attack Duration: 15s
Benign Baseline Packets: 20
================================================================================

🎯 ATTACKING TARGET: API
   Endpoint: http://localhost:5000/predict
   Description: DDoS Detection API
   Pattern: http-flood

✅ SENDING BENIGN TRAFFIC (Baseline)
  Benign Packet 1: Benign (73.2ms)
  Benign Packet 2: Benign (68.9ms)
  ...

🚨 INITIATING DDOS ATTACK SIMULATION
================================================================================
Attack Parameters:
  - Concurrent Threads: 20
  - Requests per Thread: 50
  - Total Attack Packets: 1000
  - Attack Duration: 15s
================================================================================

  [Thread 0] Attack in progress... (122 packets sent, 8.3s remaining)
  [Thread 1] Attack in progress... (256 packets sent, 7.8s remaining)
  ...

📊 ATTACK SIMULATION RESULTS
================================================================================

🚨 ATTACK PHASE STATISTICS:
  Total Attack Packets Sent: 1000
  Successful Deliveries: 1000 ✅
  Detected as Attack: 998/1000 (99.8%)
  Latency Statistics (during attack):
    - Average: 75.23ms
    - Min: 62.14ms
    - Max: 125.67ms
    - Std Dev: 12.45ms

✅ BENIGN PHASE STATISTICS:
  Total Benign Packets Sent: 20
  Successful Deliveries: 20 ✅
  Correctly Identified as Benign: 20/20 (100.0%)

🛡️ SYSTEM RESILIENCE METRICS:
  Health Checks Performed: 30
  System Uptime: 18.45s (maintained during attack)
  API Status: ✅ ALIVE (never crashed)
  Models Loaded: ✅ YES (both standard and quantized)

✅ SYSTEM OBJECTIVE ACHIEVED
  ✅ DDoS Attack Successfully Detected
  ✅ System Remained Operational During Attack
  ✅ API Response Times Within Normal Range
  ✅ No Service Crashes or Downtime
```

### Multi-Target Results
```
================================================================================
📊 AGGREGATE RESULTS - ALL TARGETS
================================================================================

Total Targets Attacked: 3
Total Attack Packets Sent: 3000
Total Detected as Attack: 2985/3000 (99.5%)

  api            → 1000/1000 detected (100.0%)
  fl-server      → 995/1000 detected (99.5%)
  client-0       → 990/1000 detected (99.0%)
```

### JSON Results File
```json
{
  "timestamp": "2025-11-02T14:30:45.123456",
  "command_args": {
    "target": "api",
    "targets": null,
    "pattern": "http-flood",
    "threads": 20,
    "packets": 50,
    "duration": 15,
    "benign": 20,
    "sequential": false
  },
  "targets_attacked": ["api"],
  "attack_pattern": "http-flood",
  "summary": {
    "total_targets": 1,
    "total_attack_packets": 1000,
    "overall_detection_rate": 99.8,
    "api_crashes": 0,
    "status": "RESILIENT"
  }
}
```

---

## TESTING STRATEGIES

### Strategy 1: Baseline Validation
```bash
python attack_simulator.py --list-targets
python attack_simulator.py --list-patterns
python attack_simulator.py --intensity light
```

### Strategy 2: Target Hardening Comparison
```bash
# Test each target individually
python attack_simulator.py --target api --intensity heavy
python attack_simulator.py --target fl-server --intensity heavy
python attack_simulator.py --target client-0 --intensity heavy
python attack_simulator.py --target client-1 --intensity heavy
```

### Strategy 3: Pattern Effectiveness
```bash
# Compare attack patterns on same target
python attack_simulator.py --target api --pattern http-flood --intensity heavy
python attack_simulator.py --target api --pattern syn-flood --intensity heavy
python attack_simulator.py --target api --pattern udp-flood --intensity heavy
python attack_simulator.py --target api --pattern slowloris --intensity heavy
```

### Strategy 4: Distributed System Resilience
```bash
# Attack all targets simultaneously
python attack_simulator.py --target all --intensity heavy
```

### Strategy 5: Full System Stress Test
```bash
# Everything at maximum
python attack_simulator.py \
  --target all \
  --threads 30 \
  --packets 100 \
  --duration 45 \
  --pattern syn-flood \
  --sequential
```

---

## TROUBLESHOOTING

### Error: "API is not running"
```bash
# Start the system first
docker-compose up -d
# or
python api_service.py
```

### Error: "Invalid target"
```bash
# List available targets
python attack_simulator.py --list-targets
```

### Error: "Invalid pattern"
```bash
# List available patterns
python attack_simulator.py --list-patterns
```

### Want to attack non-localhost?
```bash
# Modify TARGET_ENDPOINTS in attack_simulator.py
# Or use environment variables (future enhancement)
```

---

## SUMMARY TABLE

| Scenario | Command |
|----------|---------|
| Basic API test | `python attack_simulator.py` |
| Light test | `python attack_simulator.py --intensity light` |
| Heavy test | `python attack_simulator.py --intensity heavy` |
| Attack FL Server | `python attack_simulator.py --target fl-server` |
| Attack Client 0 | `python attack_simulator.py --target client-0` |
| Attack all targets | `python attack_simulator.py --target all` |
| Multiple targets | `python attack_simulator.py --targets api,client-0,client-1` |
| HTTP flood | `python attack_simulator.py --pattern http-flood` |
| SYN flood | `python attack_simulator.py --pattern syn-flood` |
| UDP flood | `python attack_simulator.py --pattern udp-flood` |
| Slowloris | `python attack_simulator.py --pattern slowloris` |
| Custom load | `python attack_simulator.py --threads 20 --packets 50` |
| Long duration | `python attack_simulator.py --duration 60` |
| Sequential | `python attack_simulator.py --target all --sequential` |
| Skip baseline | `python attack_simulator.py --skip-benign` |
| List targets | `python attack_simulator.py --list-targets` |
| List patterns | `python attack_simulator.py --list-patterns` |
| Show help | `python attack_simulator.py --help` |



