# Attack Simulator - Quick Cheat Sheet

## One-Liners (Copy & Paste)

### Default Test
```bash
python attack_simulator.py
```

### Quick Intensity Tests
```bash
python attack_simulator.py --intensity light    # 50 packets
python attack_simulator.py --intensity normal   # 200 packets (default)
python attack_simulator.py --intensity heavy    # 1000 packets
python attack_simulator.py --intensity severe   # 3000 packets
```

### Target Selection
```bash
# Attack specific targets
python attack_simulator.py --target api              # Detection API (default)
python attack_simulator.py --target fl-server        # FL Server (port 8080)
python attack_simulator.py --target client-0         # Client 0 (port 5001)
python attack_simulator.py --target client-1         # Client 1 (port 5002)
python attack_simulator.py --target client-2         # Client 2 (port 5003)
python attack_simulator.py --target client-3         # Client 3 (port 5004)
python attack_simulator.py --target all              # All targets sequentially

# Multiple targets
python attack_simulator.py --targets api,client-0,client-1
```

### Attack Patterns
```bash
python attack_simulator.py --pattern http-flood    # HTTP Request Flood (default)
python attack_simulator.py --pattern syn-flood     # TCP SYN Flood
python attack_simulator.py --pattern udp-flood     # UDP Flood
python attack_simulator.py --pattern slowloris     # Slowloris Attack
```

### Custom Control
```bash
python attack_simulator.py --threads 5 --packets 30       # 150 packets total
python attack_simulator.py --duration 30                  # 30 second attack
python attack_simulator.py --benign 5                     # 5 benign packets baseline
python attack_simulator.py --skip-benign                  # Skip baseline
python attack_simulator.py --sequential                   # Sequential targets (instead of parallel)
```

### Combination Examples
```bash
# Attack FL Server with heavy intensity
python attack_simulator.py --target fl-server --intensity heavy

# Multiple targets with custom packets
python attack_simulator.py --targets api,client-0,client-2 --threads 15 --packets 40

# UDP Flood attack on Client 1, skip benign baseline
python attack_simulator.py --target client-1 --pattern udp-flood --skip-benign

# Slowloris attack on all clients, sequential
python attack_simulator.py --target all --pattern slowloris --sequential

# Heavy SYN flood on FL Server, 30 second duration
python attack_simulator.py --target fl-server --pattern syn-flood --intensity heavy --duration 30
```

---

## Understanding Attack Strength

```
--threads N      = How many parallel attackers (default: 10)
--packets N      = How many packets each attacker sends (default: 20)

Total Packets = --threads × --packets

Examples:
  --threads 5 --packets 10     = 50 total packets
  --threads 10 --packets 20    = 200 total packets (default)
  --threads 20 --packets 50    = 1000 total packets
  --threads 30 --packets 100   = 3000 total packets
```

---

## Available Targets

| Target | Port | Description |
|--------|------|-------------|
| `api` | 5000 | DDoS Detection API (default) |
| `fl-server` | 8080 | Federated Learning Server |
| `client-0` | 5001 | Federated Client 0 |
| `client-1` | 5002 | Federated Client 1 |
| `client-2` | 5003 | Federated Client 2 |
| `client-3` | 5004 | Federated Client 3 |
| `all` | - | All targets sequentially |

---

## Available Attack Patterns

| Pattern | Description | Intensity |
|---------|-------------|-----------|
| `http-flood` | HTTP Request Flood | High |
| `syn-flood` | TCP SYN Flood (simulated) | Severe |
| `udp-flood` | UDP Flood (simulated) | High |
| `slowloris` | Slowloris Attack | Medium |

---

## What You'll See

```
✅ Benign phase: System sends baseline legitimate packets
🚨 Attack phase: System sends attack packets to target
📊 Results: Shows detection rate, latency, and system health
✅ System Resilience: Proves the system stayed up during attack
```

---

## All Available Options (Reference)

```
TARGETS:
  --target TARGET              Single target (default: api)
  --targets T1,T2,T3          Multiple targets (comma-separated)

PATTERNS:
  --pattern PATTERN            Attack type (default: http-flood)

ATTACK PARAMETERS:
  --threads N                  Concurrent attackers (default: 10)
  --packets N                  Packets per thread (default: 20)
  --duration N                 Attack duration seconds (default: 15)
  --benign N                   Benign baseline packets (default: 20)

FLAGS:
  --intensity PRESET           Quick mode: light/normal/heavy/severe
  --skip-benign                Skip baseline phase
  --sequential                 Attack targets one-by-one (not parallel)
  --list-targets               Show all available targets
  --list-patterns              Show all available patterns
  --help                       Show all options
```

---

## Recommended Testing Plan

### 1. Validate Target Availability
```bash
python attack_simulator.py --list-targets
python attack_simulator.py --list-patterns
```

### 2. Basic API Test
```bash
python attack_simulator.py --intensity light
```

### 3. Multi-Target Test
```bash
python attack_simulator.py --targets api,client-0,client-1 --intensity normal
```

### 4. Pattern Comparison
```bash
python attack_simulator.py --target fl-server --pattern http-flood --intensity heavy
python attack_simulator.py --target fl-server --pattern syn-flood --intensity heavy
python attack_simulator.py --target fl-server --pattern udp-flood --intensity heavy
```

### 5. Full Stress Test
```bash
python attack_simulator.py --target all --intensity severe --duration 30
```

---

## Output Files

Results saved to: `results/attack_resilience_test.json`

Contains:
- All attack parameters (targets, patterns, threads, packets)
- Each packet sent (success/latency/prediction)
- Aggregate results for all targets
- Overall detection rate and system health

---

## Help

```bash
python attack_simulator.py --help
```

Shows full help with all options and examples.

