# 📊 Visual Architecture & Data Flow Diagrams

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FEDERATED DDOS DETECTION SYSTEM                          │
└─────────────────────────────────────────────────────────────────────────────┘

TRAINING PHASE
──────────────────────────────────────────────────────────────────────────────

    Internet/Network               Federated Clients                 Server
    ──────────────                 ────────────────                  ──────
    
    Traffic ──→  Client 0          [Local Training]              [Central Hub]
    (Packets)       │               ├─ Load: 1000 samples
                    │               ├─ 30 Features
                    │               ├─ 3 Epochs
                    │               └─ Train Loss: 0.288
                    │                      │
                    │                      └─ Send Weights →
                    │                                        ↓
                    ├─→ Client 1                    [Multi-Krum Selection]
                    │   [Local Training]           - Compare distances
                    │                               - Select 2 best clients
                    │                               - Average weights
                    ├─→ Client 2                          │
                    │   [Local Training]                  ↓
                    │                             [Global Model v2.0]
                    └─→ Client 3                          │
                        [Local Training]                   └─→ Broadcast
                                                                │
                                                    [Repeat for Rounds 2-3]
                                                                │
                                                                ↓
                                                    [FINAL MODEL ✅]


DETECTION PHASE
──────────────────────────────────────────────────────────────────────────────

Real-time Traffic          Feature Extraction         Model Inference
──────────────────         ──────────────────         ───────────────

Flow Start:                Feature Vector:
├─ SRC IP:192.168.1.100   [f1, f2, f3, ...f30]
├─ DST IP:10.0.0.1              │
├─ SRC Port:54321               │ Normalize:
├─ DST Port:80           (X - mean) / std
├─ Packets: 523                  │
├─ Duration: 0.3s               ↓
├─ Byte Rate: 2.3MB/s    Reshape: (1, 30, 1)
└─ ...                          │
                                ↓
                        [CNN Model (Pre-trained)]
                        ├─ Conv1D(32 filters)
                        ├─ Conv1D(64 filters)
                        ├─ Conv1D(128 filters)
                        ├─ Dense(256)
                        ├─ Dense(128)
                        └─ Dense(1, sigmoid)
                                │
                                ↓
                        Prediction Score: 0.92
                                │
                                ├─ Probability of ATTACK: 92%
                                └─ Decision: 🚨 BLOCK THIS FLOW
```

---

## CNN Architecture Detailed

```
INPUT: (30, 1) features
│
├─ Reshape for CNN (batch_size, 30, 1)
│
┌───────────────────────────────────────────────┐
│  BLOCK 1: Feature Extraction                  │
├───────────────────────────────────────────────┤
│  Conv1D(32 filters, kernel=3)                 │
│    └─ Learns local patterns                   │
│    └─ Output: (batch, 28, 32)                 │
│  BatchNormalization                           │
│    └─ Stabilizes learning                     │
│  MaxPooling1D(2)                              │
│    └─ Reduces dimensions                      │
│    └─ Output: (batch, 14, 32)                 │
│  Dropout(0.25)                                │
│    └─ Prevents overfitting                    │
└───────────────────────────────────────────────┘
          ↓
┌───────────────────────────────────────────────┐
│  BLOCK 2: Pattern Combination                 │
├───────────────────────────────────────────────┤
│  Conv1D(64 filters, kernel=3)                 │
│    └─ Combines learned patterns               │
│    └─ Output: (batch, 12, 64)                 │
│  BatchNormalization                           │
│  MaxPooling1D(2)                              │
│    └─ Output: (batch, 6, 64)                  │
│  Dropout(0.25)                                │
└───────────────────────────────────────────────┘
          ↓
┌───────────────────────────────────────────────┐
│  BLOCK 3: High-Level Feature Learning         │
├───────────────────────────────────────────────┤
│  Conv1D(128 filters, kernel=3)                │
│    └─ Learns high-level anomalies             │
│    └─ Output: (batch, 4, 128)                 │
│  BatchNormalization                           │
│  GlobalMaxPooling1D()                         │
│    └─ Takes max of each feature               │
│    └─ Output: (batch, 128)                    │
└───────────────────────────────────────────────┘
          ↓
┌───────────────────────────────────────────────┐
│  DECISION HEAD: Classification                │
├───────────────────────────────────────────────┤
│  Dense(256)                                   │
│    └─ Combines all learned features           │
│    └─ Output: (batch, 256)                    │
│  BatchNormalization                           │
│  Dropout(0.5)                                 │
│                                               │
│  Dense(128)                                   │
│    └─ Reduces to classification space         │
│    └─ Output: (batch, 128)                    │
│  Dropout(0.3)                                 │
│                                               │
│  Dense(1, sigmoid)                            │
│    └─ Binary classification                   │
│    └─ Output: (batch, 1) ∈ [0, 1]             │
└───────────────────────────────────────────────┘
          ↓
OUTPUT: Prediction Score (0.0-1.0)
    0.0 ← BENIGN    0.5 (threshold)    1.0 → ATTACK
```

---

## Data Flow: Feature Engineering

```
Raw Dataset (78 features)
│
├─ Year = 2017
├─ Month = 7
├─ Day = 11
├─ ...Protocol = TCP (text)...
├─ Source Port = 54321
├─ Destination Port = 80
├─ Sequence Number = 1000000
├─ Acknowledgment = 2000000
├─ TCP Window = 4096
├─ ...78 features total...
└─ Label = "DDoS"
   
   ↓ STEP 1: NUMERIC CONVERSION
   
├─ Factorize text columns (Protocol, Label)
├─ Handle missing values (median imputation)
├─ Remove infinite values
└─ Convert all to float64
   
   ↓ STEP 2: VARIANCE FILTERING
   
Remove columns with variance = 0 (same value everywhere)
78 features → ~75 features
   
   ↓ STEP 3: CORRELATION PRUNING
   
Calculate correlation matrix
Remove highly correlated features (r > 0.95)
Keep only 1 from each correlated pair
~75 features → ~45 features
   
   ↓ STEP 4: MUTUAL INFORMATION RANKING
   
Calculate MI(feature, Binary_Label) for each feature
Score = how much info each feature gives about attack/benign
Sort by MI score (highest = most informative)
   
   ↓ STEP 5: TOP-30 SELECTION
   
Select top 30 by MI score
Save to: selected_features.json
   
   ├─ 1. Flow Duration (MI=0.234)
   ├─ 2. Total Fwd Packets (MI=0.201)
   ├─ 3. Total Backward Packets (MI=0.195)
   ├─ 4. Fwd Packet Length Mean (MI=0.187)
   ├─ 5. Bwd Packet Length Max (MI=0.176)
   ... (25 more)
   └─ 30. Init Win bytes forward (MI=0.034)

FINAL SCHEMA (Enforced for All Clients):
└─ Exactly 30 features in exact order
   └─ All clients use same 30
   └─ New data zero-filled if missing
   └─ Extra features dropped
```

---

## Multi-Krum Aggregation Process

```
ROUND N: Client Updates Received
══════════════════════════════════════════════════════════

4 Clients, Each Sends:
├─ Client 0: Weights_0 (trained on 1000 samples)
├─ Client 1: Weights_1 (trained on 950 samples)
├─ Client 2: Weights_2 (trained on 1050 samples)
└─ Client 3: Weights_3 (trained on 980 samples)

   ↓ STEP 1: FLATTEN WEIGHTS
   
Each weight matrix → Single flat vector:
├─ Client 0: [w0_layer1, w0_layer2, ..., w0_layerN]
├─ Client 1: [w1_layer1, w1_layer2, ..., w1_layerN]
├─ Client 2: [w2_layer1, w2_layer2, ..., w2_layerN]
└─ Client 3: [w3_layer1, w3_layer2, ..., w3_layerN]

   ↓ STEP 2: COMPUTE PAIRWISE DISTANCES
   
For each pair of clients, calculate Euclidean distance:
       
       C0     C1     C2     C3
C0  [  0    1.23   0.98   1.45 ]
C1  [1.23    0     0.87   1.12 ]
C2  [0.98   0.87    0     0.65 ]
C3  [1.45   1.12   0.65    0   ]

   ↓ STEP 3: KRUM SCORING (for f=0, n=4)
   
retained = n - f - 2 = 4 - 0 - 2 = 2

For each client, sum distances to 2 nearest neighbors:
Client 0: min 2 distances = 1.23 + 0.98 = 2.21
Client 1: min 2 distances = 0.87 + 1.12 = 1.99
Client 2: min 2 distances = 0.65 + 0.87 = 1.52 ← BEST
Client 3: min 2 distances = 0.65 + 1.12 = 1.77

   ↓ STEP 4: SELECT TOP-m CLIENTS
   
m = 2 (number to average)
Sort by Krum score:
1. Client 2 (score: 1.52) ← Selected
2. Client 3 (score: 1.77) ← Selected
3. Client 1 (score: 1.99) - Not selected
4. Client 0 (score: 2.21) - Not selected

Why? Clients 2 & 3 have weights closest to other clients
→ Likely honest, not outliers/malicious

   ↓ STEP 5: WEIGHTED AVERAGE
   
For each weight parameter:
w_agg = (n_2 / (n_2 + n_3)) * w_2 + (n_3 / (n_2 + n_3)) * w_3

Where n_i = number of training samples for client i

w_agg = (1050 / 2030) * w_2 + (980 / 2030) * w_3
      = 0.517 * w_2 + 0.483 * w_3

   ↓ STEP 6: SAVE AGGREGATED MODEL
   
Create global model with aggregated weights
Save to: results/federated_global_model.keras

   ↓ STEP 7: BROADCAST TO CLIENTS
   
Send aggregated weights to all 4 clients
Next round starts with these improved weights

RESULT:
├─ Multi-Krum filtered out potential outlier (Client 0)
├─ Balanced average of 2 honest clients
└─ Byzantine fault tolerant (up to f=0 malicious clients)
```

---

## Training Loop Timeline

```
WALL CLOCK TIME          SYSTEM STATE                    USER OBSERVATION
══════════════════════════════════════════════════════════════════════════

T=0.0s
│ ┌─────────────────────────────────────┐
│ │ python server.py --rounds 3         │
│ └─────────────────────────────────────┘
│ [STARTUP PHASE]
│ ├─ Load strategy (MultiKrumFedAvg)
│ ├─ Bind to 0.0.0.0:8080
│ └─ Wait for clients...
│
│ Console Output:
│ > [SERVER] Starting server on 0.0.0.0:8080
│ > [SERVER] Waiting for min 4 clients...

T=1.0s  Client 0 connects
        > [SERVER] Client 0 connected

T=2.0s  Client 1 connects
        > [SERVER] Client 1 connected

T=3.0s  Client 2 connects
        > [SERVER] Client 2 connected

T=4.0s  Client 3 connects
        > [SERVER] Client 3 connected
        > [SERVER] All clients ready! Starting training...

T=5.0s  ┌──────────────────────────────────────┐
        │ ROUND 1/3 BEGINS                     │
        ├──────────────────────────────────────┤
        │ Server broadcasts: "Start fit"       │
        └──────────────────────────────────────┘

T=5-10s [CLIENT LOCAL TRAINING]
        Each client trains locally:
        │ Client 0: Epoch 1/3 [===========] 1000/1000
        │ Client 1: Epoch 1/3 [===========] 950/950
        │ Client 2: Epoch 1/3 [===========] 1050/1050
        │ Client 3: Epoch 1/3 [===========] 980/980
        │
        │ Client 0: Epoch 2/3 [===========] 1000/1000
        │ ... (continues for all epochs)

T=10-15s [WEIGHT SENDING & AGGREGATION]
         Client 0: Sends 2000 weights → Server
         Client 1: Sends 2000 weights → Server
         Client 2: Sends 2000 weights → Server
         Client 3: Sends 2000 weights → Server
                         ↓
         Server Multi-Krum:
         ├─ Calculate distances
         ├─ Select 2 best
         ├─ Average weights
         └─ Save global model

T=15-20s [CLIENT EVALUATION]
         Client 0: Evaluate on 400 test samples
         Client 1: Evaluate on 400 test samples
         Client 2: Evaluate on 400 test samples
         Client 3: Evaluate on 400 test samples
                         ↓
         Server aggregates test results:
         > Avg Test Accuracy: 0.8688

T=20s    ✅ ROUND 1 COMPLETE
         > Round 1 complete in 20 seconds
         > Train Accuracy: 0.8789
         > Test Accuracy: 0.8688
         > Moving to Round 2...

T=20-40s ┌──────────────────────────────────────┐
         │ ROUND 2/3 BEGINS                     │
         ├──────────────────────────────────────┤
         │ [Same process, starting with         │
         │  better weights from Round 1]        │
         └──────────────────────────────────────┘
         Result: Accuracy ↑ to 0.8745

T=40-60s ┌──────────────────────────────────────┐
         │ ROUND 3/3 BEGINS                     │
         ├──────────────────────────────────────┤
         │ [Final round of training]            │
         └──────────────────────────────────────┘
         Result: Final Accuracy = 0.8801

T=60-75s [VISUALIZATION GENERATION]
         ├─ Generate client performance plots
         ├─ Generate confusion matrices
         ├─ Generate ROC curves
         └─ Save 3 essential plots

T=75s    ┌──────────────────────────────────────┐
         │ ✅ TRAINING COMPLETE!                │
         ├──────────────────────────────────────┤
         │ Model saved: federated_global_model  │
         │ Metrics saved: metrics_history       │
         │ Plots saved: federated_analysis/     │
         └──────────────────────────────────────┘

TOTAL TIME: ~75 seconds (from start to finish)
TRAINING TIME: ~60 seconds
OVERHEAD: ~15 seconds (startup, visualization)
```

---

## Decision Threshold Impact

```
THRESHOLD: 0.5 (Default)
═════════════════════════════════════════════════════════

Score Distribution:
BENIGN        UNCERTAIN     ATTACK
(Safe)        (Gray)        (Threat)
│              │              │
0.0 ┼─────────┼0.3────┼0.5────┼0.7────┼1.0
    │         │       │       │       │
    ├─────────┤       ├───────┤       │
    All BENIGN        │       All ATTACK
                      Mixed Results
                      (Decision boundary)

Example Flows:

Flow 1: Score 0.15 → BENIGN (Safe, allow)
Flow 2: Score 0.42 → BENIGN (Probably safe)
Flow 3: Score 0.51 → ATTACK (Probably threat)
Flow 4: Score 0.85 → ATTACK (Definite threat, block)

TUNING THRESHOLD (For Different Scenarios):

┌─────────────┬──────────────┬──────────┬──────────┐
│ Threshold   │ Use Case     │ False+   │ False-   │
├─────────────┼──────────────┼──────────┼──────────┤
│ 0.3         │ Ultra-Safe   │ ↑↑↑ High │ ↓ Low    │
│ 0.4         │ Aggressive   │ ↑↑ Med   │ ↓↓ Low   │
│ 0.5         │ Balanced     │ ↑ Med    │ ↓↓ Med   │
│ 0.6         │ Permissive   │ ↓ Low    │ ↑ Med    │
│ 0.7         │ Very Lenient │ ↓↓ Very  │ ↑↑ High  │
└─────────────┴──────────────┴──────────┴──────────┘

Impact on Metrics:
├─ Lower threshold → Catch more attacks but more false alarms
├─ Higher threshold → Fewer false alarms but miss attacks
└─ 0.5 is balanced
```

---

## System Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│              PRODUCTION DEPLOYMENT                      │
└─────────────────────────────────────────────────────────┘

NETWORK TRAFFIC
       ↓
    [Packet Capture]  ← tcpdump, zeek, suricata
       ↓
┌──────────────────────────────────────┐
│ Feature Extraction Module            │
├──────────────────────────────────────┤
│ • Parse IP, TCP, UDP headers         │
│ • Extract 30 required features       │
│ • Group into flow records            │
│ • Aggregate statistics               │
└──────────────────────────────────────┘
       ↓ (Flow Record)
┌──────────────────────────────────────┐
│ Preprocessing                        │
├──────────────────────────────────────┤
│ • Normalize features                 │
│ • Handle missing values              │
│ • Reshape for CNN input              │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│ CNN Model (federated_global_model)   │
├──────────────────────────────────────┤
│ • Real-time inference                │
│ • Sub-millisecond latency            │
│ • Batch or single predictions        │
└──────────────────────────────────────┘
       ↓ (Score: 0-1)
┌──────────────────────────────────────┐
│ Decision Logic                       │
├──────────────────────────────────────┤
│ if score > 0.5:                      │
│   ├─ Log alert                       │
│   ├─ Send to SIEM                    │
│   ├─ Potentially block               │
│   └─ Extract for forensics           │
│ else:                                │
│   └─ Allow traffic                   │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│ Monitoring & Alerting                │
├──────────────────────────────────────┤
│ • Dashboard (Grafana/Kibana)         │
│ • Real-time alerts (PagerDuty)       │
│ • Metrics collection (Prometheus)    │
│ • Threat intelligence sharing        │
└──────────────────────────────────────┘
       ↓
    Incident Response Team
```

---

**This complete architecture enables real-time DDoS detection with privacy-preserving federated learning!**
