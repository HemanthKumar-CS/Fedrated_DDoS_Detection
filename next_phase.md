# Next Phase: Robust Federated DDoS Detection Deployment (Server + 4 Clients with Poisoning Defenses)

This document defines a concrete, sequential plan to: (1) stand up a central federated server first, (2) connect 4 client nodes that train locally and send weights, (3) defend against model/data poisoning (malicious weight updates) using robust aggregation (Krum, Multi‑Krum, Trimmed Mean, Coordinate‑wise Median, Norm Clipping + Outlier Filtering), (4) containerize with Docker, orchestrate with Kubernetes, (5) capture/visualize traffic (Wireshark / tshark), (6) integrate live inference & traffic blocking demo on a Kali Linux CLI, and (7) provide optional visual simulation aids.

---
## Phase 0 – Clarify Target Outcomes
- Federated topology: 1 server + 4 clients (extensible to N) using Flower.
- Robust aggregation pipeline BEFORE updating global model.
- Detect & discard (or down‑weight) poisoned updates.
- Provide reproducible infra: Docker images + Kubernetes manifests.
- Runtime visibility: logs, metrics, network captures, aggregation decisions.
- Demonstration on Kali Linux terminal: run inference on sample network flows and (simulated) block suspicious traffic.

---
## Phase 1 – Codebase Enhancements (Robust Aggregation & Security Hooks)
### 1.1 Add Robust Aggregation Strategies
Implement a new strategy module (e.g. `src/federated/robust_strategies.py`) containing:
- Weight extraction helper: convert `FitRes` to flat tensors.
- Utility: compute pairwise Euclidean distances between client updates.
- Krum / Multi‑Krum:
  1. For each update i, compute distances to others, sort, sum the closest (n - f - 2) distances (where f = assumed max Byzantine clients), score = sum.
  2. Krum selects single update with minimal score.
  3. Multi‑Krum selects m candidate updates (score smallest) then average them.
- Trimmed Mean:
  - For each parameter dimension: sort values across clients; trim top/bottom k; average middle slice.
- Coordinate‑wise Median:
  - Median across clients per coordinate.
- Norm Clipping + Outlier Filter:
  - Compute update norms; clip above threshold (e.g. median_norm * factor).
  - Optional z‑score or MAD-based anomaly scoring; drop severe outliers.
- Hybrid pipeline example (recommended):
  1. Convert weights -> updates (w_i - w_global_prev).
  2. Norm clip & basic sanity (NaN/Inf) removal.
  3. Score anomalies (distance from geometric median or centroid); filter if > threshold.
  4. Apply chosen robust aggregator (e.g. Multi‑Krum when #clients >= 5, else Trimmed Mean / Median fallback).
  5. Reconstruct new global weights.

### 1.2 Integration into Server
- Replace current `DDoSFederatedStrategy.aggregate_fit` logic: after receiving results, pass updates through the pipeline above before calling base aggregation.
- Add config flags/env vars to select: `AGGREGATOR={fedavg,krum,multikrum,trimmed_mean,median,hybrid}` plus `MAX_BYZANTINE=f` and `TRIM_K=k`.
- Log decisions: accepted clients, removed clients, anomaly scores, final aggregator type.
- Persist a JSON report per round (e.g. `results/robust_round_logs.jsonl`).

### 1.3 Defensive Validation of Updates
For each client update:
- Check shapes match expected model architecture.
- Check numeric sanity: no NaN/Inf, variance not zero for all layers, norm within allowed band.
- Optionally apply differential privacy style clipping (global norm bound).

### 1.4 Inference + Traffic Blocking Hook
Create `src/runtime/traffic_guard.py`:
- Load latest global model weights.
- Expose function: `classify(packet_features) -> benign|ddos, confidence`.
- Provide a CLI script `ddos_guard_cli.py` to stream CSV (simulated flows) and issue `block` commands.
- Blocking simulation (demo friendly):
  - Linux: simulate with `iptables -A INPUT -s <ip> -j DROP` (wrap in dry-run flag if not root).
  - Or maintain in‑memory blocklist and print diff.

---
## Phase 2 – Dataset & Client Isolation
- Already have per‑client CSV splits under `data/optimized/`.
- Validate class balance per client; optionally add noise injection harness for testing poisoning (e.g. label flipping script) to evaluate robustness.
- Create `scripts/simulate_poison.py` to intentionally poison one client's batch (for controlled tests) by:
  - Scaling gradients / adding large perturbations.
  - Flipping labels for a percentage of data.
  - Swapping feature columns.
- Tag poisoned client updates in logs to verify detection.

---
## Phase 3 – Containerization (Docker)
### 3.1 Base Image
Common `Dockerfile.base`:
```
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    useradd -m appuser && chown -R appuser /app
USER appuser
COPY . .
ENV PYTHONPATH=/app
```
Build once and reuse with multi-stage or just two derived Dockerfiles.

### 3.2 Server Dockerfile
`Dockerfile.server` (FROM base) sets:
```
ENV ROLE=server AGGREGATOR=hybrid MAX_BYZANTINE=1 TRIM_K=1
CMD ["python","src/federated/flower_server.py","--clients","4","--rounds","10","--address","0.0.0.0:8080"]
```

### 3.3 Client Dockerfile
`Dockerfile.client` sets:
```
ENV ROLE=client SERVER_ADDRESS=federated-server:8080 CLIENT_ID=0 DATA_DIR=/app/data/optimized
CMD ["python","src/federated/flower_client.py","--client_id","${CLIENT_ID}","--server","${SERVER_ADDRESS}"]
```

### 3.4 Build Commands
```
# Build
docker build -f Dockerfile.base -t ddos-base .
docker build -f Dockerfile.server -t ddos-server .
docker build -f Dockerfile.client -t ddos-client .
```

### 3.5 Local Docker Compose (Optional Pre-K8s)
`docker-compose.yml` with one server service + 4 client replicas (or separate services with distinct CLIENT_ID env vars).

---
## Phase 4 – Kubernetes Orchestration
### 4.1 Namespace & Config
`k8s/namespace.yaml`
```
apiVersion: v1
kind: Namespace
metadata: { name: federated-ddos }
```

`k8s/configmap-server.yaml` & `k8s/configmap-client.yaml` for aggregator settings.

### 4.2 Server Deployment + Service
- Deployment `replicas: 1` exposing container port 8080.
- Service `ClusterIP` (or `NodePort` if external clients) named `federated-server`.

### 4.3 Clients Deployment
- Option A: Single Deployment with `replicas: 4` and `envFrom` + downward API to assign sequential CLIENT_ID (init container writes index file using the pod ordinal if using StatefulSet).
- Option B (simpler): 4 small Deployments (`client-0..client-3`) each with static `CLIENT_ID`.
- Use readinessProbe to wait until server service DNS resolves.

### 4.4 Persistent Storage (Optional)
- If clients need to cache models or logs, mount an `emptyDir` or PVC; otherwise ephemeral is fine.

### 4.5 Aggregation Logs Collection
- Sidecar (e.g., busybox) tailing log file to stdout; or rely on application stdout shipped to `kubectl logs` / a logging stack.

### 4.6 Horizontal Scaling
- To test robustness with more clients, scale the clients deployment and adjust `--clients` startup parameter of server accordingly.

---
## Phase 5 – Network Traffic Capture & Visualization
### 5.1 Capturing Federated Traffic
- Flower uses gRPC over TCP (port 8080). Capture with tshark:
```
tshark -i any -f "tcp port 8080" -w federated_capture.pcap
```
- Run inside: (a) host if using kind / minikube (grab interface), or (b) privileged debug pod:
```
apiVersion: v1
kind: Pod
metadata: { name: capture, namespace: federated-ddos }
spec:
  containers:
  - name: tshark
    image: wireshark/tshark
    command: ["tshark","-i","any","-f","tcp port 8080","-w","/data/capture.pcap"]
    securityContext: { capabilities: { add: ["NET_RAW","NET_ADMIN"] } }
    volumeMounts: [{ name: data, mountPath: /data }]
  volumes: [{ name: data, emptyDir: {} }]
```
- Copy capture: `kubectl cp federated-ddos/capture:/data/capture.pcap ./` and open in Wireshark.

### 5.2 Visual Indicators
- Use filters: `tcp.port==8080`.
- Add color rules to differentiate server (destination vs source).
- Observe packet size variance across rounds correlating with weight transmission.

### 5.3 Optional Metrics Sidecar
- Add Prometheus exporter: record per-round aggregation duration, number of dropped updates, anomaly thresholds.

---
## Phase 6 – Kali Linux CLI Demonstration
### 6.1 Environment
- On Kali (host or VM): pull built images (or run remote cluster access via `kubectl` configured context).
- Provide a lightweight CLI script (`demo_cli.py`):
  - Menu: (1) Show current global model status (latest round) (2) Simulate packet classification from sample PCAP/CSV (3) Show blocked IP list (4) Trigger poisoning test and show mitigation.

### 6.2 Packet Feature Extraction
- Use `tshark -r capture.pcap -T fields -e ip.src -e ip.dst -e frame.len ...` -> transform to feature vector (mirror training feature pipeline).
- Feed into `traffic_guard.py` classifier.

### 6.3 Blocking Simulation
- If root: execute `iptables`/`nft` commands (log them).
- Else: create `blocked_ips.json` for demonstration with timestamp & reason.

### 6.4 Security Event Output (Example)
```
[ROUND 7] Aggregator=Hybrid(MultiKrum->TrimMean)
Accepted clients: [0,1,2,3]
Filtered clients: []
Detected anomalies: 0
Prediction: Flow 192.168.1.10->10.0.0.5 flagged DDoS (confidence 0.94) -> BLOCKED
```

---
## Phase 7 – Testing & Validation
### 7.1 Unit Tests
- Add tests for aggregation functions (Krum correctness for synthetic vectors, trimming counts, median invariance to outliers).
- Add poisoning simulation test verifying filtered update count > 0.

### 7.2 Functional Rounds Test
- Spin up server + 4 local clients (docker-compose) with one client poisoned; assert global accuracy degradation bounded (< X%).

### 7.3 Performance Profiling
- Measure per-round latency before/after robust aggregation; ensure acceptable overhead (log delta ms).

### 7.4 Security Regression
- Intentionally craft extreme weight vector (very large magnitude) -> confirm clipped / dropped.
- Inject NaNs -> confirm rejected.

---
## Phase 8 – Observability & Reporting
- Round log JSONL: one line per round including aggregator used, dropped clients, metrics.
- Add `results/robust_summary.py` to summarize after run (average accuracy, number of filtered updates, false positive filters if any, poisoning scenario outcomes).
- Optional: Grafana dashboard (Prometheus metrics) for aggregator decisions over time.

---
## Phase 9 – Documentation & Automation
- Update `README.md` with: run modes (centralized, federated, robust), container usage, K8s deployment, poisoning test scenario.
- Add `Makefile` (or PowerShell script for Windows) tasks: `make build`, `make run-local`, `make k8s-up`, `make demo-poison`.
- Provide architecture diagram (add `Figure_2.png`): shows pipeline from clients -> robust aggregation -> guard -> blocklist.

---
## Phase 10 – Roadmap (Post MVP)
| Future Item | Description |
|-------------|-------------|
| Secure Channels | mTLS between clients & server; certificates per pod. |
| Differential Privacy | Add noise post-aggregation for additional privacy. |
| Adaptive Trust Scores | Weight clients by historical reliability/anomaly score. |
| Continuous Learning | Online update of model with streaming traffic features. |
| GUI Dashboard | Web UI with aggregation decisions & threat metrics. |

---
## Implementation Sketches
### Krum (Single)
```
# updates: list of flattened numpy arrays
f = max_byzantine
scores = []
for i, ui in enumerate(updates):
    dists = []
    for j, uj in enumerate(updates):
        if i==j: continue
        dists.append(np.linalg.norm(ui-uj)**2)
    dists.sort()
    score = sum(dists[:len(updates)-f-2])
    scores.append((score,i))
_, chosen = min(scores)
aggregated = updates[chosen]
```

### Trimmed Mean (Coordinate-wise)
```
stack = np.stack(updates, axis=0)  # shape: [n, d]
trim_k = k
sorted_vals = np.sort(stack, axis=0)
trimmed = sorted_vals[trim_k: n-trim_k, :]
agg = np.mean(trimmed, axis=0)
```

### Hybrid Outline
```
filtered = sanity_filter(updates)
clipped = clip_norm(filtered, max_norm)
anomies = score(clipped)  # e.g., distance to median
kept = [u for u,s in zip(clipped,anomies) if s < threshold]
if len(kept) < min_required: kept = clipped  # fallback
if aggregator == 'hybrid':
    if len(kept) >= 5: base = multikrum(kept,f)
    else: base = trimmed_mean(kept,k)
final_weights = apply_update(prev_weights, base)
```

---
## Minimal Execution Flow Summary
1. Build images.
2. Deploy server (wait ready).
3. Deploy 4 clients.
4. Capture traffic (tshark pod or host).
5. Run federated rounds with robust aggregator producing logs.
6. (Optional) Introduce poisoning via script; observe mitigation.
7. Export capture; open in Wireshark for visualization.
8. Start Kali CLI guard script -> classify sample flows; show block actions.
9. Summarize results & accuracy impact.

---
## Quick Start Command Snippets (Reference)
(Adjust for your environment; not auto-executed.)
```
# Docker build
docker build -f Dockerfile.server -t ddos-server .
docker build -f Dockerfile.client -t ddos-client .

# Kubernetes apply
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/server-deployment.yaml
kubectl apply -f k8s/server-service.yaml
kubectl apply -f k8s/client-0.yaml
kubectl apply -f k8s/client-1.yaml
kubectl apply -f k8s/client-2.yaml
kubectl apply -f k8s/client-3.yaml

# View logs
kubectl logs -n federated-ddos deploy/federated-server -f

# Capture traffic (example)
tshark -i any -f "tcp port 8080" -w federated.pcap
```

---
## Responsibility Matrix (Concise)
| Concern | Implementation Point |
|---------|----------------------|
| Robust Aggregation | `robust_strategies.py`, modified server strategy |
| Poisoning Simulation | `scripts/simulate_poison.py` |
| Traffic Classification | `traffic_guard.py`, `ddos_guard_cli.py` |
| Containerization | Dockerfiles (base/server/client) |
| Orchestration | K8s manifests under `k8s/` |
| Network Capture | tshark pod / host capture |
| Demo UI | Kali CLI script |
| Reporting | Round JSONL + summary script |

---
## Next Immediate Action Items (MVP Sprint)
1. Implement `robust_strategies.py` (Krum, MultiKrum, Trimmed Mean, Median, helper utils).
2. Integrate into `flower_server.py` with selectable aggregator.
3. Add poisoning simulation script (label flip + gradient scaling).
4. Draft Dockerfiles (base/server/client) + optional docker-compose for quick test.
5. Create initial K8s manifests (namespace, server deployment/service, four client deployments).
6. Add `traffic_guard.py` + basic CLI for inference simulation.
7. Run local functional test: one poisoned client scenario; verify aggregator filters.
8. Write tests for aggregator math correctness.

---
Feel free to request automation of any specific sub-step next.
