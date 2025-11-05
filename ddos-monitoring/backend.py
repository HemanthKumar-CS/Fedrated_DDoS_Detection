"""
FastAPI Backend for Real-time DDoS Detection Monitoring
Provides REST API and WebSocket connections for live attack data streaming
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import random
from datetime import datetime, timedelta
from typing import List

app = FastAPI(title="DDoS Detection API", version="2.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active WebSocket connections
active_connections: List[WebSocket] = []


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "DDoS Detection API v2.0"
    }


@app.get("/api/detection-status")
async def get_detection_status():
    """Get current detection status"""
    current_time = datetime.now()
    minute_of_hour = current_time.minute
    is_under_attack = minute_of_hour % 15 < 5
    
    detection_rate = random.uniform(0.7, 0.95) if is_under_attack else random.uniform(0.05, 0.2)
    
    return {
        "timestamp": current_time.isoformat(),
        "is_under_attack": is_under_attack,
        "threat_level": "CRITICAL" if detection_rate > 0.8 else "HIGH" if detection_rate > 0.6 else "MEDIUM" if detection_rate > 0.3 else "LOW",
        "attack_probability": round(detection_rate * 100, 2),
        "benign_probability": round((1 - detection_rate) * 100, 2),
        "packets_per_second": random.randint(1000, 5000) if is_under_attack else random.randint(10, 100),
        "detected_anomalies": random.randint(1, 10) if is_under_attack else random.randint(0, 2),
    }


@app.websocket("/ws/live-detection")
async def websocket_live_detection(websocket: WebSocket):
    """WebSocket endpoint for real-time detection data"""
    try:
        await websocket.accept()
        active_connections.append(websocket)
        print(f"✅ Client connected to detection. Total: {len(active_connections)}")
        
        while True:
            current_time = datetime.now()
            minute_of_hour = current_time.minute
            is_under_attack = minute_of_hour % 15 < 5
            
            detection_rate = random.uniform(0.7, 0.95) if is_under_attack else random.uniform(0.05, 0.2)
            packets = random.randint(1000, 5000) if is_under_attack else random.randint(10, 100)
            
            data = {
                "type": "detection_update",
                "timestamp": current_time.isoformat(),
                "is_under_attack": is_under_attack,
                "detection_rate": round(detection_rate * 100, 2),
                "packets_per_second": packets,
                "threat_level": "CRITICAL" if detection_rate > 0.8 else "HIGH" if detection_rate > 0.6 else "MEDIUM" if detection_rate > 0.3 else "LOW",
                "anomalies": random.randint(0, 10),
                "benign": random.randint(0, 100)
            }
            
            try:
                await websocket.send_json(data)
            except:
                break
                
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        print(f"❌ Client disconnected from detection. Total: {len(active_connections) - 1}")
    except Exception as e:
        print(f"Error in detection ws: {e}")
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)


@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket endpoint for real-time alerts"""
    try:
        await websocket.accept()
        active_connections.append(websocket)
        print(f"✅ Client connected to alerts. Total: {len(active_connections)}")
        
        alert_counter = 0
        while True:
            current_time = datetime.now()
            minute_of_hour = current_time.minute
            is_attack_window = minute_of_hour % 15 < 5
            
            if is_attack_window and random.random() > 0.7:
                alert_counter += 1
                alert = {
                    "type": "alert",
                    "id": alert_counter,
                    "message": random.choice([
                        "High anomaly detection rate",
                        "Unusual packet pattern detected",
                        "DDoS attack signature detected",
                        "Traffic spike detected",
                        "Policy violation detected"
                    ]),
                    "level": random.choice(["critical", "warning"]),
                    "timestamp": current_time.isoformat(),
                    "source": random.choice([f"sensor_{i}" for i in range(1, 5)])
                }
                try:
                    await websocket.send_json(alert)
                except:
                    break
            
            await asyncio.sleep(2)
            
    except WebSocketDisconnect:
        print(f"❌ Client disconnected from alerts")
    except Exception as e:
        print(f"Error in alerts ws: {e}")
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)


@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    """WebSocket endpoint for real-time system metrics"""
    try:
        await websocket.accept()
        active_connections.append(websocket)
        print(f"✅ Client connected to metrics. Total: {len(active_connections)}")
        
        while True:
            metrics = {
                "type": "metrics_update",
                "timestamp": datetime.now().isoformat(),
                "cpu_usage": round(random.uniform(20, 85), 2),
                "memory_usage": round(random.uniform(30, 75), 2),
                "active_connections": len(active_connections),
                "packets_processed": random.randint(1000000, 5000000),
                "requests_per_second": random.randint(100, 500)
            }
            try:
                await websocket.send_json(metrics)
            except:
                break
                
            await asyncio.sleep(2)
            
    except WebSocketDisconnect:
        print(f"❌ Client disconnected from metrics")
    except Exception as e:
        print(f"Error in metrics ws: {e}")
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI Server...")
    print("📡 API: http://localhost:8000")
    print("🔌 WebSocket: ws://localhost:8000/ws/*")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
