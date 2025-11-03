import json
from pathlib import Path
import requests
from typing import Dict, List, Optional


API_BASE_URL = "http://localhost:5000"
TIMEOUT = 5


def get_api_health() -> Dict:
    """Get API health status from backend."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=TIMEOUT)
        return response.json()
    except Exception as e:
        return {"status": "offline", "error": str(e)}


def get_detection_metrics() -> Dict:
    """Get detection metrics from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/metrics", timeout=TIMEOUT)
        return response.json()
    except Exception as e:
        return {"error": "Failed to fetch metrics", "status": "offline"}


def load_data(file_path=None):
    """Load data from JSON file or API. If no path provided, uses default sample data."""
    # Try API first
    if API_BASE_URL:
        try:
            health = get_api_health()
            if health.get("status") == "healthy":
                metrics = get_detection_metrics()
                if "error" not in metrics:
                    return metrics
        except:
            pass

    # Fallback to file-based data
    if file_path is None:
        # Use default sample data path
        file_path = Path(__file__).parent.parent / "data" / "sample_data.json"

    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        return None


def load_detection_data():
    """Load and process detection data."""
    raw_data = load_data()
    if raw_data is None:
        return None

    # Try to process API response
    if "detection_status" in raw_data:
        return {
            "timestamp": raw_data.get("detection_status", {}).get("timestamp", "N/A"),
            "status": raw_data.get("detection_status", {}).get("status", "N/A"),
            "details": raw_data.get("detection_status", {}).get("details", {})
        }

    # Fallback to file-based processing
    return process_detection_data(raw_data)


def load_alerts():
    """Load and process alert data."""
    raw_data = load_data()
    if raw_data is None:
        return []
    return process_alert_data(raw_data)


def process_detection_data(raw_data):
    processed_data = []
    for entry in raw_data.get('detection', []):
        processed_data.append({
            'timestamp': entry['timestamp'],
            'status': entry['status'],
            'confidence': entry['confidence']
        })
    return processed_data


def process_alert_data(raw_data):
    processed_alerts = []
    for alert in raw_data.get('alerts', []):
        processed_alerts.append({
            'timestamp': alert.get('timestamp', 'N/A'),
            'message': alert.get('message', 'N/A'),
            'level': alert.get('level', 'info')
        })
    return processed_alerts


def transform_system_state(raw_data):
    return {
        'overall_status': raw_data.get('system_state', {}).get('overall_status', 'unknown'),
        'current_alerts': len(raw_data.get('alerts', [])),
        'detection_status': raw_data.get('detection_status', {}).get('status', 'N/A'),
        'metrics': raw_data.get('system_state', {}).get('metrics', {})
    }
