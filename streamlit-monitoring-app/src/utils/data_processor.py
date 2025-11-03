import json
from pathlib import Path


def load_data(file_path=None):
    """Load data from JSON file. If no path provided, uses default sample data."""
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
    return process_detection_data(raw_data)


def load_alerts():
    """Load and process alert data."""
    raw_data = load_data()
    if raw_data is None:
        return []
    return process_alert_data(raw_data)


def process_detection_data(raw_data):
    processed_data = []
    for entry in raw_data['detection']:
        processed_data.append({
            'timestamp': entry['timestamp'],
            'status': entry['status'],
            'confidence': entry['confidence']
        })
    return processed_data


def process_alert_data(raw_data):
    processed_alerts = []
    for alert in raw_data['alerts']:
        processed_alerts.append({
            'timestamp': alert['timestamp'],
            'message': alert['message'],
            'level': alert['level']
        })
    return processed_alerts


def transform_system_state(raw_data):
    return {
        'normal_conditions': raw_data['system_state']['normal'],
        'current_alerts': len(raw_data['alerts']),
        'detection_status': raw_data['detection'][-1]['status'] if raw_data['detection'] else 'N/A'
    }
