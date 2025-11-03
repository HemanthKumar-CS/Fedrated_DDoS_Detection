def load_data(file_path):
    import json
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

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