import os
import json

def load_config(config_file='src/data/sample_data.json'):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as file:
        config = json.load(file)
    
    return config

def get_setting(key, config):
    return config.get(key, None)

def save_setting(key, value, config_file='src/data/sample_data.json'):
    config = load_config(config_file)
    config[key] = value
    
    with open(config_file, 'w') as file:
        json.dump(config, file, indent=4)