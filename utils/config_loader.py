# utils/config_loader.py
import yaml
import os

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML config file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# Singleton config để dùng xuyên suốt
_CONFIG = None

def get_config():
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG