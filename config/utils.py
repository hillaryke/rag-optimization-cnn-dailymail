import yaml

from typing import List, Dict, Any
from loguru import logger
from config.settings import Config
from src.env_loader import load_api_keys

load_api_keys()


def load_config(strategy_name: str) -> Config:
    config_file = f"config/config_{strategy_name}.yaml"
    with open(config_file, "r") as file:
        config_data = yaml.safe_load(file)
    return Config(**config_data)
