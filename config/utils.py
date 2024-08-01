import yaml
import os


def load_config(strategy_name: str) -> dict:
    config_file = f"config/config_{strategy_name}.yaml"
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config
