import yaml
from typing import List, Dict, Any
from loguru import logger
from config.settings import Config
from src.env_loader import load_api_keys

# Load API keys from environment
load_api_keys()


def load_config(strategy_name: str) -> Config:
    """
    Load configuration for a given strategy from a YAML file.

    Args:
        strategy_name (str): The name of the strategy to load the configuration for.

    Returns:
        Config: The configuration object.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
        TypeError: If the configuration data cannot be converted to a Config object.
    """
    config_file = f"config/config_{strategy_name}.yaml"
    try:
        with open(config_file, "r") as file:
            config_data = yaml.safe_load(file)
        return Config(**config_data)
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {config_file}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {config_file}")
        raise
    except TypeError as e:
        logger.error(f"Error converting configuration data to Config object: {e}")
        raise
