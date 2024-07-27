import os
from dotenv import load_dotenv

def load_api_key(key_name: str = None) -> str:
    """
    Load an API key from environment variables.

    Args:
        key_name (str): The name of the environment variable to load the API key from.

    Returns:
        str: The API key.

    Raises:
        ValueError: If the API key is not found in the environment variables.
    """
    load_dotenv()  # take environment variables from .env.

    if key_name is None:
        raise ValueError("key_name must be provided")

    api_key = os.getenv(key_name)

    if api_key is None:
        raise ValueError(f"API key '{key_name}' not found in environment variables.")
    
    return api_key