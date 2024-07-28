import os
from dotenv import load_dotenv

def load_api_keys(key_name: str = None) -> str:
    """
    Load an API keys from environment variables. If key_name is provided, load the API key with that name.

    Args:
        key_name (str): The name of the environment variable to load the API key from.

    Returns:
        str: The API key.

    Raises:
        ValueError: If the API key is not found in the environment variables.
    """
    load_dotenv()  # take environment variables from .env.

    if key_name:
        api_key = os.getenv(key_name)

        if api_key is None:
            raise ValueError(f"API key '{key_name}' not found in environment variables.")
        
        return api_key