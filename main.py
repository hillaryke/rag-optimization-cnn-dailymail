from pydantic import ValidationError
from config.settings import Config
from config.utils import load_config
from src.rag_pipeline.rag_system import RAGSystem

try:
    strategy_name = "base"
    config = load_config(strategy_name)
except ValidationError as e:
    print(f"Configuration error: {e}")
    exit(1)

config = load_config("base")
rag_system = RAGSystem(config=config)

rag_system.initialize(5)
