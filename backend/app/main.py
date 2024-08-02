from typing import Optional

from fastapi import FastAPI, HTTPException, Depends
from pydantic import ValidationError, BaseModel
from config.utils import load_config
from src.rag_pipeline.rag_system import RAGSystem
from src.env_loader import load_api_keys

# Load API keys from environment
load_api_keys()

app = FastAPI()


class InitializeRequest(BaseModel):
    """
    Request model for initializing the RAG system.

    Attributes:
        strategy_name (str): The name of the strategy to use for initialization.
        split_docs (Optional[int]): The number of split documents to use. Defaults to None.
    """

    strategy_name: str
    split_docs: Optional[int] = None


class QueryRequest(BaseModel):
    """
    Request model for querying the RAG system.

    Attributes:
        question (str): The question to query the RAG system with.
    """

    question: str


# Global variable to store the initialized RAGSystem
rag_system_instance: Optional[RAGSystem] = None


def get_rag_system() -> RAGSystem:
    """
    Dependency to get the initialized RAGSystem.

    Raises:
        HTTPException: If the RAG system is not initialized.

    Returns:
        RAGSystem: The initialized RAG system instance.
    """
    if rag_system_instance is None:
        raise HTTPException(status_code=500, detail="RAG system is not initialized")
    return rag_system_instance


@app.get("/")
def read_root():
    """
    Root endpoint to check if the API is running.

    Returns:
        dict: A welcome message.
    """
    return {"message": "Hello from FastAPI backend!"}


@app.get("/health")
def health_check():
    """
    Health check endpoint to verify if the RAG system is initialized and running.

    Returns:
        dict: The status of the RAG system.
    """
    try:
        rag_system = get_rag_system()
        return {"status": "RAG system is initialized and running"}
    except HTTPException as e:
        return {"status": "RAG system is not initialized", "detail": str(e)}


@app.post("/initialize")
def initialize_rag_system(init_request: InitializeRequest):
    """
    Endpoint to initialize the RAG system.

    Args:
        init_request (InitializeRequest): The initialization request containing strategy name and split_docs.

    Returns:
        dict: A message indicating the result of the initialization.

    Raises:
        HTTPException: If there is a configuration error or initialization fails.
    """
    global rag_system_instance
    try:
        config = load_config(init_request.strategy_name)
        rag_system_instance = RAGSystem(config=config)
        rag_system_instance.initialize(init_request.split_docs)
        return {
            "message": f"RAG system initialized with strategy '{init_request.strategy_name}' and split_docs={init_request.split_docs}"
        }
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Configuration error: {e}") from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Initialization failed: {e}"
        ) from e


@app.post("/query")
def query_rag_system(
    query_request: QueryRequest, rag_system: RAGSystem = Depends(get_rag_system)
):
    """
    Endpoint to query the RAG system.

    Args:
        query_request (QueryRequest): The query request containing the question.
        rag_system (RAGSystem): The initialized RAG system instance.

    Returns:
        dict: The answer from the RAG system.

    Raises:
        HTTPException: If the query fails.
    """
    try:
        answer = rag_system.query(query_request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}") from e
