from typing import Optional

from fastapi import FastAPI, HTTPException, Depends
from pydantic import ValidationError
from pydantic import BaseModel
from config.utils import load_config
from src.rag_pipeline.rag_system import RAGSystem
from src.env_loader import load_api_keys

load_api_keys()

app = FastAPI()

# Global variable to store the initialized RAGSystem
rag_system_instance: Optional[RAGSystem] = None


@app.on_event("startup")
async def startup_event():
    try:
        strategy_name = "base"
        config = load_config(strategy_name)
        rag_system_instance = RAGSystem(config=config)
        rag_system_instance.initialize(5)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Configuration error: {e}") from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Initialization failed: {e}"
        ) from e


# Dependency to get the initialized RAGSystem
def get_rag_system():
    if rag_system_instance is None:
        raise HTTPException(status_code=500, detail="RAG system is not initialized")
    return rag_system_instance


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI backend!"}


@app.get("/health")
def health_check():
    try:
        rag_system = get_rag_system()
        print(rag_system)
        return {"status": "RAG system is initialized and running"}
    except HTTPException as e:
        return {"status": "RAG system is not initialized", "detail": str(e)}


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
def query_rag_system(
    query_request: QueryRequest, rag_system: RAGSystem = Depends(get_rag_system)
):
    try:
        answer = rag_system.query(query_request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}") from e
