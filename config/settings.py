from pydantic import BaseModel


class VectorStoreConfig(BaseModel):
    collection_name: str = "cnn_dailymail"
    clear_store: bool = True
    use_existing_vectorstore: bool = False


class ChunkingConfig(BaseModel):
    chunk_type: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200


class RetrievalConfig(BaseModel):
    k_documents: int = 5
    use_ensemble: bool = False
    use_multiquery: bool = False
    use_reranker: bool = False
    use_cohere_reranker: bool = False
    top_n_ranked: int = 5


class ModelsConfig(BaseModel):
    generator_model: str = "gpt-4o-mini"
    queries_generator_model: str = "gpt-4o-mini"


class Config(BaseModel):
    vectorstore: VectorStoreConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    models: ModelsConfig
