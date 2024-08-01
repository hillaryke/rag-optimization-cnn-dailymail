from pydantic import BaseModel


class DatabaseConfig(BaseModel):
    clear_store: bool
    use_existing_vectorstore: bool


class ChunkingConfig(BaseModel):
    chunk_type: str
    chunk_size: int
    chunk_overlap: int


class RetrievalConfig(BaseModel):
    k_documents: int
    use_ensemble: bool
    use_multiquery: bool
    use_reranker: bool
    use_cohere_reranker: bool
    top_n_ranked: int


class ModelsConfig(BaseModel):
    generator_model: str


class Config(BaseModel):
    vectorstore: DatabaseConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    models: ModelsConfig
