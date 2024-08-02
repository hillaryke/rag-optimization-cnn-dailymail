from pydantic import BaseModel


class VectorStoreConfig(BaseModel):
    """
    Configuration for the vector store.

    Attributes:
        collection_name (str): The name of the collection in the vector store.
        clear_store (bool): Whether to clear the store before use.
        use_existing_vectorstore (bool): Whether to use an existing vector store.
    """

    collection_name: str = "cnn_dailymail"
    clear_store: bool = True
    use_existing_vectorstore: bool = False


class ChunkingConfig(BaseModel):
    """
    Configuration for document chunking.

    Attributes:
        chunk_type (str): The type of chunking strategy.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The overlap between chunks.
    """

    chunk_type: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200


class RetrievalConfig(BaseModel):
    """
    Configuration for document retrieval.

    Attributes:
        k_documents (int): The number of documents to retrieve.
        use_ensemble (bool): Whether to use an ensemble retriever.
        use_multiquery (bool): Whether to use a multi-query retriever.
        use_reranker (bool): Whether to use a reranker.
        use_cohere_reranker (bool): Whether to use the Cohere reranker.
        top_n_ranked (int): The number of top-ranked documents to return.
    """

    k_documents: int = 5
    use_ensemble: bool = False
    use_multiquery: bool = False
    use_reranker: bool = False
    use_cohere_reranker: bool = False
    top_n_ranked: int = 5


class ModelsConfig(BaseModel):
    """
    Configuration for models.

    Attributes:
        generator_model (str): The name of the generator model.
        queries_generator_model (str): The name of the queries generator model.
    """

    generator_model: str = "gpt-4o-mini"
    queries_generator_model: str = "gpt-4o-mini"


class Config(BaseModel):
    """
    Main configuration class that aggregates all other configurations.

    Attributes:
        vectorstore (VectorStoreConfig): Configuration for the vector store.
        chunking (ChunkingConfig): Configuration for document chunking.
        retrieval (RetrievalConfig): Configuration for document retrieval.
        models (ModelsConfig): Configuration for models.
    """

    vectorstore: VectorStoreConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    models: ModelsConfig
