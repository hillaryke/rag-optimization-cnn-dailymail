from typing import Optional
import logging
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_cohere import CohereRerank

# No need to import Cohere if not directly used within the class


class Reranker:
    """
    Enhances document retrieval by re-ranking results based on relevance to a query.
    Offers a choice between open-source or Cohere's commercial reranking model.
    """

    def __init__(
        self,
        retriever,
        top_n: int = 5,
        reranker_model: Optional[HuggingFaceCrossEncoder] = None,
        use_cohere_reranker: bool = False,
    ):
        """
        Initializes the Reranker.

        Args:
            retriever: The base document retriever to use.
            top_n: The number of top-ranked documents to consider (default: 5).
            reranker_model: A custom HuggingFaceCrossEncoder model (optional).
            use_cohere_reranker: Whether to use Cohere's reranking model (default: False).
        """

        self.retriever = retriever
        self.top_n = top_n
        self.use_cohere_reranker = use_cohere_reranker

        # Initialize with default model or provided custom model
        self.reranker_model = reranker_model or HuggingFaceCrossEncoder(
            model_name="BAAI/bge-reranker-base"
        )

        # Initialize after deciding on the model
        self._initialize_reranker()

    def _initialize_reranker(self):
        """Initializes the appropriate reranking model and compression retriever."""

        # Use logger instead of print statements for better logging practices
        if self.use_cohere_reranker:
            logging.info("Using Cohere model for reranking")
            compressor = CohereRerank(model="rerank-english-v3.0")
        else:
            logging.info("Using open source model for reranking")
            compressor = CrossEncoderReranker(
                model=self.reranker_model, top_n=self.top_n
            )

        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=self.retriever
        )
