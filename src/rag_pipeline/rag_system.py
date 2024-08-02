from typing import List, Any, Optional
from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from src.rag_pipeline.rag_utils import rag_chain_setup
from src.rag_pipeline.chunking_strategies import chunk_by_recursive_split
from src.rag_pipeline.load_docs import load_docs_from_csv
from src.rag_pipeline.reranker import Reranker
from misc import Settings
from config.settings import Config

load_dotenv()

# Constants - can be easily moved to a config file
PG_CONNECTION_STRING = Settings.PG_CONNECTION_STRING
COLLECTION_NAME = Settings.COLLECTION_NAME
SOURCE_FILE_PATH = Settings.SOURCE_FILE_PATH
CHUNK_SIZE = Settings.CHUNK_SIZE
CHUNK_OVERLAP = Settings.CHUNK_OVERLAP


class RAGSystem:
    def __init__(
        self,
        config: Config,
        embeddings: Optional[Any] = None,
        source_file_path: str = SOURCE_FILE_PATH,
    ):
        """
        Initialize the RAGSystem with configuration and optional embeddings.

        Args:
            config (Config): Configuration object.
            embeddings (Any, optional): Embeddings object. Defaults to OpenAIEmbeddings.
            source_file_path (str, optional): Path to the source file. Defaults to SOURCE_FILE_PATH.
        """
        self.config = config
        self.generator_model = config.models.generator_model
        self.llm_queries_generator = ChatOpenAI(
            model_name=config.models.queries_generator_model, temperature=0
        )
        self.llm = None
        self.source_file_path = source_file_path
        self.documents = []
        self.split_docs = []
        self.collection_name = config.vectorstore.collection_name
        self.embeddings = embeddings if embeddings else OpenAIEmbeddings()
        self.vectorstore = None
        self.rag_chain = None
        self.clear_store = config.vectorstore.clear_store
        self.base_retriever = None
        self.final_retriever = None
        self.bm25_retriever = None
        self.use_existing_vectorstore = config.vectorstore.use_existing_vectorstore
        self.ensemble_retriever = None
        self.use_ensemble = config.retrieval.use_ensemble
        self.use_multiquery = config.retrieval.use_multiquery
        self.use_reranker = config.retrieval.use_reranker
        self.use_cohere_reranker = config.retrieval.use_cohere_reranker
        self.chunk_size = config.chunking.chunk_size
        self.chunk_overlap = config.chunking.chunk_overlap
        self.k_documents = config.retrieval.k_documents
        self.top_n_ranked = config.retrieval.top_n_ranked

    def load_documents(self):
        """Load documents from the source file."""
        try:
            self.documents = load_docs_from_csv(as_document=True)
        except Exception as e:
            print(f"Error loading documents: {e}")
            raise

    def prepare_documents(self, len_split_docs: int = 0) -> List[Document]:
        """
        Prepare documents by chunking them.

        Args:
            len_split_docs (int, optional): Number of split documents to return for testing purposes. Defaults to 0.

        Returns:
            List[Document]: List of split documents.
        """
        try:
            split_docs = chunk_by_recursive_split(
                self.documents,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            if len_split_docs:
                split_docs = split_docs[:len_split_docs]
            print(f"--documents_no: {len(split_docs)}")
            return split_docs
        except Exception as e:
            print(f"Error preparing documents: {e}")
            raise

    def initialize_vectorstore(self):
        """Initialize the vectorstore."""
        try:
            self.vectorstore = PGVector(
                embeddings=self.embeddings,
                collection_name=self.collection_name,
                connection=PG_CONNECTION_STRING,
                use_jsonb=True,
            )
        except Exception as e:
            print(f"Error initializing vectorstore: {e}")
            raise

    def setup_vectorstore(self):
        """Setup the vectorstore, optionally clearing it first."""
        self.initialize_vectorstore()

        if self.clear_store:
            try:
                self.vectorstore.drop_tables()
                self.initialize_vectorstore()
            except Exception as e:
                print(f"Error clearing vectorstore: {e}")
                raise

    def setup_bm25_retriever(self, split_docs: List[Document]):
        """Setup the BM25 retriever."""
        try:
            self.bm25_retriever = BM25Retriever.from_documents(split_docs)
            self.bm25_retriever.k = self.k_documents
        except Exception as e:
            print(f"Error setting up BM25 retriever: {e}")
            raise

    def setup_base_retriever(self):
        """Setup the base retriever."""
        try:
            self.base_retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.k_documents}
            )
            self.final_retriever = self.base_retriever
        except Exception as e:
            print(f"Error setting up base retriever: {e}")
            raise

    def setup_ensemble_retriever(self):
        """Setup the ensemble retriever."""
        try:
            base_retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.k_documents}
            )
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, base_retriever], weights=[0.5, 0.5]
            )
            self.final_retriever = self.ensemble_retriever
        except Exception as e:
            print(f"Error setting up ensemble retriever: {e}")
            raise

    def setup_multiquery_retriever(self, retriever):
        """Setup the multi-query retriever."""
        try:
            self.final_retriever = MultiQueryRetriever.from_llm(
                retriever=retriever,
                llm=self.llm_queries_generator,
            )
        except Exception as e:
            print(f"Error setting up multi-query retriever: {e}")
            raise

    def setup_reranker(self):
        """Setup the reranker."""
        try:
            print("--SETUP RERANKER--")
            my_reranker = Reranker(
                retriever=self.final_retriever,
                top_n=self.top_n_ranked,
                use_cohere_reranker=self.use_cohere_reranker,
            )
            self.final_retriever = my_reranker.initialize()
        except Exception as e:
            print(f"Error setting up reranker: {e}")
            raise

    def setup_llm(self):
        """Setup the language model."""
        try:
            self.llm = ChatOpenAI(model_name=self.generator_model, temperature=0)
            return self.llm
        except Exception as e:
            print(f"Error setting up LLM: {e}")
            raise

    def setup_rag_chain(self):
        """Setup the RAG chain."""
        try:
            print("--SETUP RAG CHAIN--")
            llm = self.setup_llm()
            self.rag_chain = rag_chain_setup(self.final_retriever, llm)
            print("--RAGCHAIN SETUP COMPLETE!--")
        except Exception as e:
            print(f"Error setting up RAG chain: {e}")
            raise

    def query(self, question: str) -> str:
        """
        Query the RAG system.

        Args:
            question (str): The question to query.

        Returns:
            str: The answer from the RAG system.
        """
        try:
            result = self.rag_chain.invoke(question)
            return result["answer"]
        except Exception as e:
            print(f"Error querying RAG system: {e}")
            raise

    def initialize(self, len_split_docs: int = 0):
        """Initialize the RAG system."""
        try:
            self.load_documents()
            self.setup_vectorstore()
            self.setup_base_retriever()

            if not self.use_existing_vectorstore:
                print("--SETUP NEW VECTORSTORE--")
                self.split_docs = self.prepare_documents(len_split_docs)
                self.vectorstore.add_documents(self.split_docs)

                if self.use_ensemble:
                    print("--USING ENSEMBLE RETRIEVER--")
                    self.setup_bm25_retriever(self.split_docs)
                    self.setup_ensemble_retriever()
                elif self.use_multiquery:
                    print("--USING MULTIQUERY RETRIEVER--")
                    self.setup_multiquery_retriever(self.base_retriever)
                else:
                    print("--USING BASE RETRIEVER--")
                    self.setup_base_retriever()

            if self.use_reranker:
                self.setup_reranker()

            self.setup_rag_chain()
        except Exception as e:
            print(f"Error initializing RAG system: {e}")
            raise
