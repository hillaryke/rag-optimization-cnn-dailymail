from typing import List, Any
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

from pprint import pprint

load_dotenv()

# constants - can be easily moved to a config file
PG_CONNECTION_STRING = Settings.PG_CONNECTION_STRING
COLLECTION_NAME = Settings.COLLECTION_NAME
SOURCE_FILE_PATH = Settings.SOURCE_FILE_PATH
CHUNK_SIZE = Settings.CHUNK_SIZE
CHUNK_OVERLAP = Settings.CHUNK_OVERLAP


class RAGSystem:
    def __init__(
        self,
        config: Config = None,
        llm: Any = None,
        embeddings: Any = None,
        collection_name: str = COLLECTION_NAME,
        source_file_path: str = SOURCE_FILE_PATH,
        use_existing_vectorstore: str = False,
        clear_store: bool = True,
        use_ensemble: bool = False,
        use_multiquery: bool = False,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        k_documents: int = 5,
        use_reranker: bool = False,
        use_cohere_reranker: bool = False,
        top_n_ranked: int = 5,
    ):
        # pprint(config)
        self.config = config
        self.generator_model = config["models"]["generator_model"]
        self.llm = llm
        self.llm_queries_generator = ChatOpenAI(
            model_name=config["models"]["queries_generator_model"], temperature=0
        )
        self.source_file_path = source_file_path
        self.documents = []
        self.split_docs = List[Document]
        self.collection_name = collection_name
        self.embeddings = embeddings if embeddings else OpenAIEmbeddings()
        self.vectorstore = None
        self.rag_chain = None
        self.clear_store = clear_store
        self.base_retriever = None
        self.final_retriever = None
        self.bm25_retriever = None
        self.use_existing_vectorstore = use_existing_vectorstore
        self.ensemble_retriever = None
        self.use_ensemble = use_ensemble
        self.use_multiquery = use_multiquery
        self.use_reranker = use_reranker
        self.use_cohere_reranker = use_cohere_reranker
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k_documents = k_documents
        self.top_n_ranked = top_n_ranked

    def load_documents(self):
        documents = load_docs_from_csv(as_document=True)
        self.documents = documents

    def prepare_documents(self, len_split_docs: int = 0):
        split_docs = chunk_by_recursive_split(
            self.documents, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        if len_split_docs:
            split_docs = split_docs[:len_split_docs]
        return split_docs

    def initialize_vectorstore(self):
        self.vectorstore = PGVector(
            embeddings=self.embeddings,
            collection_name=self.collection_name,
            connection=PG_CONNECTION_STRING,
            use_jsonb=True,
        )
        return self.vectorstore

    def setup_vectorstore(self):
        # Initialize the vectorstore - this could be an existing collection
        self.initialize_vectorstore()

        if self.clear_store:
            self.vectorstore.drop_tables()
            # Reinitialize the vectorstore once the tables have been dropped
            self.initialize_vectorstore()

            # TODO - calculate the embedding cost if using openai embeddings
            # check instance of embeddings if OpenAIEmbeddings
            # if isinstance(self.embeddings, OpenAIEmbeddings):
            # calculate cost here

            # Get the existing vectorstore collection

            # Add documents to the vectorstore

    def setup_bm25_retriever(self, split_docs: List[str]):
        self.bm25_retriever = BM25Retriever.from_documents(split_docs)
        self.bm25_retriever.k = self.k_documents

    def setup_base_retriever(self):
        self.base_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.k_documents}
        )
        self.final_retriever = self.base_retriever

    def setup_ensemble_retriever(self):
        base_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.k_documents}
        )
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, base_retriever], weights=[0.5, 0.5]
        )
        self.final_retriever = self.ensemble_retriever

    def setup_multiquery_retriever(self, retriever):
        self.final_retriever = MultiQueryRetriever.from_llm(
            retriever=retriever,
            llm=self.llm_queries_generator,
        )

    def setup_reranker(self):
        print("--SETUP RERANKER--")
        my_reranker = Reranker(
            retriever=self.final_retriever,
            top_n=self.top_n_ranked,
            use_cohere_reranker=self.use_cohere_reranker,
        )
        self.final_retriever = my_reranker.initialize()

    def setup_llm(self):
        self.llm = ChatOpenAI(model_name=self.generator_model, temperature=0)

        return self.llm

    def setup_rag_chain(self):
        print("--SETUP RAG CHAIN--")
        llm = self.setup_llm()
        self.rag_chain = rag_chain_setup(self.final_retriever, llm)
        print("--RAGCHAIN SETUP COMPLETE!--")

    def query(self, question: str) -> str:
        result = self.rag_chain.invoke(question)
        return result["answer"]

    def initialize(self, len_split_docs: int = 0):
        self.load_documents()
        self.setup_vectorstore()
        self.setup_base_retriever()

        if not self.use_existing_vectorstore:
            print("--SETUP NEW VECTORSTORE--")
            # Set up a new vectorstore
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
