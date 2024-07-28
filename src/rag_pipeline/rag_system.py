import os
import pandas as pd

from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_postgres import PGVector

from dotenv import load_dotenv
load_dotenv()

from src.rag_pipeline.rag_utils import rag_chain_setup
from src.rag_pipeline.chunking_strategies import chunk_by_recursive_split
from src.rag_pipeline.load_docs import load_docs_from_csv

# constants - can be easily moved to a config file
PG_CONNECTION_STRING = os.getenv("PG_CONNECTION_STRING")
COLLECTION_NAME = "cnn_dailymail_validation_subset"
SOURCE_FILE_PATH = "data/cnn_dailymail_validation_subset.csv"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

class RAGSystem:
    def __init__(self, 
                model_name: str,
                embeddings: Any = None,
                collection_name: str = COLLECTION_NAME,
                source_file_path: str = SOURCE_FILE_PATH,
                existing_vectorstore: str = False,
                clear_store: bool = False,
                use_ensemble_retriever: bool = False,
                chunk_size: int = CHUNK_SIZE,
                chunk_overlap: int = CHUNK_OVERLAP
    ):
        self.model_name = model_name
        self.llm = None
        self.source_file_path = source_file_path
        self.documents = []
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.rag_chain = None
        self.clear_store = clear_store
        self.vectorstore_retriever = None
        self.bm25_retriever = None
        self.existing_vectorstore = existing_vectorstore
        self.ensemble_retriever = None
        self.use_ensemble_retriever = use_ensemble_retriever
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.k_documents = 5

    def load_documents(self, file_path: str = None):
        documents = load_docs_from_csv(as_document=True)
        self.documents = documents
        
    def prepare_documents(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        split_docs = chunk_by_recursive_split(self.documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return split_docs
    
    def initialize_vectorstore(self):
        self.vectorstore = PGVector(
            embeddings=self.embeddings,
            collection_name=COLLECTION_NAME,
            connection=PG_CONNECTION_STRING,
            use_jsonb=True
        )
        return self.vectorstore

    def setup_vectorstore(self, split_docs: List[str] = None):
        # For testing
        split_docs = split_docs[:3]
        # Initialize the vectorstore - this could be an existing collection
        self.initialize_vectorstore()
        
        if self.clear_store:
            self.vectorstore.drop_tables()
        
        if split_docs:
            self.initialize_vectorstore()
            self.vectorstore.add_documents(split_docs)

            # caclculate the embedding cost if using openai embeddings
            # check instance of embeddings if OpenAIEmbeddings
            # if isinstance(self.embeddings, OpenAIEmbeddings):
                # calculate cost here
            
            # Get the existing vectorstore collection
            
            # Add documents to the vectorstore

    def setup_bm25_retriever(self, split_docs: List[str]):
        self.bm25_retriever = BM25Retriever.from_documents(split_docs)
        self.bm25_retriever.k = self.k_documents
        
    def setup_basic_retriever(self):
        self.vectorstore_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k_documents})

    def setup_ensemble_retriever(self):
        chroma_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k_documents})
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, chroma_retriever],
            weights=[0.5, 0.5]
        )

    def setup_llm(self):
        llm = ChatOpenAI(model_name=self.model_name, temperature=0)
        self.llm = llm
        return llm

    def setup_rag_chain(self):
        llm = self.setup_llm()
        if self.use_ensemble_retriever:
            self.rag_chain = rag_chain_setup(self.ensemble_retriever, llm)
        else:
            self.rag_chain = rag_chain_setup(self.vectorstore_retriever, llm)

    def query(self, question: str) -> str:
        result = self.rag_chain.invoke(question)
        return result["answer"]

    def initialize(self):
        self.load_documents()
        
        if self.existing_vectorstore:
            self.setup_vectorstore()
            self.setup_basic_retriever()
        else:
            split_docs = self.prepare_documents()
            self.setup_vectorstore(split_docs)

            if self.use_ensemble_retriever:
                self.setup_bm25_retriever(split_docs)
                self.setup_ensemble_retriever()
            else:
                self.setup_basic_retriever()
            
        self.setup_rag_chain()

