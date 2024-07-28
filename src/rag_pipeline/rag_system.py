import os
import pandas as pd

from typing import List, Dict
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from dotenv import load_dotenv
load_dotenv()

from src.rag_pipeline.rag_utils import rag_chain_setup
from src.rag_pipeline.chunking_strategies import chunk_by_recursive_split
from src.rag_pipeline.load_docs import load_docs_from_csv

CHROMA_PATH = "chromadb"

class RAGSystem:
    def __init__(self, 
                model_name: str, 
                source_file_path: 
                str = "data/cnn_dailymail_validation_subset.csv",
                existing_chroma: str = False,
                use_ensemble_retriever: bool = False,
                chunk_size: int = 1000,
                chunk_overlap: int = 200
    ):
        self.model_name = model_name
        self.llm = None
        self.source_file_path = source_file_path
        self.documents = []
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.rag_chain = None
        self.existing_chroma = existing_chroma
        self.vectorstore_retriever = None
        self.bm25_retriever = None
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

    def setup_vectorstore(self, split_docs: List[str] = None):
        if split_docs is None:
            # Load an existing Chroma instance
            self.vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embeddings)
        else:
            # Create a new Chroma instance
            self.vectorstore = Chroma.from_documents(split_docs, embedding=self.embeddings, persist_directory=CHROMA_PATH)

    def setup_bm25_retriever(self, split_docs: List[str]):
        self.bm25_retriever = BM25Retriever.from_documents(split_docs)
        self.bm25_retriever.k = 3
        
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
        
        if self.existing_chroma:
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

