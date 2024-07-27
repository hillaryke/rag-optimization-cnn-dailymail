import os
import pandas as pd

from typing import List, Dict
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

def load_documents_from_csv(
  file_path: str = "data/cnn_dailymail_validation_subset.csv", 
  page_content_column: str = "article"
) -> List[str]:
    df = pd.read_csv(file_path)
    return df[page_content_column].tolist()

CHROMA_PATH = "chromadb"

class RAGSystem:
    def __init__(self, 
                 model_name: str, 
                 source_file_path: 
                  str = "data/cnn_dailymail_validation_subset.csv",
                  existing_chroma: str = False
    ):
        self.model_name = model_name
        self.llm = None
        self.source_file_path = source_file_path
        self.documents = []
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.rag_chain = None
        self.existing_chroma = existing_chroma

    def load_documents(self, file_path: str = None):
        documents = load_documents_from_csv(self.source_file_path)
        self.documents = documents
        
    def prepare_documents(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.create_documents(self.documents)

    def setup_vectorstore(self, split_docs: List[str] = None):
        if split_docs is None:
            # Load an existing Chroma instance
            self.vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embeddings)
            
        else:
          # Create a new Chroma instance
          self.vectorstore = Chroma.from_documents(split_docs, embedding=self.embeddings, persist_directory=CHROMA_PATH)

    def setup_llm(self):
        llm = ChatOpenAI(model_name=self.model_name, temperature=0)
        self.llm = llm
        return llm

    def setup_rag_chain(self):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        llm = self.setup_llm()
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

    def query(self, question: str) -> str:
        result = self.rag_chain.invoke(question)
        return result["result"]

    def initialize(self):
        self.load_documents()
        if not self.existing_chroma:
            split_docs = self.prepare_documents()
            self.setup_vectorstore(split_docs)
        else:
            self.setup_vectorstore()
        self.setup_rag_chain()
