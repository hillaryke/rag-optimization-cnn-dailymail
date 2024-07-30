from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere

class Reranker:
  def __init__(self, retriever, top_n: int = 5, reranker_model = None, use_cohere_reranker: bool = False):
    self.reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base") if reranker_model is None else reranker_model
    self.top_n = top_n
    self.retriever = retriever
    self.use_cohere_reranker = use_cohere_reranker
    self.compression_retriever = None
    self.compressor = None

  def setup_opensource_model(self):
    print("--USING OPEN SOURCE MODEL FOR RERANKING--")
    self.compressor = CrossEncoderReranker(model=self.reranker_model, top_n=3)
    return self.compression_retriever
  
  def setup_cohere_model(self):
    print("--USING COHERE MODEL FOR RERANKING--")
    self.compressor = CohereRerank(model="rerank-english-v3.0")
    return self.compression_retriever
  
  def setup_compression_retriever(self):
    self.compression_retriever = ContextualCompressionRetriever(
        base_compressor=self.compressor, base_retriever=self.retriever
    )
    
  def initialize(self):
    if self.use_cohere_reranker:
      self.setup_cohere_model()
    else:
      self.setup_opensource_model()
    
    self.setup_compression_retriever()
    
    return self.compression_retriever