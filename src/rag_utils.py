from typing import Dict, List

from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.docstore.document import Document
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel, RunnableLambda

GENERATOR_TEMPLATE = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

Context: {context}

Question: {question}

Helpful Answer:"""


def format_docs(docs: List[Document]) -> str:
    """Formats a list of documents into a concatenated string."""
    return "\n\n".join(doc.page_content for doc in docs)


def ragas_output_parser(docs: List[Document]) -> List[str]:
    """Extracts page content from a list of documents."""
    return [doc.page_content for doc in docs]


def rag_chain_setup(retriever, llm) -> RunnableParallel:
    """Sets up a RAG chain for LangSmith integration.

    Args:
        retriever: The retriever object used to fetch relevant documents.
        llm: The language model to use for generating answers.

    Returns:
        RunnableParallel: A RunnableParallel object representing the RAG chain.
    """

    custom_template = PromptTemplate.from_template(GENERATOR_TEMPLATE)
    generator = custom_template | llm | StrOutputParser()

    context_retriever = RunnableParallel(
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
    )

    filter_langsmith_dataset = RunnableLambda(
        lambda x: x["question"] if isinstance(x, dict) else x
    )
    
    rag_chain = RunnableParallel(
        {
            "question": filter_langsmith_dataset,
            "answer": filter_langsmith_dataset | context_retriever | generator,
            "contexts": filter_langsmith_dataset
            | retriever
            | ragas_output_parser,
        }
    )

    return rag_chain