from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableLambda

GENERATOR_TEMPLATE = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

Context: {context}

Question: {question}

Helpful Answer:"""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ragas_output_parser(docs):
    return [doc.page_content for doc in docs]

def rag_chain_setup(retriever, llm):
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