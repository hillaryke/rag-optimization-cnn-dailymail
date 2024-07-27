from typing import List
from langchain.docstore.document import Document
from datasets import load_dataset
import pandas as pd

def load_and_process_dataset(split: str = "validation[:1000]") -> List[Document]:
    """Loads the CNN/Daily Mail dataset, creates LangChain Documents, and returns them.

    Args:
        split (str, optional): The dataset split to load (e.g., "train", "validation[:1000]").
            Defaults to "validation[:1000]".

    Returns:
        List[Document]: A list of LangChain Document objects.
    """

    dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation[:1000]")
    
    documents = [
        Document(page_content=article["article"], metadata={"source": "cnn_dailymail"})
        for article in dataset
    ]

    return documents

def pretty_print_docs(docs):
    """
    Prints the content of documents in a formatted manner.

    Parameters:
    - docs (list): A list of document objects. Each document object must have a 'page_content' attribute.

    Returns:
    None. This function prints the content of each document to the console.
    """
    try:
        # Check if docs is a list
        if not isinstance(docs, list):
            raise ValueError("The 'docs' parameter should be a list of document objects.")

        # Check if each document has 'page_content' attribute
        for d in docs:
            if not hasattr(d, 'page_content'):
                raise AttributeError("Each document object must have a 'page_content' attribute.")

        # Print each document's content
        print(
            f"\n{'-' * 100}\n".join(
                [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
            )
        )
    except Exception as e:
        print(f"An error occurred while printing documents: {e}")
