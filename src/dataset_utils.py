from typing import List
from langchain.docstore.document import Document
from datasets import load_dataset

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
