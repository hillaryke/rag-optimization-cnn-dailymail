import pandas as pd

from typing import List
from langchain.docstore.document import Document
from datasets import load_dataset

from misc import Settings

PAGE_CONTENT_COLUMN = Settings.PAGE_CONTENT_COLUMN

def load_and_process_dataset(page_content_column: str = PAGE_CONTENT_COLUMN, split: str = "validation[:1000]", as_document: bool = False) -> List[Document]:
    """Loads the CNN/Daily Mail dataset, creates LangChain Documents, and returns them.

    Args:
        split (str, optional): The dataset split to load (e.g., "train", "validation[:1000]").
            Defaults to "validation[:1000]".
        as_document (bool): If True, returns the documents as LangChain Document objects.


    Returns:
        List[Document]: A list of LangChain Document objects.
    """

    # Load the dataset from huggingface
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation[:1000]")
    
    if as_document:
        documents = [
            Document(page_content=article[page_content_column], metadata={"source": "cnn_dailymail"})
            for article in dataset
        ]
    else:
        documents = [article[page_content_column] for article in dataset]

    return documents

def load_docs_from_csv(
    file_path: str = "data/cnn_dailymail_validation_subset.csv",
    page_content_column: str = "article", 
    as_document: bool = False
) -> List[Document]:
    """Loads documents from a CSV file and returns them as a list of LangChain Document objects.

    Args:
        file_path (str): The path to the CSV file.
        page_content_column (str): The name of the column containing the page content.
        as_document (bool): If True, returns the documents as LangChain Document objects.

    Returns:
        List[Document]: A list of LangChain Document objects.
    """
    df = pd.read_csv(file_path)
    if as_document:
        return [
            Document(
                page_content=row[page_content_column], 
                metadata={"source": "cnn_dailymail", "id": row["id"]}
            ) for _, row in df.iterrows()]
    else:
        return df[page_content_column].tolist()
