from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai.embeddings import OpenAIEmbeddings

def chunk_by_recursive_split(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 0) -> list[Document]:
    """
    Splits a list documents into chunks of a specified size using a recursive character-based approach. Splits are based purely on
    character count.

    Args:
        documents (list[Document]): A list of Document objects to be split into smaller chunks.
        chunk_size (int, optional): The desired number of characters in each chunk. Defaults to 400.

    Returns:
        list[Document]: A list of Document objects, each representing a chunk of the original documents.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"--Split {len(documents)} documents into {len(chunks)} chunks.--")

    except Exception as e:
        print(f"Error during recursive split: {e}")
        chunks = []  # Ensure chunks is defined even in case of error
    return chunks