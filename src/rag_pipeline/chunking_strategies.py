from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

def chunk_by_recursive_split(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 0) -> list[Document]:
    """
    Splits documents into chunks of a specified size using a recursive character-based approach.

    This function takes a list of documents and splits each one into smaller chunks based on a specified character count,
    using a RecursiveCharacterTextSplitter. This method does not consider semantic content, and splits are based purely on
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

        document = chunks[10] if len(chunks) > 10 else None
        if document:
            print(document.page_content)
            print(document.metadata)
    except Exception as e:
        print(f"Error during recursive split: {e}")
        chunks = []  # Ensure chunks is defined even in case of error
    return chunks