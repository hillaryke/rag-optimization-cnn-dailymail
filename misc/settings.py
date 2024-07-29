from typing import Final

class Settings:
    PG_CONNECTION_STRING: Final = os.getenv("PG_CONNECTION_STRING")
    COLLECTION_NAME: Final = "cnn_dailymail_validation_subset"
    SOURCE_FILE_PATH: Final = "data/cnn_dailymail_validation_subset.csv"
    CHUNK_SIZE: Final = 1000
    CHUNK_OVERLAP: Final = 200
    PAGE_CONTENT_COLUMN: Final = "article"
    
    GENERATOR_TEMPLATE: Final = """
      Use the following pieces of context to answer the question at the end.
      If you don't know the answer, just say that you don't know, don't try to make up an answer.
      Use three sentences maximum and keep the answer as concise as possible.

      Context: {context}

      Question: {question}

      Helpful Answer:
    """

    EVALUATION_FILE_PATH = "data/evaluation_sets/evaluation_set_20d20.csv"
    EVALUAION_DATASET_NAME: Final = "CNN DailyMail Evaluation Dataset"
    EVALUATION_DATASET_DESCRIPTION = """
      Evaluation dataset questions and ground truth answers for RAGAS   pipeline on cnn_dailymail dataset.
    """
    
