import pandas as pd
from datasets import Dataset
from ragas.integrations.langsmith import upload_dataset

from misc import Settings

EVALUATION_FILE_PATH = Settings.EVALUATION_FILE_PATH
EVALUAION_DATASET_NAME = Settings.EVALUAION_DATASET_NAME
EVALUATION_DATASET_DESCRIPTION = Settings.EVALUATION_DATASET_DESCRIPTION

def load_evaluation_data(csv_file_path: str = EVALUATION_FILE_PATH) -> dict:
    """Loads evaluation data from a CSV file and returns questions and ground truths.

    Args:
        csv_file_path (str): The path to the CSV file containing the evaluation data.
            The CSV should have columns named "question" and "ground_truth".

    Returns:
        dict: A dictionary containing:
            - "questions": A list of questions.
            - "ground_truths": A list of corresponding ground truth answers.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the CSV file does not contain the required columns.
        pd.errors.EmptyDataError: If the CSV file is empty.
        pd.errors.ParserError: If the CSV file is malformed.
    """

    try:
        df = pd.read_csv(csv_file_path)  # Read the CSV file
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at path '{csv_file_path}' was not found.")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"The file at path '{csv_file_path}' is empty.")
    except pd.errors.ParserError:
        raise pd.errors.ParserError(f"The file at path '{csv_file_path}' is malformed and cannot be parsed.")

    # Check if required columns are present
    if "question" not in df.columns or "ground_truth" not in df.columns:
        raise ValueError("The CSV file must contain 'question' and 'ground_truth' columns.")

    questions = df["question"].tolist()
    ground_truths = df["ground_truth"].tolist()

    return {"questions": questions, "ground_truths": ground_truths}

def upload_csv_dataset_to_langsmith(
    csv_file_path: str = EVALUATION_FILE_PATH,
    dataset_name: str = EVALUAION_DATASET_NAME, 
    dataset_desc: str = EVALUATION_DATASET_DESCRIPTION
) -> Dataset:
    """Uploads an evaluation dataset from a CSV file to LangSmith.

    Args:
        csv_file_path (str): The path to the CSV file containing the evaluation data.
        dataset_name (str): The name to give the dataset on LangSmith.
        dataset_desc (str): A description of the dataset for LangSmith.

    Returns:
        Dataset: The uploaded dataset object.
    """
    
    df = pd.read_csv(csv_file_path)
    eval_set = Dataset.from_pandas(df)

    dataset = upload_dataset(eval_set, dataset_name, dataset_desc)
    return dataset