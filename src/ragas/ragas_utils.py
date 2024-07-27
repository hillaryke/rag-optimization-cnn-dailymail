import pandas as pd

def load_evaluation_data(csv_file_path: str = "data/evaluation_set.csv") -> dict:
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
