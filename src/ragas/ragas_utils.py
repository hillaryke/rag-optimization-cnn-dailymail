import os
import sys
from dotenv import load_dotenv

load_dotenv()
PROJECT_PATH = os.environ.get('PROJECT_PATH')

# Add the project root path to sys.path
if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)

def load_evaluation_data(csv_file_path: str = "data/evaluation_set.csv") -> dict:
    """Loads evaluation data from a CSV file and returns questions and ground truths.

    Args:
        csv_file_path (str): The path to the CSV file containing the evaluation data.
            The CSV should have columns named "question" and "ground_truth".

    Returns:
        dict: A dictionary containing:
            - "questions": A list of questions.
            - "ground_truths": A list of corresponding ground truth answers.
    """

    df = pd.read_csv(csv_file_path)  # Read the CSV file
    
    # Check if required columns are present
    if "question" not in df.columns or "ground_truth" not in df.columns:
        raise ValueError("The CSV file must contain 'question' and 'ground_truth' columns.")

    questions = df["question"].tolist()
    ground_truths = df["ground_truth"].tolist()

    return {"questions": questions, "ground_truths": ground_truths}
