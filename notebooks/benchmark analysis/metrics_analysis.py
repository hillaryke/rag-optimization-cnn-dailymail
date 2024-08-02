import os
import json
import pandas as pd


NUMERICAL_COLUMNS = [
    "answer_correctness",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
]


def update_json_with_metrics(csv_filename, json_filename, numeric_columns=None):
    """
    Read a CSV file, calculate averages and standard deviations for specified columns,
    and update or create a JSON file with the results.

    Parameters:
    csv_filename (str): The name of the CSV file to read.
    numeric_columns (list): A list of column names for which to calculate statistics.
    json_filename (str): The name of the JSON file to update or create.
    """
    if numeric_columns is None:
        numeric_columns = NUMERICAL_COLUMNS
    # Load the CSV file
    df = pd.read_csv(csv_filename)

    # Calculate averages and standard deviations
    statistics = {}
    for column in numeric_columns:
        if column in df.columns:
            avg = df[column].mean()
            std_dev = df[column].std()
            statistics[column] = {"average": avg, "std_dev": std_dev}
        else:
            statistics[column] = {
                "average": None,
                "std_dev": None,
                "error": f"Column '{column}' not found in DataFrame.",
            }

    # Get the key from the filename (remove .csv)
    key_name = os.path.splitext(os.path.basename(csv_filename))[0]

    # Load existing JSON data or create a new dictionary
    if os.path.exists(json_filename):
        with open(json_filename, "r", encoding="utf-8") as json_file:
            json_data = json.load(json_file)
    else:
        json_data = {}

    # Update the JSON data with the new statistics
    json_data[key_name] = statistics

    # Write the updated data back to the JSON file
    with open(json_filename, "w", encoding="utf-8") as json_file:
        json.dump(json_data, json_file, indent=4)


# Example usage
# csv_filename = 'baseline_results.csv'  # Replace with your actual CSV file name
# json_filename = 'benchmark_results.json'  # The JSON file to update/create

# update_json_with_metrics(bge_large, json_filename, numeric_columns)
