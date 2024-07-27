import os
os.chdir("../../")

from ragas import evaluate
from datasets import Dataset
from ragas.metrics import (
    answer_correctness,
    faithfulness,
    answer_relevancy,
    context_precision,
)

from src.ragas.ragas_utils import load_evaluation_data

def evaluate_metrics(dataset):
  # evaluating dataest on listed metrics
  result = evaluate(
      dataset=dataset,
      metrics=[
          answer_correctness,
          faithfulness,
          answer_relevancy,
          context_precision
      ]
  )


  df_results = result.to_pandas()

  return df_results
