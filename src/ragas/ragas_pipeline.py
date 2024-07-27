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
from typing import List, Dict
from langchain.docstore.document import Document

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


def get_context_and_answer(
    evaluation_data: List[Dict[str, List[str]]],
    rag_chain, 
) -> List[Dict[str, str]]:

    results = []

    for question, ground_truth in zip(
        evaluation_data["questions"], evaluation_data["ground_truths"]
    ):
        result = rag_chain.invoke(question)
        contexts = [doc.page_content for doc in result["source_documents"]]
        concatenated_context = " ".join(contexts)  

        results.append(
            {
                "question": question,
                "context": concatenated_context,
                "answer": result["result"],
                "ground_truth": ground_truth,
            }
        )

    return results