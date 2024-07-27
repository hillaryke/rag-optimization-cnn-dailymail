import os

from ragas import evaluate
from datasets import Dataset
from ragas.metrics import (
    answer_correctness,
    faithfulness,
    answer_relevancy,
    context_precision,
)

from src.ragas.ragas_utils import load_evaluation_data
from src import pretty_print_docs
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
    """Retrieves context and generates answers for each question in the evaluation data.

    Args:
        evaluation_data (Dict[str, List[str]]): A dictionary containing:
            - "questions": A list of questions.
            - "ground_truths": A list of corresponding ground truth answers.
        rag_chain: The RAG chain instance to use for retrieval and generation.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing:
            - "question": The original question.
            - "context": A string of concatenated relevant contexts.
            - "answer": The generated answer from the RAG chain.
            - "ground_truth": The ground truth answer (from the evaluation data).
    """

    results = []

    for question, ground_truth in zip(
        evaluation_data["questions"], evaluation_data["ground_truths"]
    ):
        response = rag_chain.invoke(question)
        contexts_list = [doc.page_content for doc in response["source_documents"]]
                
        test_data = {
            "question": question,
            "context": contexts_list,
            "answer": response["result"],
            "ground_truth": ground_truth,
        }

        results.append(test_data)

    return results
