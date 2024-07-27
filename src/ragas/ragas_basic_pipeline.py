import os
import pandas as pd

from ragas import evaluate
from datasets import Dataset
from ragas.metrics import (
    answer_correctness,
    faithfulness,
    answer_relevancy,
    context_precision,
)

from src.ragas.ragas_utils import load_evaluation_data
from typing import List, Dict, Any


def run_ragas_evaluation(
    rag_chain: Any,
    use_langsmith: bool = False,
    dataset_name: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Runs the evaluation of the RAG chain on the evaluation dataset.

    Args:
        rag_chain (Any): The RAG chain to evaluate.
        use_langsmith (bool, optional): If True, uploads results to LangSmith. Defaults to False.
        dataset_name (str, optional): Required if use_langsmith is True. The name of the dataset in LangSmith.
        experiment_name (str, optional): Required if use_langsmith is True. The name of the experiment in LangSmith.

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation results.
    """
    eval_data = load_evaluation_data()
    testset = get_context_and_answer(eval_data, rag_chain)

    # Evaluating test set on listed metrics
    result = evaluate(
        dataset=testset,
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

    results = {
        "question": [],
        "contexts": [],
        "answer": [],
        "ground_truth": [],
    }

    for question, ground_truth in zip(
        evaluation_data["questions"], evaluation_data["ground_truths"]
    ):
        response = rag_chain.invoke(question)
        contexts_list = [doc.page_content for doc in response["source_documents"]]
                
        results["question"].append(question)
        results["contexts"].append(contexts_list)
        results["answer"].append(response["result"])
        results["ground_truth"].append(ground_truth)
        
    dataset = Dataset.from_dict(results)
    return dataset