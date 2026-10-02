# eval_rag_pipeline.py
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)
from deepeval.evaluate.configs import AsyncConfig

from src.rag_pipeline import RagPipeline
from Evals.harness import load_goldens, summarize_by_metric, print_summary
from Evals.eval_retriever import MistralLLM

load_dotenv()

GOLDEN_PATH = "Goldens/faithfulness_dataset.json"   # reuse the queries
JUDGE_MODEL_NAME = "open-mistral-7b"
THRESHOLD = 0.7

judge_model = MistralLLM(model_name=JUDGE_MODEL_NAME)


def run(rag):
    # 1. LOAD queries (we only need the queries --- context comes from the pipeline now)
    goldens = load_goldens(GOLDEN_PATH)

    # 2. RUN THE INJECTED PIPELINE per query, build a test case from LIVE output
    test_cases = []
    for g in goldens:
        result = rag.invoke(g["query"])          # retrieve -> rerank -> generate

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=result["answer"],       # what the generator produced
                retrieval_context=result["context"],  # what the RETRIEVER returned
            )
        )

    # 3. THE THREE TRIAD METRICS
    metrics = [
        ContextualRelevancyMetric(threshold=THRESHOLD, model=judge_model, include_reason=True),
        FaithfulnessMetric(threshold=THRESHOLD, model=judge_model, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=judge_model, include_reason=True),
    ]

    # 4. EVALUATE
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False),
    )
    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: build the pipeline, then run."""
    return run(RagPipeline())


if __name__ == "__main__":
    print_summary("rag_pipeline", run_local())