# eval_retriever.py
import os
import time
import asyncio
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.evaluate.configs import AsyncConfig
from langchain_mistralai import ChatMistralAI

from src.retriever import build_retriever
from Evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

GOLDEN_PATH = "Goldens/retriever_goldens.json"
import re

JUDGE_MODEL_NAME = "open-mistral-7b"
THRESHOLD = 0.7


import json
from json_repair import repair_json


def clean_json_output(res: str) -> str:
    """Strips preambles, markdown codeblocks, extra trailing blocks, and returns a single valid JSON string."""
    if not isinstance(res, str):
        return res
    res = res.strip()

    try:
        parsed = repair_json(res, return_objects=True)
        if isinstance(parsed, dict):
            if "verdicts" in parsed and isinstance(parsed["verdicts"], list):
                for item in parsed["verdicts"]:
                    if isinstance(item, dict) and "reason" not in item:
                        item["reason"] = "N/A"
            elif "verdict" in parsed and "reason" not in parsed:
                parsed["reason"] = "N/A"
            return json.dumps(parsed)
        elif isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "verdict" in item and "reason" not in item:
                    item["reason"] = "N/A"
            return json.dumps(parsed)
    except Exception:
        pass

    # Extract first complete `{ ... }` JSON object using brace stack tracking
    first_b = res.find("{")
    if first_b != -1:
        stack = 0
        end_b = -1
        for i in range(first_b, len(res)):
            if res[i] == "{":
                stack += 1
            elif res[i] == "}":
                stack -= 1
                if stack == 0:
                    end_b = i
                    break
        if end_b != -1:
            res = res[first_b:end_b + 1]

    try:
        parsed = repair_json(res, return_objects=True)
        if isinstance(parsed, (dict, list)):
            return json.dumps(parsed)
    except Exception:
        pass

    return res


# --- Custom Mistral Model Wrapper for DeepEval ---
class MistralLLM(DeepEvalBaseLLM):
    def __init__(self, model_name=JUDGE_MODEL_NAME):
        self.model_name = model_name
        self.model = ChatMistralAI(
            model=model_name,
            api_key=os.getenv("MISTRAL_API_KEY"),
            max_retries=5,
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema=None, *args, **kwargs) -> str:
        if schema is not None:
            raise TypeError("Schema not supported natively by MistralLLM")
        for attempt in range(5):
            try:
                res = self.model.invoke(prompt).content
                return clean_json_output(res)
            except Exception as e:
                if ("429" in str(e) or "rate" in str(e).lower()) and attempt < 4:
                    time.sleep(3 * (attempt + 1))
                else:
                    raise e

    async def a_generate(self, prompt: str, schema=None, *args, **kwargs) -> str:
        if schema is not None:
            raise TypeError("Schema not supported natively by MistralLLM")
        for attempt in range(5):
            try:
                res = await self.model.ainvoke(prompt)
                return clean_json_output(res.content)
            except Exception as e:
                if ("429" in str(e) or "rate" in str(e).lower()) and attempt < 4:
                    await asyncio.sleep(3 * (attempt + 1))
                else:
                    raise e

    def get_model_name(self):
        return self.model_name


mistral_judge_model = MistralLLM(model_name=JUDGE_MODEL_NAME)



def run(retriever, retriever_name: str = "bge-m3-chroma", top_k: int = 5):
    # 1. LOAD the golden set --- the fixed, human-authored truth
    goldens = load_goldens(GOLDEN_PATH)

    # 2. RUN THE INJECTED RETRIEVER on each question to fill retrieval_context,
    #    then build one test case per golden.
    test_cases = []
    for g in goldens:
        retrieved = retriever.invoke(g["query"])
        retrieval_context = [doc.page_content for doc in retrieved]

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated in this run)",
            )
        )

    # 3. THE METRICS --- recall (did we miss?) and precision (ranked well?)
    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=mistral_judge_model, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=mistral_judge_model, include_reason=True),
    ]

    # 4. EVALUATE --- every metric on every case, printed report.
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False),
        hyperparameters={
            "retriever": retriever_name,
            "embedding_model": "BAAI/bge-m3",
            "chunk_size": 1000,
            "chunk_overlap": 150,
            "top_k": top_k,
            "judge_model": JUDGE_MODEL_NAME,
            "golden_set": GOLDEN_PATH,
        },
    )
    return summarize_by_metric(result)


def run_local(use_reranker: bool = False, fetch_k: int = 25, top_k: int = 7):
    """Standalone convenience: build the retriever, then run."""
    retriever = build_retriever(use_reranker=use_reranker, fetch_k=fetch_k, top_k=top_k)
    name = f"bge-m3-chroma-reranked(fetch_{fetch_k}_top_{top_k})" if use_reranker else "bge-m3-chroma"
    return run(retriever, retriever_name=name, top_k=top_k)


if __name__ == "__main__":
    import sys
    use_reranker = "--rerank" in sys.argv or "-r" in sys.argv
    retriever_label = "reranking_retriever" if use_reranker else "retriever"
    print_summary(retriever_label, run_local(use_reranker=use_reranker))