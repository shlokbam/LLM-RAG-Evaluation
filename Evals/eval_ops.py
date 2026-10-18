"""
Evals/eval_ops.py -- unified runner for all operational evaluations.

Evaluates operational metrics for the open-mistral-7b RAG pipeline:
  1. Cost (eval_cost.py: derived token economics at $0.25/1M in/out tokens)
  2. Latency & TTFT (eval_latency.py: percentile distributions & SLOs)
  3. Reliability (eval_reliability.py: success, error, and retry rates)

Returns a flat dictionary suitable for prefixing under the 'ops' namespace
in run_suite.py and metric_registry.py.
"""

from dotenv import load_dotenv

from src.rag_pipeline import RagPipeline
from Evals import eval_cost, eval_latency, eval_reliability

load_dotenv()


def run_ops(rag, verbose=True):
    """
    Run all operational evaluations (Cost, Latency, Reliability)
    for the open-mistral-7b pipeline.
    """
    flat = {}

    # 1. Cost Evaluation (open-mistral-7b official pricing)
    if verbose:
        print("  -> measuring operational cost (open-mistral-7b @ $0.25/$0.25 per 1M tokens)...")
    cost_rows = eval_cost.benchmark(rag)
    avg_cost = eval_cost.avg(cost_rows, "cost_total")
    avg_in = eval_cost.avg(cost_rows, "input")
    avg_out = eval_cost.avg(cost_rows, "output")
    avg_cached = eval_cost.avg(cost_rows, "cached")
    cost_pass = avg_cost <= eval_cost.COST_BUDGET_PER_QUERY_USD

    flat["cost.cost_per_query_usd"] = round(avg_cost, 6)
    flat["cost.budget_pass"] = cost_pass
    flat["cost.avg_input_tokens"] = round(avg_in, 1)
    flat["cost.avg_output_tokens"] = round(avg_out, 1)
    flat["cost.avg_cached_tokens"] = round(avg_cached, 1)
    flat["cost.daily_usd"] = round(avg_cost * eval_cost.QUERIES_PER_DAY, 2)
    flat["cost.monthly_usd"] = round(avg_cost * eval_cost.QUERIES_PER_DAY * 30, 2)

    # 2. Latency Evaluation (end-to-end and perceived TTFT)
    if verbose:
        print("  -> measuring operational latency (open-mistral-7b generation + reranker)...")
    lat_results = eval_latency.benchmark(rag)
    total_sum = eval_latency.summarize(lat_results["total"])
    ttft_sum = eval_latency.summarize(lat_results["ttft"]) if lat_results.get("ttft") else {}
    ret_sum = eval_latency.summarize(lat_results["retrieval"]) if lat_results.get("retrieval") else {}
    gen_sum = eval_latency.summarize(lat_results["generation"]) if lat_results.get("generation") else {}

    flat["latency.e2e_p95_ms"] = round(total_sum["p95"], 1)
    flat["latency.e2e_p50_ms"] = round(total_sum["p50"], 1)
    flat["latency.e2e_mean_ms"] = round(total_sum["mean"], 1)
    flat["latency.full_answer_pass"] = total_sum["p95"] <= eval_latency.SLO_P95_MS

    if ttft_sum:
        flat["latency.ttft_p95_ms"] = round(ttft_sum["p95"], 1)
        flat["latency.ttft_p50_ms"] = round(ttft_sum["p50"], 1)
        flat["latency.first_token_pass"] = ttft_sum["p95"] <= eval_latency.SLO_TTFT_P95_MS

    if ret_sum:
        flat["latency.retrieval_mean_ms"] = round(ret_sum["mean"], 1)
    if gen_sum:
        flat["latency.generation_mean_ms"] = round(gen_sum["mean"], 1)

    # 3. Reliability Evaluation
    if verbose:
        print("  -> measuring operational reliability (retries, errors, availability)...")
    rel = eval_reliability.benchmark(rag)
    success_rate = (100 * rel.successes / rel.calls) if rel.calls else 0.0
    error_rate = (100 * rel.failures / rel.calls) if rel.calls else 0.0
    retry_rate = (100 * rel.retries / rel.calls) if rel.calls else 0.0

    flat["reliability.success_rate"] = round(success_rate, 2)
    flat["reliability.error_rate"] = round(error_rate, 2)
    flat["reliability.retry_rate"] = round(retry_rate, 2)

    return flat


def main():
    rag = RagPipeline()
    results = run_ops(rag, verbose=True)
    print("\n" + "=" * 60)
    print("OPERATIONAL EVALUATION SUMMARY (open-mistral-7b)")
    print("=" * 60)
    for k, v in sorted(results.items()):
        print(f"  {k:<35} : {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
