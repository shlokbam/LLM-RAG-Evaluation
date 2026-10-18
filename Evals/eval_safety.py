"""
Evals/eval_safety.py -- unified runner for all safety evaluations.

Evaluates the RAG pipeline across three safety dimensions judged by open-mistral-7b:
  1. Toxicity (eval_toxicity.py)
  2. Scope Adherence & Role Boundary Safety (eval_scope_safety.py)
  3. Data & System Leakage (eval_leakage.py: Prompt, Content, PII)

Returns a flat dictionary suitable for prefixing under the 'safety' namespace
in run_suite.py and metric_registry.py.
"""

from dotenv import load_dotenv

from src.rag_pipeline import RagPipeline
from Evals import eval_toxicity, eval_scope_safety, eval_leakage

load_dotenv()


def run_safety(rag, verbose=True):
    """
    Run all safety evaluations against a single live pipeline instance.
    Judged by open-mistral-7b (MistralLLM).
    """
    if verbose:
        print("  -> running toxicity eval (open-mistral-7b judge)...")
    tox_summary = eval_toxicity.run(rag)

    if verbose:
        print("  -> running scope safety eval (open-mistral-7b judge)...")
    scope_summary = eval_scope_safety.run(rag)

    if verbose:
        print("  -> running data & system leakage evals (open-mistral-7b judge)...")
    leak_summary = eval_leakage.run(rag)

    flat = {}

    # 1. Toxicity
    if "Toxicity" in tox_summary:
        tox = tox_summary["Toxicity"]
        flat["toxicity.avg_score"] = tox["avg_score"]
        flat["toxicity.avg_toxicity"] = tox["avg_score"]
        flat["toxicity.pass_rate"] = tox["pass_rate"]
        flat["toxicity.total_cases"] = tox["total_cases"]

    # 2. Scope Adherence
    if "Scope Adherence" in scope_summary:
        scope = scope_summary["Scope Adherence"]
        flat["scope.avg_score"] = scope["avg_score"]
        flat["scope.pass_rate"] = scope["pass_rate"]
        flat["scope.total_cases"] = scope["total_cases"]

    # 3. Leakage (Prompt, Course Content, PII)
    if "Prompt Leakage" in leak_summary:
        flat["leakage.prompt_avg_score"] = leak_summary["Prompt Leakage"]["avg_score"]
        flat["leakage.prompt_pass_rate"] = leak_summary["Prompt Leakage"]["pass_rate"]

    if "Course Content Leakage" in leak_summary:
        flat["leakage.protected_avg_score"] = leak_summary["Course Content Leakage"]["avg_score"]
        flat["leakage.protected_pass_rate"] = leak_summary["Course Content Leakage"]["pass_rate"]

    if "PII Leakage" in leak_summary:
        flat["leakage.pii_avg_score"] = leak_summary["PII Leakage"]["avg_score"]
        flat["leakage.pii_pass_rate"] = leak_summary["PII Leakage"]["pass_rate"]

    # Aggregate leakage pass rate
    leak_passed = sum(v["passed"] if "passed" in v else int(v["pass_rate"] * v["total_cases"])
                      for v in leak_summary.values())
    leak_total = sum(v["total_cases"] for v in leak_summary.values())
    flat["leakage.pass_rate"] = round(leak_passed / leak_total, 4) if leak_total else 1.0

    return flat


def main():
    rag = RagPipeline()
    results = run_safety(rag, verbose=True)
    print("\n" + "=" * 60)
    print("SAFETY EVALUATION SUMMARY (open-mistral-7b judge)")
    print("=" * 60)
    for k, v in sorted(results.items()):
        print(f"  {k:<35} : {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
