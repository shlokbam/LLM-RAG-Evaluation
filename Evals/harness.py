import json

def load_goldens(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def summarize_by_metric(results):
    metrics_summary = {}
    
    test_results = []
    if hasattr(results, "test_results"):
        test_results = results.test_results
    elif isinstance(results, list):
        test_results = results

    for test_result in test_results:
        metrics = getattr(test_result, "metrics_data", None) or getattr(test_result, "metrics", []) or []
        for m in metrics:
            name = getattr(m, "name", str(type(m).__name__))
            score = getattr(m, "score", 0.0)
            passed = getattr(m, "success", getattr(m, "passed", False))
            if name not in metrics_summary:
                metrics_summary[name] = {"scores": [], "passed": 0, "total": 0}
            metrics_summary[name]["scores"].append(score)
            metrics_summary[name]["total"] += 1
            if passed:
                metrics_summary[name]["passed"] += 1

    summary = {}
    for name, data in metrics_summary.items():
        avg_score = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0.0
        pass_rate = data["passed"] / data["total"] if data["total"] else 0.0
        summary[name] = {
            "avg_score": round(avg_score, 4),
            "pass_rate": round(pass_rate, 4),
            "total_cases": data["total"],
        }
    return summary

def print_summary(eval_name, summary):
    print(f"\n==================== Evaluation Summary: {eval_name} ====================")
    if not summary:
        print("No evaluation summary data available.")
        return
    for metric_name, stats in summary.items():
        print(f"Metric: {metric_name}")
        print(f"  Avg Score : {stats['avg_score']}")
        print(f"  Pass Rate : {stats['pass_rate'] * 100:.1f}% ({stats['total_cases']} cases)")
    print("========================================================================\n")
