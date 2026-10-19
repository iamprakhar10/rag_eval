"""Shared loading and per-metric summaries for the eval suite.

Keep each metric separate so a regression cannot hide behind other metrics.
Support EvaluationResult.test_results or a bare list, and metrics_data or
legacy metrics. Pass rates use each metric's success flag, not its score.
"""

import json


def load_goldens(path):
    """Read a golden JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def summarize_by_metric(result):
    """Return {metric_name: {n, pass_rate, avg_score, min_score, max_score}}."""
    test_results = getattr(result, "test_results", None)
    if test_results is None:
        test_results = result if isinstance(result, list) else []

    buckets = {}
    for tr in test_results:
        metrics = getattr(tr, "metrics_data", None) or getattr(tr, "metrics", None) or []
        for metric in metrics:
            name = getattr(metric, "name", "unknown")
            bucket = buckets.setdefault(name, {"scores": [], "passed": 0, "total": 0})
            bucket["total"] += 1
            score = getattr(metric, "score", None)
            if score is not None:
                bucket["scores"].append(score)
            if getattr(metric, "success", False):
                bucket["passed"] += 1

    summary = {}
    for name, bucket in buckets.items():
        scores = bucket["scores"]
        summary[name] = {
            "n": bucket["total"],
            "pass_rate": 100 * bucket["passed"] / bucket["total"],
            "avg_score": sum(scores) / len(scores) if scores else float("nan"),
            "min_score": min(scores) if scores else float("nan"),
            "max_score": max(scores) if scores else float("nan"),
        }
    return summary


def print_summary(title, summary):
    """Print a per-metric recap beneath DeepEval's own report."""
    print("\n" + "=" * 60)
    print(f"{title}  (per-metric summary)")
    print("=" * 60)
    for name, stats in summary.items():
        print(
            f"  {name:<26} pass_rate={stats['pass_rate']:5.0f}%  "
            f"avg={stats['avg_score']:.2f}  n={stats['n']}"
        )
    print("=" * 60)
