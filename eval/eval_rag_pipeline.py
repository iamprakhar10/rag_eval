"""Evaluate the full RAG pipeline on two rows.

Run manually: uv run python -m eval.eval_rag_pipeline
Context comes from live retrieval and reranking, not the golden context.
"""

import os
import sys
from pathlib import Path

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, DisplayConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

ROOT = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(ROOT))

from eval.eval_generator import GroqJudge
from eval.harness import load_goldens, print_summary, summarize_by_metric
from src.rag_pipeline import RagPipeline

load_dotenv(ROOT / ".env")

GOLDEN_PATH = ROOT / "goldens/faithfulness_dataset.json"
JUDGE_MODEL = os.getenv("EVAL_MODEL", "openai/gpt-oss-120b")
THRESHOLD = 0.7
LIMIT = 2


def run(rag):
    """Evaluate an injected pipeline using live answers and retrieved context."""
    # 1. Load only the first two queries; golden context is not used.
    goldens = load_goldens(GOLDEN_PATH)[:LIMIT]

    # 2. Run retrieve -> rerank -> generate and capture the actual output.
    test_cases = []
    for golden in goldens:
        print(f"Running pipeline for {golden['id']}...", flush=True)
        output = rag.invoke(golden["query"])
        test_cases.append(LLMTestCase(
            name=golden["id"],
            input=golden["query"],
            actual_output=output["answer"],
            retrieval_context=output["context"],
        ))

    # 3. Score each leg of the RAG triad separately.
    judge = GroqJudge(model=JUDGE_MODEL)
    metrics = [
        ContextualRelevancyMetric(
            threshold=THRESHOLD, model=judge, include_reason=True, async_mode=False,
        ),
        FaithfulnessMetric(
            threshold=THRESHOLD, model=judge, include_reason=True, async_mode=False,
        ),
        AnswerRelevancyMetric(
            threshold=THRESHOLD, model=judge, include_reason=True, async_mode=False,
        ),
    ]

    # 4. Evaluate sequentially and return a per-metric summary.
    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=False),
        display_config=DisplayConfig(results_folder=str(ROOT / ".deepeval/rag_pipeline_results")),
    )
    return summarize_by_metric(result)


def run_local():
    """Build the pipeline once, then evaluate it."""
    return run(RagPipeline())


if __name__ == "__main__":
    print_summary("rag_pipeline", run_local())
