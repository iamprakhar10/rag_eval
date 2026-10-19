"""Retrieve, rerank, and generate a grounded answer with its context.

Manual smoke test: uv run python -m src.rag_pipeline
"""

from langsmith import traceable

from src.generator import generate
from src.reranker import RerankingRetriever


class RagPipeline:
    def __init__(self, fetch_k=10, top_k=5):
        # Reuse one store and reranker model across queries.
        self.retriever = RerankingRetriever(fetch_k=fetch_k, top_k=top_k)

    @traceable(run_type="chain", name="RagPipeline")
    def invoke(self, query: str) -> dict:
        # 1. Retrieve extra candidates, then rerank down to top_k documents.
        docs = self.retriever.invoke(query)

        # 2. Use the same context strings for generation and evaluation.
        context = [doc.page_content for doc in docs]

        # 3. Generate an answer grounded in the retrieved context.
        answer = generate(query, context)

        return {
            "query": query,
            "context": context,
            "answer": answer,
        }


if __name__ == "__main__":
    rag = RagPipeline()
    result = rag.invoke("Why do we need golden datasets?")
    print("QUERY:  ", result["query"])
    print("ANSWER: ", result["answer"])
    print("\nCONTEXT CHUNKS:")
    for i, chunk in enumerate(result["context"]):
        print(f"  [{i}] {chunk[:120]}...")
