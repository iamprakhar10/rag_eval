"""Retrieve extra transcript chunks, then rerank them with a local cross-encoder."""

import os

from langsmith import traceable
from sentence_transformers import CrossEncoder

from src.retriever import BASE_DIR, load_store

# Small CPU-friendly reranker; downloaded once into the project's model cache.
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankingRetriever:
    def __init__(self, fetch_k=10, top_k=5):
        if not 1 <= top_k <= fetch_k:
            raise ValueError("Require 1 <= top_k <= fetch_k.")

        self.store = load_store()
        self.reranker = CrossEncoder(
            CROSS_ENCODER,
            device="cpu",
            cache_folder=os.path.join(BASE_DIR, ".model_cache"),
        )
        self.fetch_k = fetch_k
        self.top_k = top_k

    @traceable(run_type="retriever", name="RerankingRetriever")
    def invoke(self, query):
        """Return the best top_k documents, preserving their source metadata."""
        if not query.strip():
            raise ValueError("Provide a nonempty question.")

        # 1. Over-retrieve with the existing bi-encoder and Chroma store.
        candidates = self.store.similarity_search(query, k=self.fetch_k)
        if not candidates:
            return []

        # 2. Score each query and chunk together with the cross-encoder.
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs, batch_size=16, show_progress_bar=False)

        # 3. Sort by relevance and return the best documents.
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return [doc for doc, _ in ranked[:self.top_k]]
