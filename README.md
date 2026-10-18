# Simple transcript retriever

Uses `sentence-transformers/all-MiniLM-L6-v2` locally and saves vectors in Chroma.
This is a small, 22.7-million-parameter embedding model with approximately 90 MB
of weights. The Python runtime and dependencies use additional disk space and
memory. No GPU or paid embedding API is needed.
[Model details](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

```sh
uv sync
uv run python main.py "What is regression testing?"
uv run python main.py "What is LLM evaluation?" --top-k 3
```

The code follows three steps:

1. `load_transcripts()` reads the cleaned transcripts in `data/`. New `.vtt`
   files are also accepted, with timestamp lines skipped.
2. `load_store()` splits transcripts into 1,000-character chunks with 150
   characters of overlap, embeds them, and saves vectors in `chroma_store/`.
3. `build_retriever()` returns the five most similar passages, with session and
   source metadata.

The first run downloads the model into `.model_cache/` and indexes the documents.
Later runs load the model and reuse the saved document vectors, embedding only
the new question. CPU execution is limited to two PyTorch threads, with small
embedding batches. First-time indexing uses more compute than searching.

No embedding API key is required. Existing keys in `.env` are preserved; Groq
can be used for evaluation and answer generation later.

```python
from rag_eval_deepeval.retriever import build_retriever

retriever = build_retriever()
results = retriever.invoke("What is regression testing?")
for result in results:
    print(result.metadata, result.page_content)
```

After editing transcripts or chunk settings, rename or remove `chroma_store/`
to rebuild the index. The model downloads only once and is reused from disk.

Run `uv run python eval/eval_retriever.py` to evaluate the first three goldens
with DeepEval's contextual precision and recall. The single evaluation script is
[eval/eval_retriever.py](eval/eval_retriever.py); it uses the Groq key in `.env`
and saves reports under `.deepeval/retriever_results/`.
