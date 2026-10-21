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
from src.retriever import build_retriever

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

To generate an answer from retrieved chunks, use
[src/generator.py](src/generator.py):

```python
from src.retriever import build_retriever
from src.generator import generate, generate_stream

question = "What is online eval?"
docs = build_retriever().invoke(question)
context = [doc.page_content for doc in docs]
print(generate(question, context))

# Or stream the answer:
# for piece in generate_stream(question, context):
#     print(piece, end="", flush=True)
```

The generator uses `GROQ_API_KEY` from `.env` and defaults to
`openai/gpt-oss-120b`. Set `GENERATOR_MODEL` to select another Groq model.
Both entry points use the same prompt and abstain when no context is supplied.

Evaluate application toxicity on the first two prompts in
[goldens/toxicity_goldens.json](goldens/toxicity_goldens.json):

```sh
uv run python -m eval.eval_toxicity
```

Each prompt runs through the full RAG pipeline. DeepEval's built-in
`ToxicityMetric` judges the live answer using the existing `GROQ_API_KEY` and
`EVAL_MODEL` (default: `openai/gpt-oss-120b`). With this project's DeepEval 3.x,
lower toxicity is better and scores at or below `0.3` pass. The run prints
scores, reasons, and a per-metric summary, and saves reports under
`.deepeval/toxicity_results/`. This evaluation does not require reference answers.

Evaluate leakage with the first two rows of each subtype in
[goldens/leakage_goldens.json](goldens/leakage_goldens.json) (six responses total):

```sh
uv run python -m eval.eval_leakage
```

This uses the same Groq judge configuration. Prompt and course-content leakage
use GEval with `expected_action` as the reference and a passing score of at least
`0.7`. PII leakage uses DeepEval's built-in `PIILeakageMetric` with a passing score
of at least `0.9`. Higher scores mean better protection for all three metrics.
Reports save separately by subtype under `.deepeval/leakage_results/`.

Evaluate scope safety with the first two rows of
[goldens/scope_goldens.json](goldens/scope_goldens.json):

```sh
uv run python -m eval.eval_scope
```

This runs the full RAG pipeline and uses GEval's `Scope Adherence` metric with
the existing Groq judge. Each row's `expected_action` and `success_criteria`
define the expected behavior; scores at or above `0.7` pass. Evaluation runs
sequentially and saves reports under `.deepeval/scope_results/`.
The default `LIMIT = 2` selects two ANSWER cases in the current dataset.
To include all ANSWER, DECLINE, and PARTIAL cases, remove `[:LIMIT]` from the
golden-loading line in `eval/eval_scope.py`.
