"""Retrieve transcript passages with a small local MiniLM embedding model."""

import argparse
import glob
import os
import re

from dotenv import load_dotenv
import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "chroma_store")
INDEX_READY = os.path.join(DB_DIR, "index_ready")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

load_dotenv(os.path.join(BASE_DIR, ".env"))


# 1. LOAD ---- read the cleaned transcripts (also accepts new VTT files).
def load_transcripts():
    docs = []
    paths = sorted(glob.glob(f"{DATA_DIR}/*.txt") + glob.glob(f"{DATA_DIR}/*.vtt"))
    for path in paths:
        lines = []
        with open(path, encoding="utf-8-sig") as transcript:
            for line in transcript:
                line = line.strip()
                if not line or line == "WEBVTT" or "-->" in line:
                    continue
                lines.append(line)

        text = " ".join(lines)
        if not text:
            continue
        filename = os.path.basename(path)
        session = re.search(r"Session[ _]*(\d+)", filename)
        docs.append(Document(
            page_content=text,
            metadata={"session": session.group(1) if session else "unknown", "source": filename},
        ))
    return docs


# 2. BUILD ---- embed once and keep the vectors on disk.
def load_store():
    # Limit CPU use. This model needs no GPU or paid embedding API.
    torch.set_num_threads(2)
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        cache_folder=os.path.join(BASE_DIR, ".model_cache"),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
    )
    store = Chroma(
        collection_name="transcripts",
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )
    ready_model = ""
    if os.path.exists(INDEX_READY):
        with open(INDEX_READY, encoding="utf-8") as ready:
            ready_model = ready.read().strip()
    if ready_model == MODEL_NAME and store.get(limit=1)["ids"]:
        return store

    # Rebuild if indexing was interrupted or the embedding model changed.
    if store.get(limit=1)["ids"]:
        store.reset_collection()

    docs = load_transcripts()
    if not docs:
        raise ValueError(f"No transcripts found in {DATA_DIR}")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    ).split_documents(docs)
    print(f"Indexing {len(chunks)} chunks from {len(docs)} transcripts...")
    # Embed in small batches to keep memory use modest.
    for start in range(0, len(chunks), 32):
        store.add_documents(chunks[start : start + 32])
    with open(INDEX_READY, "w", encoding="utf-8") as ready:
        ready.write(MODEL_NAME + "\n")
    return store


def build_retriever(top_k=5):
    return load_store().as_retriever(search_kwargs={"k": top_k})


# 3. TRY IT ---- uv run python main.py "What is regression testing?"
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default="What is regression testing?")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if not args.question.strip() or args.top_k < 1:
        parser.error("Provide a nonempty question and --top-k of at least 1.")

    retriever = build_retriever(top_k=args.top_k)
    for result in retriever.invoke(args.question):
        print(f"[Session {result.metadata['session']}] {result.metadata['source']}")
        print(f"{result.page_content}\n")


if __name__ == "__main__":
    main()
