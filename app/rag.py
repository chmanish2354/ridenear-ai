"""Retrieve rental-guide chunks and answer only from those texts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import httpx
from fastapi import HTTPException

COLLECTION_NAME = "rental_policies"
KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"
TOP_K = 3

AnswerFn = Callable[[str, list[str]], str]


def create_embedding_function():
    # ONNX MiniLM is the same model as all-MiniLM-L6-v2 without the PyTorch runtime,
    # so the free host (512 MB) can load the index.
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    return ONNXMiniLM_L6_V2()


def passes_score_gate(best_score: float, min_score: float) -> bool:
    return best_score >= min_score


def min_score() -> float:
    raw = os.environ.get("RAG_MIN_SCORE", "0.35")
    try:
        return float(raw)
    except ValueError:
        return 0.35


def load_chunks(directory: Path | None = None) -> list[dict[str, str]]:
    root = directory or KNOWLEDGE_DIR
    names = [
        "cancellation.md",
        "late-return.md",
        "insurance.md",
        "security-deposit.md",
        "fuel-policy.md",
    ]
    chunks: list[dict[str, str]] = []
    for name in names:
        document = ""
        chunk_id = ""
        body: list[str] = []

        def flush() -> None:
            nonlocal chunk_id, body
            text = " ".join(line.strip() for line in body if line.strip()).strip()
            if chunk_id and text:
                chunks.append({"id": chunk_id, "document": document, "text": text})
            chunk_id = ""
            body = []

        for line in (root / name).read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                flush()
                document = line[2:].strip()
            elif line.startswith("## "):
                flush()
                chunk_id = line[3:].strip()
            else:
                body.append(line)
        flush()
    return chunks


def chroma_path() -> str:
    configured = os.environ.get("CHROMA_PATH")
    if configured:
        return configured
    return str(KNOWLEDGE_DIR.parent / "data" / "chroma")


def open_collection(client, embedding_function):
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(collection, chunks: list[dict[str, str]]) -> None:
    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[
            {"document": chunk["document"], "chunkId": chunk["id"]} for chunk in chunks
        ],
    )


def similarity_from_cosine_distance(distance: float) -> float:
    return 1.0 - float(distance)


def retrieve(collection, query: str, limit: int = TOP_K) -> list[dict[str, object]]:
    count = collection.count()
    if count <= 0 or not query:
        return []
    found = collection.query(
        query_texts=[query],
        n_results=min(limit, count),
        include=["documents", "metadatas", "distances"],
    )
    ids = found["ids"][0]
    documents = found["documents"][0]
    metadatas = found["metadatas"][0]
    distances = found["distances"][0]
    hits: list[dict[str, object]] = []
    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] or {}
        hits.append(
            {
                "document": str(metadata.get("document", "")),
                "chunkId": str(metadata.get("chunkId", chunk_id)),
                "score": similarity_from_cosine_distance(distances[index]),
                "text": str(documents[index]),
            }
        )
    return hits


def compose_answer(
    query: str,
    hits: list[dict[str, object]],
    *,
    score_floor: float,
    complete: AnswerFn,
) -> dict[str, object]:
    kept = hits[:TOP_K]
    best = float(kept[0]["score"]) if kept else None
    if best is None or not passes_score_gate(best, score_floor):
        return {"answer": "", "citations": [], "grounded": False}
    texts = [str(hit["text"]) for hit in kept]
    prose = complete(query, texts)
    citations = [
        {
            "document": hit["document"],
            "chunkId": hit["chunkId"],
            "score": float(hit["score"]),
        }
        for hit in kept
    ]
    return {"answer": prose, "citations": citations, "grounded": True}


def complete_answer(query: str, excerpts: list[str]) -> str:
    base = os.environ.get("LLM_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("LLM_API_KEY", "")
    model = os.environ.get("LLM_MODEL") or "gpt-4o-mini"
    if not base or not api_key:
        raise HTTPException(status_code=502, detail="Answer model unavailable")

    payload = {
        "model": model,
        "temperature": 0.2,
        "tools": [],
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer the question using only the rental guide excerpts. "
                    "Do not add facts that are not in the excerpts."
                ),
            },
            {
                "role": "user",
                "content": "Question:\n" + query + "\n\nExcerpts:\n" + "\n\n".join(excerpts),
            },
        ],
    }
    try:
        response = httpx.post(
            f"{base}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=6,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Answer model unavailable") from exc
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="Answer model unavailable")
    try:
        body = response.json()
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="Answer model unavailable") from exc
    if not isinstance(content, str):
        raise HTTPException(status_code=502, detail="Answer model unavailable")
    return content


_embedding_factory = create_embedding_function
_answer_fn: AnswerFn = complete_answer


def set_embedding_factory(factory) -> None:
    global _embedding_factory
    _embedding_factory = factory


def set_answer_fn(fn: AnswerFn) -> None:
    global _answer_fn
    _answer_fn = fn


class PolicyIndex:
    def __init__(self, client, collection, chunks: list[dict[str, str]]):
        self.client = client
        self.collection = collection
        self.chunks = chunks

    def ingest(self) -> None:
        upsert_chunks(self.collection, self.chunks)

    def vector_count(self) -> int:
        return int(self.collection.count())

    def close(self) -> None:
        self.client.close()

    def answer(self, query: str) -> dict[str, object]:
        hits = retrieve(self.collection, query, TOP_K)
        return compose_answer(
            query,
            hits,
            score_floor=min_score(),
            complete=_answer_fn,
        )


def build_index() -> PolicyIndex:
    import chromadb

    chunks = load_chunks()
    client = chromadb.PersistentClient(path=chroma_path())
    collection = open_collection(client, _embedding_factory())
    index = PolicyIndex(client, collection, chunks)
    index.ingest()
    return index
