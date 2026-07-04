"""Chroma vector store: two collections, one per modality.

See architecture.md Section 3.3. Text and image embeddings come from
different models with different vector spaces, so they're kept in separate
collections and queried independently; combining results is retrieval's job
(retrieval/retriever.py).
"""

from pathlib import Path
from typing import Any

import chromadb

DEFAULT_PERSIST_DIR = Path(__file__).parent.parent / "data" / "chroma_db"

TEXT_COLLECTION_NAME = "text_chunks"
IMAGE_COLLECTION_NAME = "images"

# Chroma stores embeddings + metadata, not arbitrary Python objects, so PIL
# images can't live in a collection directly. Keep the actual image content
# in-process, keyed by id, so retrieval/generation can pass real images to
# the LLM after a similarity search returns ids + metadata.
_image_cache: dict[str, Any] = {}


def get_client(persist_directory: str | Path = DEFAULT_PERSIST_DIR) -> chromadb.ClientAPI:
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_directory))


def get_text_collection(client: chromadb.ClientAPI):
    return client.get_or_create_collection(TEXT_COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def get_image_collection(client: chromadb.ClientAPI):
    return client.get_or_create_collection(IMAGE_COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def add_text_records(client: chromadb.ClientAPI, records: list[dict]) -> None:
    """Insert embedded text-chunk records (each with an "embedding" field)."""
    if not records:
        return
    collection = get_text_collection(client)
    collection.upsert(
        ids=[r["id"] for r in records],
        embeddings=[r["embedding"] for r in records],
        documents=[r["content"] for r in records],
        metadatas=[r["metadata"] for r in records],
    )


def add_image_records(client: chromadb.ClientAPI, records: list[dict]) -> None:
    """Insert embedded image records (each with an "embedding" field)."""
    if not records:
        return
    collection = get_image_collection(client)
    collection.upsert(
        ids=[r["id"] for r in records],
        embeddings=[r["embedding"] for r in records],
        metadatas=[r["metadata"] for r in records],
    )
    for r in records:
        _image_cache[r["id"]] = r["content"]


def get_cached_image(image_id: str) -> Any | None:
    """Look up the original PIL.Image for an id returned by query_images."""
    return _image_cache.get(image_id)


def query_text(client: chromadb.ClientAPI, query_embedding: list[float], top_k: int) -> dict:
    collection = get_text_collection(client)
    n_results = min(top_k, collection.count())
    if n_results == 0:
        return {"ids": [[]], "distances": [[]], "metadatas": [[]], "documents": [[]]}
    return collection.query(query_embeddings=[query_embedding], n_results=n_results)


def query_images(client: chromadb.ClientAPI, query_embedding: list[float], top_k: int) -> dict:
    collection = get_image_collection(client)
    n_results = min(top_k, collection.count())
    if n_results == 0:
        return {"ids": [[]], "distances": [[]], "metadatas": [[]]}
    return collection.query(query_embeddings=[query_embedding], n_results=n_results)


def clear(client: chromadb.ClientAPI) -> None:
    """Reset both collections (backs the UI's "clear index" button)."""
    for name in (TEXT_COLLECTION_NAME, IMAGE_COLLECTION_NAME):
        try:
            client.delete_collection(name)
        except (ValueError, chromadb.errors.NotFoundError):
            pass
    _image_cache.clear()
