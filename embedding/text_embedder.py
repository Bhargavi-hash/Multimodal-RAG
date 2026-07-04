"""Text embedding via sentence-transformers (all-MiniLM-L6-v2).

See architecture.md Section 3.2.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def embed_text_records(records: list[dict]) -> list[dict]:
    """Add an "embedding" field to each text ingestion record.

    Takes the dicts produced by ingestion.pdf_loader.load_pdf (modality "text")
    and returns new dicts with an added "embedding": list[float] field.
    """
    if not records:
        return []

    model = _get_model()
    texts = [record["content"] for record in records]
    vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    return [{**record, "embedding": vector.tolist()} for record, vector in zip(records, vectors)]


def embed_text_query(query: str) -> list[float]:
    """Embed a user's text query with the same model used for text chunks."""
    model = _get_model()
    vector = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    return vector.tolist()
