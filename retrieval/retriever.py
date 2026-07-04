"""Cross-modal retrieval: query both collections, threshold weak matches.

See architecture.md Section 3.4.
"""

from dataclasses import dataclass, field
from typing import Any

import chromadb

from embedding.clip_embedder import embed_clip_text_query
from embedding.text_embedder import embed_text_query
from store import vector_store as vs

DEFAULT_TEXT_TOP_K = 4
DEFAULT_IMAGE_TOP_K = 2

# Separate defaults per modality: sentence-transformers cosine similarities
# span a wide range (near 0 for unrelated, 0.6+ for related), but CLIP's
# cross-modal text-to-image similarities sit in a much narrower band
# regardless of relevance (empirically ~0.18-0.30 here), so the same cutoff
# doesn't work for both.
DEFAULT_TEXT_RELEVANCE_THRESHOLD = 0.2
DEFAULT_IMAGE_RELEVANCE_THRESHOLD = 0.25


@dataclass
class RetrievedItem:
    id: str
    modality: str  # "text" | "image"
    score: float  # cosine similarity, higher is more relevant
    content: Any  # chunk text (str) for text items, PIL.Image for image items
    metadata: dict


@dataclass
class RetrievalResult:
    text_items: list[RetrievedItem] = field(default_factory=list)
    image_items: list[RetrievedItem] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.text_items and not self.image_items


def _distance_to_similarity(distance: float) -> float:
    """Chroma's cosine space reports distance = 1 - cosine_similarity."""
    return 1 - distance


def _apply_threshold(items: list[RetrievedItem], threshold: float) -> list[RetrievedItem]:
    """Drop the whole result set if even the best match is below threshold.

    Chroma already returns results sorted by ascending distance (descending
    similarity), so items[0] is the best match.
    """
    if not items or items[0].score < threshold:
        return []
    return items


def retrieve(
    client: chromadb.ClientAPI,
    question: str,
    text_top_k: int = DEFAULT_TEXT_TOP_K,
    image_top_k: int = DEFAULT_IMAGE_TOP_K,
    text_relevance_threshold: float = DEFAULT_TEXT_RELEVANCE_THRESHOLD,
    image_relevance_threshold: float = DEFAULT_IMAGE_RELEVANCE_THRESHOLD,
) -> RetrievalResult:
    """Embed `question` with both encoders and query both collections.

    Each modality is thresholded independently: if a modality's best match is
    below its relevance threshold, that modality's results are dropped
    entirely rather than forcing in a weak match (architecture.md Section 3.4).
    """
    text_query_vector = embed_text_query(question)
    clip_query_vector = embed_clip_text_query(question)

    text_raw = vs.query_text(client, text_query_vector, top_k=text_top_k)
    image_raw = vs.query_images(client, clip_query_vector, top_k=image_top_k)

    text_items = [
        RetrievedItem(
            id=item_id,
            modality="text",
            score=_distance_to_similarity(distance),
            content=document,
            metadata=metadata,
        )
        for item_id, distance, document, metadata in zip(
            text_raw["ids"][0],
            text_raw["distances"][0],
            text_raw["documents"][0],
            text_raw["metadatas"][0],
        )
    ]
    image_items = [
        RetrievedItem(
            id=item_id,
            modality="image",
            score=_distance_to_similarity(distance),
            content=vs.get_cached_image(item_id),
            metadata=metadata,
        )
        for item_id, distance, metadata in zip(
            image_raw["ids"][0],
            image_raw["distances"][0],
            image_raw["metadatas"][0],
        )
    ]

    return RetrievalResult(
        text_items=_apply_threshold(text_items, text_relevance_threshold),
        image_items=_apply_threshold(image_items, image_relevance_threshold),
    )
