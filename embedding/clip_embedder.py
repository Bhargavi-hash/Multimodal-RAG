"""Image + cross-modal text embedding via CLIP (openai/clip-vit-base-patch32).

CLIP's image encoder and text encoder share one coordinate space, so a text
query embedded with `embed_clip_text_query` can be compared directly against
image embeddings produced by `embed_image_records`. See architecture.md
Section 3.2.
"""

from functools import lru_cache

import torch
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512


@lru_cache(maxsize=1)
def _get_model_and_processor() -> tuple[CLIPModel, CLIPProcessor]:
    model = CLIPModel.from_pretrained(MODEL_NAME)
    model.eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor


def embed_image_records(records: list[dict]) -> list[dict]:
    """Add an "embedding" field to each image ingestion record.

    Takes the dicts produced by ingestion.image_loader.load_image (modality
    "image") and returns new dicts with an added "embedding": list[float] field.
    """
    if not records:
        return []

    model, processor = _get_model_and_processor()
    images = [record["content"] for record in records]
    inputs = processor(images=images, return_tensors="pt")

    with torch.no_grad():
        features = model.get_image_features(**inputs).pooler_output
    features = features / features.norm(p=2, dim=-1, keepdim=True)

    return [{**record, "embedding": vector.tolist()} for record, vector in zip(records, features)]


def embed_clip_text_query(query: str) -> list[float]:
    """Embed a user's text query with CLIP's text encoder, for image retrieval."""
    model, processor = _get_model_and_processor()
    inputs = processor(text=[query], return_tensors="pt", padding=True)

    with torch.no_grad():
        features = model.get_text_features(**inputs).pooler_output
    features = features / features.norm(p=2, dim=-1, keepdim=True)

    return features[0].tolist()
