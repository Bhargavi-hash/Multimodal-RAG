from pathlib import Path

import numpy as np

from embedding.clip_embedder import EMBEDDING_DIM as CLIP_DIM
from embedding.clip_embedder import embed_clip_text_query, embed_image_records
from embedding.text_embedder import EMBEDDING_DIM as TEXT_DIM
from embedding.text_embedder import embed_text_query, embed_text_records
from ingestion.image_loader import load_image

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def cosine_similarity(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class TestTextEmbedder:
    def test_embedding_has_expected_dimension(self):
        records = [{"id": "a", "modality": "text", "content": "hello world", "metadata": {}}]
        out = embed_text_records(records)

        assert len(out[0]["embedding"]) == TEXT_DIM

    def test_preserves_original_fields(self):
        records = [{"id": "a", "modality": "text", "content": "hello world", "metadata": {"page_number": 1}}]
        out = embed_text_records(records)

        assert out[0]["id"] == "a"
        assert out[0]["content"] == "hello world"
        assert out[0]["metadata"] == {"page_number": 1}

    def test_empty_records_returns_empty_list(self):
        assert embed_text_records([]) == []

    def test_similar_texts_more_similar_than_unrelated(self):
        records = [
            {"id": "a", "modality": "text", "content": "The cat sat on the mat.", "metadata": {}},
            {"id": "b", "modality": "text", "content": "A kitten was resting on the rug.", "metadata": {}},
            {"id": "c", "modality": "text", "content": "Quarterly revenue grew by 12 percent.", "metadata": {}},
        ]
        out = embed_text_records(records)
        vec_a, vec_b, vec_c = (r["embedding"] for r in out)

        sim_similar = cosine_similarity(vec_a, vec_b)
        sim_unrelated = cosine_similarity(vec_a, vec_c)
        assert sim_similar > sim_unrelated

    def test_query_embedding_matches_chunk_embedding_dimension(self):
        query_vector = embed_text_query("what is the revenue?")
        assert len(query_vector) == TEXT_DIM


class TestClipEmbedder:
    def test_image_embedding_has_expected_dimension(self):
        record = load_image(FIXTURES_DIR / "sample.png")
        out = embed_image_records([record])

        assert len(out[0]["embedding"]) == CLIP_DIM

    def test_preserves_original_fields(self):
        record = load_image(FIXTURES_DIR / "sample.png")
        out = embed_image_records([record])

        assert out[0]["id"] == record["id"]
        assert out[0]["metadata"] == record["metadata"]
        assert out[0]["content"] is record["content"]

    def test_empty_records_returns_empty_list(self):
        assert embed_image_records([]) == []

    def test_similar_images_more_similar_than_unrelated(self):
        records = [
            load_image(FIXTURES_DIR / "red_square_a.png"),
            load_image(FIXTURES_DIR / "red_square_b.png"),
            load_image(FIXTURES_DIR / "checkerboard.png"),
        ]
        out = embed_image_records(records)
        vec_a, vec_b, vec_c = (r["embedding"] for r in out)

        sim_similar = cosine_similarity(vec_a, vec_b)
        sim_unrelated = cosine_similarity(vec_a, vec_c)
        assert sim_similar > sim_unrelated

    def test_text_query_embedding_matches_image_embedding_dimension(self):
        query_vector = embed_clip_text_query("a red square")
        assert len(query_vector) == CLIP_DIM

    def test_cross_modal_text_query_prefers_matching_image(self):
        # CLIP's shared text/image space: a text query about color should sit
        # closer to the matching-colored image than to an unrelated one.
        records = [
            load_image(FIXTURES_DIR / "red_square_a.png"),
            load_image(FIXTURES_DIR / "checkerboard.png"),
        ]
        out = embed_image_records(records)
        red_vec, checker_vec = (r["embedding"] for r in out)

        query_vec = embed_clip_text_query("a solid red square")
        sim_red = cosine_similarity(query_vec, red_vec)
        sim_checker = cosine_similarity(query_vec, checker_vec)
        assert sim_red > sim_checker
