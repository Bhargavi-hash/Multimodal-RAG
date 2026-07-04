from pathlib import Path

import pytest
from PIL import Image

from embedding.clip_embedder import embed_image_records
from embedding.text_embedder import embed_text_records
from ingestion.image_loader import load_image
from retrieval.retriever import RetrievedItem, _apply_threshold, retrieve
from store import vector_store as vs

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def client(tmp_path):
    return vs.get_client(tmp_path / "chroma_db")


@pytest.fixture
def indexed_client(client):
    text_records = embed_text_records(
        [
            {
                "id": "paris1",
                "modality": "text",
                "content": "The Eiffel Tower is located in Paris, France, and was completed in 1889.",
                "metadata": {"source_file": "paris.pdf", "page_number": 1, "chunk_index": 0},
            },
            {
                "id": "python1",
                "modality": "text",
                "content": "Python is a popular programming language widely used for data science.",
                "metadata": {"source_file": "python.pdf", "page_number": 1, "chunk_index": 0},
            },
        ]
    )
    vs.add_text_records(client, text_records)

    image_records = embed_image_records([load_image(FIXTURES_DIR / "red_square_a.png", upload_id="red1")])
    vs.add_image_records(client, image_records)

    return client


class TestApplyThreshold:
    """Unit tests for the threshold cutoff logic with controlled scores,
    independent of real model score noise."""

    def test_keeps_items_when_best_score_above_threshold(self):
        items = [RetrievedItem(id="a", modality="text", score=0.5, content="x", metadata={})]
        assert _apply_threshold(items, threshold=0.2) == items

    def test_drops_all_items_when_best_score_below_threshold(self):
        items = [RetrievedItem(id="a", modality="text", score=0.1, content="x", metadata={})]
        assert _apply_threshold(items, threshold=0.2) == []

    def test_empty_input_returns_empty(self):
        assert _apply_threshold([], threshold=0.2) == []

    def test_boundary_score_equal_to_threshold_is_kept(self):
        items = [RetrievedItem(id="a", modality="text", score=0.2, content="x", metadata={})]
        assert _apply_threshold(items, threshold=0.2) == items


class TestRetrieveText:
    def test_relevant_query_returns_matching_chunk_first(self, indexed_client):
        result = retrieve(indexed_client, "Where is the Eiffel Tower located?")

        assert len(result.text_items) > 0
        best = result.text_items[0]
        assert best.metadata["source_file"] == "paris.pdf"
        assert best.modality == "text"
        assert isinstance(best.content, str)

    def test_unrelated_query_returns_no_text_results(self, indexed_client):
        result = retrieve(indexed_client, "What is the boiling point of mercury on Jupiter?")

        assert result.text_items == []

    def test_metadata_passes_through_unchanged(self, indexed_client):
        result = retrieve(indexed_client, "Tell me about the Eiffel Tower")

        best = result.text_items[0]
        assert best.metadata == {"source_file": "paris.pdf", "page_number": 1, "chunk_index": 0}

    def test_text_top_k_limits_results(self, client):
        records = embed_text_records(
            [
                {"id": f"t{i}", "modality": "text", "content": f"This is document number {i} about topic {i}.", "metadata": {"source_file": f"doc{i}.pdf", "page_number": 1, "chunk_index": 0}}
                for i in range(5)
            ]
        )
        vs.add_text_records(client, records)

        result = retrieve(client, "document number 2", text_top_k=2)
        assert len(result.text_items) <= 2


class TestRetrieveImage:
    def test_relevant_query_returns_image(self, indexed_client):
        result = retrieve(indexed_client, "a solid red square")

        assert len(result.image_items) == 1
        best = result.image_items[0]
        assert best.modality == "image"
        assert best.metadata["source_file"] == "red_square_a.png"
        assert isinstance(best.content, Image.Image)

    def test_unrelated_query_returns_no_image_results(self, indexed_client):
        result = retrieve(indexed_client, "Where is the Eiffel Tower located?")

        assert result.image_items == []


class TestRetrieveEmptyStore:
    def test_empty_store_returns_no_results_without_crashing(self, client):
        result = retrieve(client, "anything at all")

        assert result.text_items == []
        assert result.image_items == []
        assert result.is_empty
