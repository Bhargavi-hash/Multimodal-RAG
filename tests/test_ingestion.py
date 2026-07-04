from pathlib import Path

from PIL import Image

from ingestion.image_loader import load_image
from ingestion.pdf_loader import chunk_text, load_pdf

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestChunkText:
    def test_word_count_per_chunk(self):
        text = " ".join(f"word{i}" for i in range(1200))
        chunks = chunk_text(text, chunk_words=500, overlap_words=50)

        assert len(chunks) == 3
        assert len(chunks[0].split()) == 500
        assert len(chunks[1].split()) == 500
        assert len(chunks[2].split()) == 300  # remainder

    def test_overlap_between_consecutive_chunks(self):
        text = " ".join(f"word{i}" for i in range(1200))
        chunks = chunk_text(text, chunk_words=500, overlap_words=50)

        first_words = chunks[0].split()
        second_words = chunks[1].split()
        # last 50 words of chunk 0 == first 50 words of chunk 1
        assert first_words[-50:] == second_words[:50]

        second_words = chunks[1].split()
        third_words = chunks[2].split()
        assert second_words[-50:] == third_words[:50]

    def test_empty_text_produces_no_chunks(self):
        assert chunk_text("") == []

    def test_short_text_produces_single_chunk(self):
        text = "just a few words here"
        chunks = chunk_text(text, chunk_words=500, overlap_words=50)
        assert chunks == [text]


class TestLoadPdf:
    def test_record_structure(self):
        records = load_pdf(FIXTURES_DIR / "sample.pdf")

        assert len(records) > 0
        for record in records:
            assert set(record.keys()) == {"id", "modality", "content", "metadata"}
            assert record["modality"] == "text"
            assert isinstance(record["content"], str)
            assert isinstance(record["id"], str)

    def test_metadata(self):
        records = load_pdf(FIXTURES_DIR / "sample.pdf")

        assert len(records) == 2  # one chunk per page in the fixture
        assert records[0]["metadata"] == {
            "source_file": "sample.pdf",
            "page_number": 1,
            "chunk_index": 0,
        }
        assert records[1]["metadata"] == {
            "source_file": "sample.pdf",
            "page_number": 2,
            "chunk_index": 0,
        }

    def test_content_extracted(self):
        records = load_pdf(FIXTURES_DIR / "sample.pdf")

        assert "page one" in records[0]["content"]
        assert "page two" in records[1]["content"]


class TestLoadImage:
    def test_record_structure(self):
        record = load_image(FIXTURES_DIR / "sample.png")

        assert set(record.keys()) == {"id", "modality", "content", "metadata"}
        assert record["modality"] == "image"
        assert isinstance(record["content"], Image.Image)
        assert isinstance(record["id"], str)

    def test_metadata(self):
        record = load_image(FIXTURES_DIR / "sample.png")

        assert record["metadata"]["source_file"] == "sample.png"
        assert record["metadata"]["upload_id"] == record["id"]

    def test_upload_id_override(self):
        record = load_image(FIXTURES_DIR / "sample.png", upload_id="custom-id")

        assert record["id"] == "custom-id"
        assert record["metadata"]["upload_id"] == "custom-id"

    def test_generates_unique_ids_by_default(self):
        record_a = load_image(FIXTURES_DIR / "sample.png")
        record_b = load_image(FIXTURES_DIR / "sample.png")

        assert record_a["id"] != record_b["id"]
