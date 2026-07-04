"""PDF ingestion: extract text per page and split into overlapping chunks.

See architecture.md Section 3.1 for the data contract this module produces.
"""

from pathlib import Path

from pypdf import PdfReader

CHUNK_WORDS = 500
OVERLAP_WORDS = 50


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap_words: int = OVERLAP_WORDS) -> list[str]:
    """Split text into overlapping chunks of roughly `chunk_words` words."""
    words = text.split()
    if not words:
        return []

    chunks = []
    step = chunk_words - overlap_words
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_words]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        if start + chunk_words >= len(words):
            break
    return chunks


def load_pdf(path: str | Path) -> list[dict]:
    """Extract text from each page of a PDF and split into overlapping chunks.

    Returns a list of dicts matching the ingestion data contract:
    {"id": str, "modality": "text", "content": str, "metadata": {...}}
    """
    path = Path(path)
    reader = PdfReader(str(path))

    records = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        chunks = chunk_text(page_text)
        for chunk_index, chunk in enumerate(chunks):
            records.append(
                {
                    "id": f"{path.name}_p{page_number}_c{chunk_index}",
                    "modality": "text",
                    "content": chunk,
                    "metadata": {
                        "source_file": path.name,
                        "page_number": page_number,
                        "chunk_index": chunk_index,
                    },
                }
            )
    return records
