"""Image ingestion: each uploaded image is treated as one retrievable unit.

See architecture.md Section 3.1 for the data contract this module produces.
"""

import uuid
from pathlib import Path

from PIL import Image


def load_image(path: str | Path, upload_id: str | None = None) -> dict:
    """Load an image file as a single retrievable record.

    Returns a dict matching the ingestion data contract:
    {"id": str, "modality": "image", "content": PIL.Image, "metadata": {...}}
    """
    path = Path(path)
    image = Image.open(path)
    image.load()  # decode now so the file handle can be released

    upload_id = upload_id or str(uuid.uuid4())
    return {
        "id": upload_id,
        "modality": "image",
        "content": image,
        "metadata": {
            "source_file": path.name,
            "upload_id": upload_id,
        },
    }
