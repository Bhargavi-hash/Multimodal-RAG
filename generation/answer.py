"""Answer generation: build a grounded prompt, call Gemini, return a cited answer.

See architecture.md Section 3.5.
"""

from functools import lru_cache

from google import genai
from google.genai import types

from retrieval.retriever import RetrievalResult

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_MAX_OUTPUT_TOKENS = 1024

NOT_FOUND_MESSAGE = "Not found in the provided documents."

SYSTEM_PROMPT = """You are a question-answering assistant that only uses the context provided below \
— excerpts from documents and images the user has uploaded. Never answer from your own general \
knowledge, even if you know the answer.

Rules:
- Answer only using the provided text excerpts and images.
- If the context does not address the question, respond with exactly: "Not found in the provided documents."
- Every claim in your answer must be followed by an inline citation matching its source, using the \
exact format shown next to that piece of context: "[source_file.ext]" for an image, or \
"[source_file.ext, page N]" for a text excerpt.
- Do not cite a source that doesn't support the specific claim next to it."""


@lru_cache(maxsize=1)
def _get_client() -> genai.Client:
    return genai.Client()


def _build_content(question: str, retrieval: RetrievalResult) -> list:
    text_blocks = []
    for item in retrieval.text_items:
        label = f"[{item.metadata['source_file']}, page {item.metadata['page_number']}]"
        text_blocks.append(f"Excerpt {label}:\n{item.content}")

    context_text = "\n\n".join(text_blocks) if text_blocks else "(no text excerpts retrieved)"
    content = [f"Question: {question}\n\nText context:\n{context_text}"]

    for item in retrieval.image_items:
        label = f"[{item.metadata['source_file']}]"
        content.append(f"Image {label}:")
        content.append(item.content)  # PIL.Image passed natively for vision input

    return content


def generate_answer(
    question: str,
    retrieval: RetrievalResult,
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> str:
    """Generate a cited answer grounded only in the retrieved context.

    If retrieval found nothing relevant in either modality, short-circuits to
    the fixed "not found" message without calling the API — this guarantees
    the "I don't know" path fires reliably rather than depending on the model
    to comply with the system prompt (architecture.md Sections 3.4/3.5).
    """
    if retrieval.is_empty:
        return NOT_FOUND_MESSAGE

    client = _get_client()
    content = _build_content(question, retrieval)

    response = client.models.generate_content(
        model=model,
        contents=content,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=max_output_tokens,
        ),
    )

    return response.text or ""
