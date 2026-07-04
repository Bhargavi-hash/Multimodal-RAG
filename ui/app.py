"""Streamlit UI: upload, indexed-file list, chat with citations.

See architecture.md Section 3.6.
"""

import tempfile
from pathlib import Path

import streamlit as st

from embedding.clip_embedder import embed_image_records
from embedding.text_embedder import embed_text_records
from generation.answer import generate_answer
from ingestion.image_loader import load_image
from ingestion.pdf_loader import load_pdf
from retrieval.retriever import retrieve
from store import vector_store as vs

st.set_page_config(page_title="Multimodal RAG", layout="wide")


@st.cache_resource
def get_client():
    return vs.get_client()


client = get_client()

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def _save_temp(uploaded_file) -> Path:
    """Write an uploaded file to disk under its original name so ingestion
    metadata (source_file) reflects the name the user uploaded."""
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / uploaded_file.name
    tmp_path.write_bytes(uploaded_file.getvalue())
    return tmp_path


st.title("Multimodal RAG")

st.header("Upload documents")
uploaded_files = st.file_uploader(
    "Upload PDFs or images",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

for uploaded_file in uploaded_files or []:
    if uploaded_file.name in st.session_state.indexed_files:
        continue

    tmp_path = _save_temp(uploaded_file)
    suffix = tmp_path.suffix.lower()

    with st.spinner(f"Indexing {uploaded_file.name}..."):
        if suffix == ".pdf":
            records = load_pdf(tmp_path)
            embedded = embed_text_records(records)
            vs.add_text_records(client, embedded)
            st.session_state.indexed_files[uploaded_file.name] = {
                "modality": "text",
                "count": len(records),
            }
        else:
            record = load_image(tmp_path)
            embedded = embed_image_records([record])
            vs.add_image_records(client, embedded)
            st.session_state.indexed_files[uploaded_file.name] = {
                "modality": "image",
                "count": 1,
            }
    st.success(f"Indexed {uploaded_file.name}")

st.subheader("Indexed files")
if st.session_state.indexed_files:
    for fname, info in st.session_state.indexed_files.items():
        unit = "chunks" if info["modality"] == "text" else "image"
        st.write(f"- **{fname}** — {info['count']} {unit}")
else:
    st.write("No files indexed yet.")

if st.button("Clear index"):
    vs.clear(client)
    st.session_state.indexed_files = {}
    st.session_state.chat_history = []
    st.rerun()

st.header("Ask a question")

for past_question, past_answer, past_retrieval in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(past_question)
    with st.chat_message("assistant"):
        st.write(past_answer)
        for item in past_retrieval.image_items:
            st.image(item.content, caption=item.metadata["source_file"], width=150)

question = st.chat_input("Ask something about your uploaded documents...")

if question:
    with st.spinner("Retrieving and generating answer..."):
        retrieval = retrieve(client, question)
        answer = generate_answer(question, retrieval)
    st.session_state.chat_history.append((question, answer, retrieval))
    st.rerun()
