# Multimodal RAG

A retrieval-augmented Q&A system: upload PDFs and images, ask natural-language
questions, and get answers grounded in the actual content you uploaded — with
citations back to the source page or image. If the answer isn't in your documents,
the system says so instead of falling back on the LLM's own general knowledge.

See [`architecture.md`](./architecture.md) for the full system design and key
decisions.

---

## What this project does

```
Upload (PDFs, images)
    -> Ingestion (chunk PDF text ~500 words w/ overlap; images as single units)
    -> Embedding (text -> sentence-transformers, images -> CLIP)
    -> Vector store (Chroma, two collections: text_chunks, images)
    -> Retrieval (query both collections, per-modality relevance thresholds)
    -> Generation (Gemini 2.5 Flash, answers only from retrieved context + citations)
    -> Streamlit UI (upload, indexed-file list, chat with citations/thumbnails)
```

- **Ingestion** — extracts PDF text per page and splits into overlapping chunks;
  treats each uploaded image as one retrievable unit.
- **Embedding** — text chunks via `sentence-transformers` (local, 384-dim); images
  via CLIP (local, 512-dim), which also embeds text queries into the same space so
  a text question can retrieve relevant images directly.
- **Vector store** — Chroma, two cosine-similarity collections (text and image),
  queried independently since they use different embedding spaces.
- **Retrieval** — applies separate relevance thresholds per modality (CLIP's
  cross-modal similarity band is narrower than sentence-transformers', so a shared
  threshold under-filters image results).
- **Generation** — Gemini 2.5 Flash (free tier, supports vision), explicitly
  instructed to answer only from retrieved context and say "not found" otherwise.
- **UI** — Streamlit: upload files, see what's indexed, ask questions, see answers
  with citations and image thumbnails.

---

## Prerequisites

- Python 3.10+
- A virtual environment (recommended)
- A free Gemini API key: [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
  (no credit card required)

---

## Setup

```bash
git clone <your-repo-url>
cd multimodal-rag

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
# torch + torchvision are needed locally for CLIP; install as a matched pair:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Environment variables

```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

Add this to `~/.bashrc` or a local `.env` file (gitignored) so you don't have to
re-export it every session.

---

## Running locally

```bash
python3 -m streamlit run ui/app.py
```

Opens in your browser (usually `localhost:8501`). From there:

1. Upload one or more PDFs and/or images using the file uploader.
2. Check the indexed-file list to confirm they were processed.
3. Ask a question in the chat box.
4. Review the answer, its citations (file + page, or file name for images), and
   any retrieved image thumbnails.

**Sanity check worth running once:** ask a question your uploaded documents
genuinely don't cover — the system should respond that it isn't found in the
provided documents, rather than answering from general knowledge. This is the
core behavior that makes the system trustworthy.

---

## Deploying (Streamlit Community Cloud)

1. Push this repo to GitHub. Confirm `.gitignore` excludes `.env`, `venv/`,
   `.venv/`, and the local Chroma DB directory (`data/chroma_db/`) — never commit
   API keys or the local vector store.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Create a new app, pointing at this repo and `ui/app.py` as the entry point.
4. In the app's **Settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your_gemini_api_key_here"
   ```
   This is Streamlit Cloud's own secrets store — keys entered here never touch
   your repo or git history.
5. Deploy. Streamlit Cloud auto-redeploys on future pushes to the connected branch.

**Note on resource usage:** CLIP + torch is a fairly heavy dependency footprint for
a free-tier hosting environment. If the app is slow to boot or crashes on deploy,
this is the likely cause — worth knowing rather than being surprised.

---

## Project structure

```
multimodal-rag/
├── architecture.md
├── README.md
├── requirements.txt
├── ingestion/
│   ├── pdf_loader.py         # PDF text extraction + chunking
│   └── image_loader.py       # image loading as single retrievable units
├── embedding/
│   ├── text_embedder.py      # sentence-transformers wrapper
│   └── clip_embedder.py      # CLIP wrapper (image + cross-modal text encoders)
├── store/
│   └── vector_store.py       # Chroma client, collection setup, insert/query
├── retrieval/
│   └── retriever.py          # cross-collection query + relevance thresholds
├── generation/
│   └── answer.py              # prompt assembly, Gemini call, citation formatting
├── ui/
│   └── app.py                 # Streamlit app (includes sys.path fix for deployment)
├── tests/
│   ├── test_ingestion.py
│   ├── test_embedding.py
│   └── test_retrieval.py
└── data/
    └── chroma_db/              # local persistent Chroma store (gitignored)
```

---

## Known limitations

- No video or audio support — text (PDF) and static images only.
- No OCR — assumes PDFs have an extractable text layer; scanned image-only PDFs
  won't yield usable text chunks.
- Figures/charts embedded inside PDF pages aren't separately cropped and indexed
  as images — only whole page text is chunked.
- No authentication or multi-user isolation — single shared index per running
  instance.
- Free-tier Gemini usage means your prompts/responses may be used by Google to
  improve their models — worth keeping in mind if you upload anything sensitive.

See `architecture.md` Section 6 for the full list of non-goals and constraints.

---

## License / disclaimer

Personal learning project. Answers are only as good as the retrieval step and the
uploaded documents — verify anything important against the cited source.


