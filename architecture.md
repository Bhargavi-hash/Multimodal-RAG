# Multimodal RAG — Architecture

## 1. Goal

A retrieval-augmented Q&A system where a user uploads PDFs and/or images, and asks
natural-language questions that get answered grounded in the content of those
uploads — including citations back to the source (file + page, or file + image name).

**This project explicitly does NOT:**
- Train or fine-tune any model — embedding models and the LLM are used as-is
- Support video or audio (text + image only, to keep scope small)
- Persist data across restarts unless explicitly noted (see Section 6) — this is a
  demo/learning tool, not a production document store
- Guarantee correctness of retrieved answers — the system should say "not found in
  the provided documents" rather than fall back on the LLM's own memorized knowledge

**Primary learning goals:**
- Understand chunking + embedding + vector search as the core RAG mechanism
- Handle two different modalities (text, image) in one retrieval pipeline
- Ship something interactively usable (Streamlit) and deployable, not just a script

---

## 2. High-level data flow

```
                    ┌───────────────────────┐
   User uploads  -->│  Ingestion             │
   PDFs / images    │  - PDF: chunk text     │
                     │  - Images: whole-image │
                     └───────────┬───────────┘
                                 v
                    ┌───────────────────────┐
                    │  Embedding             │
                    │  - text chunks -> text │
                    │    embedding model     │
                    │  - images -> CLIP      │
                    └───────────┬───────────┘
                                 v
                    ┌───────────────────────┐
                    │  Vector Store (Chroma) │
                    │  one collection per    │
                    │  modality              │
                    └───────────┬───────────┘
                                 v
   User asks    -->  ┌───────────────────────┐
   a question        │  Retrieval             │
                     │  embed query with BOTH │
                     │  text + CLIP text      │
                     │  encoders, search both │
                     │  collections           │
                     └───────────┬───────────┘
                                 v
                    ┌───────────────────────┐
                    │  Generation (LLM)      │
                    │  question + retrieved  │
                    │  chunks/images -> answer│
                    │  + citations            │
                    └───────────┬───────────┘
                                 v
                    ┌───────────────────────┐
                    │  Streamlit UI          │
                    │  upload + chat + cite  │
                    └───────────────────────┘
```

Design principle: **retrieval and generation are separate, swappable stages.** The
vector store never talks to the LLM directly; the UI never talks to the vector store
directly. Each stage has one job.

---

## 3. Components

### 3.1 Ingestion (`ingestion/`)

- **PDFs** (`ingestion/pdf_loader.py`): extract text per page (`pypdf` or
  `pdfplumber`), then split into overlapping chunks — target ~500 words per chunk,
  ~50-word overlap between consecutive chunks, so a fact split across a chunk
  boundary isn't lost entirely. Each chunk keeps metadata: `source_file`, `page_number`,
  `chunk_index`.
- **Images** (`ingestion/image_loader.py`): no chunking needed — treat each uploaded
  image as one retrievable unit. Store metadata: `source_file`, `upload_id`.
  (If a user uploads a large scanned PDF page as an image with multiple figures,
  that's a known limitation — see Section 6.)

**Data contract**: ingestion produces a list of dicts:
```python
{"id": str, "modality": "text" | "image", "content": str | PIL.Image, "metadata": {...}}
```

### 3.2 Embedding (`embedding/`)

- **Text chunks** → a text embedding model. Default: `sentence-transformers`
  (`all-MiniLM-L6-v2`), local, free, no API key required. (Swappable later for an
  API-based embedding model if quality needs improve.)
- **Images** → CLIP (`openai/clip-vit-base-patch32` via `transformers` or
  `open_clip`), which maps images into a shared image/text vector space.
- **Query embedding**: at query time, the user's text question gets embedded
  **twice** — once with the text embedding model (to search the text collection)
  and once with CLIP's text encoder (to search the image collection). This is what
  makes cross-modal retrieval work: CLIP's text encoder and image encoder share one
  coordinate space, so a text query can retrieve relevant images directly.

**Data contract**: embedding produces a fixed-length float vector per chunk/image,
alongside its original metadata, ready to insert into the vector store.

### 3.3 Vector Store (`store/`)

- **Chroma**, running locally (no hosted DB needed for this scope).
- **Two collections**: `text_chunks` and `images` — kept separate since they use
  different embedding models with different vector dimensions and semantics.
  Retrieval queries both independently and merges results by relevance score.
- Each stored vector carries its full metadata (source file, page/upload id) so
  retrieved results can be cited back to their origin.

### 3.4 Retrieval (`retrieval/retriever.py`)

- Given a user question: embed it with both the text model and CLIP's text encoder.
- Query `text_chunks` for top-k (default k=4) nearest text chunks.
- Query `images` for top-k (default k=2) nearest images.
- Return both sets together, each tagged with its similarity score and metadata.
- **Relevance threshold**: if the best result's similarity score falls below a
  configurable cutoff, treat that modality as "nothing relevant found" rather than
  forcing in a weak match — this directly supports the "say I don't know" goal.

### 3.5 Generation (`generation/answer.py`)

- Build a prompt containing: the user's question, the retrieved text chunks (with
  source labels), and the retrieved images (passed as image content, since Claude
  accepts image input directly).
- Explicit instruction in the system prompt: **answer only using the provided
  context; if the context doesn't address the question, say so directly rather than
  answering from general knowledge.**
- Response includes inline citations (e.g. "[document.pdf, page 3]" or
  "[photo_2.jpg]") so the user can verify claims against the source.

### 3.6 UI (`ui/app.py`)

- **Streamlit**, single-page app:
  - File uploader (accepts `.pdf`, `.png`, `.jpg`, `.jpeg`) — triggers ingestion +
    embedding + storage on upload.
  - A simple list showing what's currently indexed (filenames, chunk/image counts).
  - A chat-style input box for questions, showing the answer plus its citations and
    (for image citations) a thumbnail of the retrieved image.
  - A "clear index" button to reset the vector store for a fresh session.

---

## 4. Repo structure

```
multimodal-rag/
├── architecture.md
├── requirements.txt
├── ingestion/
│   ├── pdf_loader.py
│   └── image_loader.py
├── embedding/
│   ├── text_embedder.py     # sentence-transformers wrapper
│   └── clip_embedder.py     # CLIP wrapper (image + text encoders)
├── store/
│   └── vector_store.py      # Chroma client, collection setup, insert/query helpers
├── retrieval/
│   └── retriever.py         # cross-collection query + relevance thresholding
├── generation/
│   └── answer.py            # prompt assembly + LLM call + citation formatting
├── ui/
│   └── app.py                # Streamlit app
├── tests/
│   ├── test_ingestion.py
│   ├── test_embedding.py
│   └── test_retrieval.py
└── data/
    └── chroma_db/             # local persistent Chroma store (gitignored)
```

---

## 5. Key decisions and why

- **Chroma over a hosted vector DB** — zero setup, runs embedded in-process, ideal
  for a small/local/demo project. Swap later only if there's an actual scaling need.
- **`sentence-transformers` (local) for text embeddings** — free, no API key, fast
  enough for a small document set; avoids adding an API dependency just to embed text.
- **CLIP for images** — the standard choice for joint text/image retrieval; no
  captioning step needed since CLIP's encoders already share a coordinate space
  (see architecture discussion — CLIP doesn't require user-provided captions).
- **Two separate collections, not one merged space** — text and image embeddings
  come from different models with different vector spaces; merging them into one
  collection would produce meaningless similarity scores. Querying both separately
  and combining results at the retrieval layer is the correct approach.
- **Explicit "say I don't know" instruction + relevance threshold** — the single
  most important RAG design choice; prevents the system from silently falling back
  to the LLM's own memorized knowledge when retrieval comes up empty or weak.
- **Streamlit over a custom frontend** — fastest path to a usable, hostable UI for
  this scope; can be deployed via Streamlit Community Cloud for free with minimal
  configuration.

---

## 6. Non-goals / constraints

- No video or audio support in this version
- No OCR for scanned image-based PDFs (assumes PDFs have extractable text layers)
- No handling of figures/charts embedded inside PDF pages as separate retrievable
  images — a full page's text is chunked as text; embedded figures are not
  separately cropped and indexed (documented limitation, could be a future extension)
- No authentication/multi-user isolation — single shared index per running instance
- No persistence guarantees beyond the local Chroma directory — not backed up,
  not multi-instance safe

---

## 7. Open questions

- [ ] Which LLM for generation — Claude API, or something else? (assumed Claude API
      here, given multimodal input support)
- [ ] Should image citations show a thumbnail inline in the Streamlit UI, or just
      the filename?
- [ ] Deployment target — Streamlit Community Cloud (free, simplest) vs. self-hosted?
- [ ] Chunk size/overlap — 500/50 is a reasonable default; worth tuning after seeing
      real retrieval quality on your actual documents

---

## 8. How to use this doc with Claude Code

- Point Claude Code at this file first: *"Read architecture.md before making any
  changes."*
- Build layer by layer: ingestion → embedding → vector store → retrieval →
  generation → UI. Verify each stage independently before moving to the next —
  e.g., confirm chunks look right before embedding them, confirm embeddings retrieve
  sensible neighbors before wiring in the LLM.
- The riskiest layer to get subtly wrong is retrieval relevance — after building it,
  manually test with a question you know the answer to and a question you know
  ISN'T covered by your uploaded documents, to confirm the "I don't know" path
  actually triggers correctly.