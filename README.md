# AI Document Assistant

A beginner-friendly Streamlit RAG-style document assistant.

## What it does

- Uploads local **PDF, DOCX, TXT, and MD** files.
- Extracts text and shows document information.
- Keeps PDF page numbers when available.
- Splits text into overlapping chunks.
- Creates Sentence Transformers embeddings once per source set.
- Stores embeddings and metadata in Streamlit session state.
- Uses FAISS for semantic vector search.
- Adds simple keyword matching.
- Combines semantic + keyword scores into a hybrid search.
- Sends the retrieved chunks to Grok.
- Instructs Grok to answer only from retrieved context.
- Shows the retrieved source chunks after every answer.
- Accepts a public/shared Google Drive file or folder link.
- Processes Google Drive documents through the same extraction → chunking → embedding → search pipeline.
- Reuses the processed vector index when the source documents have not changed.

## Project structure

```text
ai-document-assistant/
├── app.py
├── requirements.txt
└── README.md
```

## 1. Install

Create a virtual environment if you want, then run:

```bash
pip install -r requirements.txt
```

## 2. Add the Grok API key

Create this file:

```text
.streamlit/secrets.toml
```

Put your key inside:

```toml
XAI_API_KEY = "your-xai-api-key"
```

**Never put the real key inside `app.py` or commit `secrets.toml` to GitHub.**

The app uses xAI's OpenAI-compatible API endpoint and the `grok-4.6` model.

## 3. Run

```bash
streamlit run app.py
```

## Google Drive

Paste a Google Drive file or folder URL in the sidebar.

The Drive file/folder must be accessible to the app. The app uses `gdown` to download supported files and then sends them through the same pipeline as local uploads.

Supported Drive file types:

- PDF
- DOCX
- TXT
- MD

Legacy `.doc` files are not included because reliable `.doc` extraction normally requires an external office/conversion dependency. Save them as `.docx` first.

## How the pipeline works

```text
Local Upload / Google Drive
          ↓
      Extraction
          ↓
        Chunks
          ↓
   Sentence Embeddings
          ↓
        FAISS
          ↓
 ┌──────────────────────┐
 │ Semantic Search      │
 │ + Keyword Search     │
 └──────────────────────┘
          ↓
     Hybrid Ranking
          ↓
   Top Relevant Chunks
          ↓
        Grok
          ↓
   Answer + Sources
```

## Why embeddings are not recreated for every question

The app stores:

- extracted document records
- chunks
- embeddings
- FAISS index
- source signature

inside `st.session_state`.

When the user asks another question, the existing embeddings and FAISS index are reused.

If the uploaded files or Google Drive URL changes, the source signature changes and the documents are processed again.

## Important limitation

`st.session_state` lasts for the current Streamlit session. It is not a permanent database.

For a production application, the next step would be persistent storage such as a vector database or a saved FAISS index plus metadata database.

## Learning value

This small project demonstrates the core pieces of a simple RAG application:

1. Document ingestion
2. Text extraction
3. Chunking
4. Embeddings
5. Vector search
6. Keyword retrieval
7. Hybrid retrieval
8. Metadata preservation
9. Context grounding
10. Source display
11. LLM generation
12. Session-state caching
