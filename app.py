import io
import os
import re
import hashlib
import tempfile
from pathlib import Path

import faiss
import gdown
import numpy as np
import streamlit as st
from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# -----------------------------
# App settings
# -----------------------------
st.set_page_config(page_title="AI Document Assistant", page_icon="📚", layout="wide")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 5

STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
    "to", "of", "in", "on", "for", "with", "from", "by", "as", "at",
    "it", "this", "that", "these", "those", "be", "can", "what", "why",
    "how", "when", "where", "who", "which", "do", "does", "did", "about"
}


# -----------------------------
# Model
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Extraction
# -----------------------------
def extract_pdf(file_bytes, file_name):
    """Return one record per PDF page."""
    reader = PdfReader(io.BytesIO(file_bytes))
    records = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            records.append({
                "text": text.strip(),
                "file_name": file_name,
                "page": page_number,
            })

    return records


def extract_docx(file_bytes, file_name):
    """Extract DOCX paragraphs. DOCX page numbers are not reliable here."""
    document = Document(io.BytesIO(file_bytes))
    text = "\n".join(p.text for p in document.paragraphs if p.text.strip())

    if not text.strip():
        return []

    return [{
        "text": text.strip(),
        "file_name": file_name,
        "page": None,
    }]


def extract_text_file(file_bytes, file_name):
    """Extract TXT or MD text."""
    text = file_bytes.decode("utf-8", errors="replace")

    if not text.strip():
        return []

    return [{
        "text": text.strip(),
        "file_name": file_name,
        "page": None,
    }]


def extract_document(file_bytes, file_name):
    """Choose the correct extractor from the file extension."""
    extension = Path(file_name).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_bytes, file_name)
    if extension == ".docx":
        return extract_docx(file_bytes, file_name)
    if extension in {".txt", ".md"}:
        return extract_text_file(file_bytes, file_name)

    raise ValueError(f"Unsupported file type: {extension}")


# -----------------------------
# Chunking
# -----------------------------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping character chunks."""
    if not text.strip():
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = max(0, end - overlap)

    return chunks


def build_chunks(document_records):
    """Create chunks while preserving file and page metadata."""
    chunks = []

    for record in document_records:
        for chunk_number, text in enumerate(chunk_text(record["text"]), start=1):
            chunks.append({
                "text": text,
                "file_name": record["file_name"],
                "page": record["page"],
                "chunk_number": chunk_number,
            })

    return chunks


# -----------------------------
# Embeddings + FAISS
# -----------------------------
def build_vector_store(chunks):
    model = load_embedding_model()

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings


# -----------------------------
# Keyword search
# -----------------------------
def important_words(text):
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [word for word in words if word not in STOP_WORDS and len(word) > 2]


def keyword_score(question, chunk_text):
    """Simple keyword overlap score between 0 and 1."""
    question_words = set(important_words(question))

    if not question_words:
        return 0.0

    chunk_words = set(important_words(chunk_text))
    matches = question_words.intersection(chunk_words)

    return len(matches) / len(question_words)


# -----------------------------
# Hybrid search
# -----------------------------
def hybrid_search(question, index, chunks, model, top_k=TOP_K):
    """Combine semantic similarity and keyword overlap."""
    question_embedding = model.encode(
        [question],
        normalize_embeddings=True,
    ).astype("float32")

    # Get more candidates than we finally return.
    candidate_k = min(max(top_k * 3, 10), len(chunks))
    semantic_scores, indices = index.search(question_embedding, candidate_k)

    candidates = []

    for semantic_score, index_number in zip(
        semantic_scores[0], indices[0]
    ):
        if index_number < 0:
            continue

        chunk = chunks[index_number]
        semantic = float((semantic_score + 1) / 2)
        keyword = keyword_score(question, chunk["text"])

        # 70% semantic + 30% keyword
        hybrid = (0.70 * semantic) + (0.30 * keyword)

        candidates.append({
            **chunk,
            "semantic_score": semantic,
            "keyword_score": keyword,
            "hybrid_score": hybrid,
        })

    candidates.sort(key=lambda item: item["hybrid_score"], reverse=True)

    return candidates[:top_k]


# -----------------------------
# Google Drive
# -----------------------------
def download_drive_source(url):
    """
    Download a public/shared Google Drive file or folder.
    The link must be accessible without an interactive login.
    """
    temp_dir = tempfile.mkdtemp(prefix="document_assistant_")

    if "folders/" in url:
        gdown.download_folder(
            url=url,
            output=temp_dir,
            quiet=True,
            use_cookies=False,
        )

    else:
        output_file = os.path.join(temp_dir, "drive_file")

        downloaded = gdown.download(
            url=url,
            output=output_file,
            quiet=True,
        )

        if not downloaded:
            raise ValueError(
                "Google Drive file could not be downloaded. "
                "Check that the file is shared as "
                "'Anyone with the link → Viewer'."
            )

        file_path = Path(downloaded)

        # Read downloaded file
        file_bytes = file_path.read_bytes()

        # Detect PDF
        if file_bytes.startswith(b"%PDF"):
            new_path = file_path.with_suffix(".pdf")

        # Detect DOCX
        elif file_bytes.startswith(b"PK"):
            new_path = file_path.with_suffix(".docx")

        # Detect TXT / MD
        else:
            try:
                file_bytes.decode("utf-8")
                new_path = file_path.with_suffix(".txt")
            except UnicodeDecodeError:
                raise ValueError(
                    "Unsupported Google Drive file type. "
                    "Please use PDF, DOCX, TXT, or MD."
                )

        file_path.rename(new_path)

    return temp_dir


def read_drive_files(folder):
    """Read supported files from a downloaded Drive file/folder."""
    loaded = []

    for path in Path(folder).rglob("*"):
        if not path.is_file():
            continue

        extension = path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            continue

        loaded.append((path.read_bytes(), path.name))

    return loaded


# -----------------------------
# Source ingestion + caching
# -----------------------------
def make_source_signature(uploaded_files, drive_url):
    hasher = hashlib.sha256()

    for uploaded_file in uploaded_files:
        hasher.update(uploaded_file.name.encode("utf-8"))
        hasher.update(uploaded_file.getvalue())

    hasher.update((drive_url or "").strip().encode("utf-8"))

    return hasher.hexdigest()


def process_sources(uploaded_files, drive_url):
    """Run extraction -> chunking -> embedding -> FAISS only when sources change."""
    source_signature = make_source_signature(uploaded_files, drive_url)

    if (
        st.session_state.get("source_signature") == source_signature
        and st.session_state.get("chunks")
        and st.session_state.get("index") is not None
    ):
        return False

    document_records = []

    # Local uploads
    for uploaded_file in uploaded_files:
        try:
            records = extract_document(
                uploaded_file.getvalue(),
                uploaded_file.name,
            )
            document_records.extend(records)
        except Exception as error:
            st.error(f"Could not read {uploaded_file.name}: {error}")

    # Google Drive
    if drive_url.strip():
        try:
            temp_dir = download_drive_source(drive_url.strip())
            drive_files = read_drive_files(temp_dir)

            for file_bytes, file_name in drive_files:
                try:
                    records = extract_document(file_bytes, file_name)
                    document_records.extend(records)
                except Exception as error:
                    st.warning(f"Could not read Drive file {file_name}: {error}")

        except Exception as error:
            st.error(f"Google Drive error: {error}")

    if not document_records:
        st.session_state.clear()
        return False

    chunks = build_chunks(document_records)

    if not chunks:
        st.session_state.clear()
        return False

    index, embeddings = build_vector_store(chunks)

    # Save everything in session state so questions do not rebuild embeddings.
    st.session_state["source_signature"] = source_signature
    st.session_state["document_records"] = document_records
    st.session_state["chunks"] = chunks
    st.session_state["embeddings"] = embeddings
    st.session_state["index"] = index

    return True


# -----------------------------
# Grok
# -----------------------------
def ask_grok(question, retrieved_chunks):
    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to Streamlit secrets."
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    context_parts = []

    for number, chunk in enumerate(retrieved_chunks, start=1):
        page = (
            f"page {chunk['page']}"
            if chunk["page"] is not None
            else "page not available"
        )

        context_parts.append(
            f"[SOURCE {number}]\n"
            f"File: {chunk['file_name']}\n"
            f"{page}\n"
            f"Text:\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are a document question-answering assistant.

Answer the user's question ONLY using the provided document context.
Do not use outside knowledge.

If the answer is not available in the context, say:
"I couldn't find that information in the provided documents."

Keep the answer clear and concise.
Do not invent facts, sources, page numbers, or quotations.
"""

    user_prompt = f"""
DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content


# -----------------------------
# UI
# -----------------------------
st.title("📚 AI Document Assistant")
st.caption(
    "Upload documents or connect a public/shared Google Drive file or folder. "
    "Ask questions using hybrid semantic + keyword retrieval."
)

with st.sidebar:
    st.header("Document Sources")

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT or MD files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    drive_url = st.text_input(
        "Google Drive file or folder link",
        placeholder="Paste a shared Google Drive link",
    )

    st.caption(
        "Google Drive links must be accessible to the app. "
        "Legacy .doc files are not supported; use .docx."
    )

    if st.button("Clear processed documents"):
        st.session_state.clear()
        st.rerun()


files_ready = uploaded_files or drive_url.strip()

if files_ready:
    with st.spinner("Processing documents..."):
        processed_now = process_sources(
            uploaded_files or [],
            drive_url,
        )

    if st.session_state.get("chunks"):
        if processed_now:
            st.success("Documents processed and embeddings created.")
        else:
            st.info("Documents already processed. Reusing saved embeddings.")

        documents = st.session_state["document_records"]
        chunks = st.session_state["chunks"]

        st.subheader("Document Information")

        col1, col2, col3 = st.columns(3)
        col1.metric("Document records", len(documents))
        col2.metric("Created chunks", len(chunks))
        col3.metric("Embedding size", st.session_state["embeddings"].shape[1])

        with st.expander("Show extracted document information"):
            for record in documents:
                page = (
                    f"Page {record['page']}"
                    if record["page"] is not None
                    else "Page not available"
                )

                st.markdown(
                    f"**{record['file_name']}** — {page}"
                )
                st.write(record["text"][:1000])
                st.divider()

        st.subheader("Ask a question")

        question = st.text_input(
            "Question",
            placeholder="What does the document say about...?",
        )

        if question.strip():
            model = load_embedding_model()

            with st.spinner("Searching documents..."):
                retrieved = hybrid_search(
                    question,
                    st.session_state["index"],
                    st.session_state["chunks"],
                    model,
                )

            with st.spinner("Asking Grok..."):
                try:
                    answer = ask_grok(question, retrieved)
                    st.markdown("### Answer")
                    st.write(answer)
                except Exception as error:
                    st.error(f"Grok error: {error}")
                    answer = None

            if answer:
                st.markdown("### Retrieved Sources")

                for number, source in enumerate(retrieved, start=1):
                    page = (
                        str(source["page"])
                        if source["page"] is not None
                        else "N/A"
                    )

                    with st.expander(
                        f"{number}. {source['file_name']} | Page: {page}"
                    ):
                        st.caption(
                            f"Hybrid: {source['hybrid_score']:.3f} | "
                            f"Semantic: {source['semantic_score']:.3f} | "
                            f"Keyword: {source['keyword_score']:.3f}"
                        )
                        st.write(source["text"])

else:
    st.info(
        "Upload at least one supported document or paste a public/shared "
        "Google Drive file or folder link to begin."
    )
