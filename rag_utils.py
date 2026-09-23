"""
rag_utils.py
Handles document loading, chunking, embedding, FAISS indexing, and retrieval.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Constants ────────────────────────────────────────────────────────────────
CHUNK_SIZE = 400          # characters per chunk
CHUNK_OVERLAP = 80        # overlap between consecutive chunks
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K_DEFAULT = 5
SAMPLE_DATA_DIR = Path(__file__).parent / "sample_data"

# ── Model (cached at module level, loaded once per process) ──────────────────
_embed_model: SentenceTransformer | None = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


# ── Text extraction ──────────────────────────────────────────────────────────

def extract_text_from_txt(raw: bytes) -> str:
    """Decode a .txt file bytes to string."""
    for enc in ("utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def extract_text_from_pdf(raw: bytes) -> str:
    """Extract text from PDF bytes using pypdf."""
    try:
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except Exception as exc:
        raise RuntimeError(f"PDF extraction failed: {exc}") from exc


def extract_text(filename: str, raw: bytes) -> str:
    """Route extraction by file extension."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(raw)
    return extract_text_from_txt(raw)


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, source_name: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[dict]:
    """
    Split text into overlapping character-based chunks.
    Returns list of dicts: {text, source, chunk_id}
    """
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks: List[dict] = []
    start = 0
    chunk_id = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "text": chunk,
                "source": source_name,
                "chunk_id": chunk_id,
            })
            chunk_id += 1
        start += chunk_size - overlap
    return chunks


# ── FAISS index management ────────────────────────────────────────────────────

def build_faiss_index(chunks: List[dict]) -> Tuple[faiss.IndexFlatL2, List[dict]]:
    """
    Embed all chunks and build a FAISS flat L2 index.
    Returns (index, chunks_list).
    """
    model = get_embed_model()
    texts = [c["text"] for c in chunks]
    embeddings: np.ndarray = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, chunks


def add_to_faiss_index(
    existing_index: faiss.IndexFlatL2 | None,
    existing_chunks: List[dict],
    new_chunks: List[dict],
) -> Tuple[faiss.IndexFlatL2, List[dict]]:
    """
    Add new chunks to an existing FAISS index (or create a new one).
    """
    if not new_chunks:
        return existing_index, existing_chunks

    model = get_embed_model()
    texts = [c["text"] for c in new_chunks]
    new_embeddings: np.ndarray = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    new_embeddings = new_embeddings.astype(np.float32)

    if existing_index is None:
        dim = new_embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
    else:
        index = existing_index

    index.add(new_embeddings)
    all_chunks = existing_chunks + new_chunks
    return index, all_chunks


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    index: faiss.IndexFlatL2,
    chunks: List[dict],
    top_k: int = TOP_K_DEFAULT,
) -> List[dict]:
    """
    Embed the query and retrieve top-k most similar chunks from FAISS.
    Returns list of chunk dicts sorted by similarity (best first).
    """
    model = get_embed_model()
    q_emb: np.ndarray = model.encode([query], convert_to_numpy=True).astype(np.float32)
    k = min(top_k, len(chunks))
    distances, indices = index.search(q_emb, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        chunk = dict(chunks[idx])
        chunk["score"] = float(dist)
        results.append(chunk)
    return results


# ── Sample data loader ────────────────────────────────────────────────────────

def load_sample_data() -> Tuple[faiss.IndexFlatL2, List[dict]]:
    """
    Load all .txt files from sample_data/, chunk and embed them.
    Returns (index, all_chunks).
    """
    all_chunks: List[dict] = []
    if not SAMPLE_DATA_DIR.exists():
        return None, []

    for txt_file in sorted(SAMPLE_DATA_DIR.glob("*.txt")):
        raw = txt_file.read_bytes()
        text = extract_text_from_txt(raw)
        chunks = chunk_text(text, source_name=txt_file.stem.replace("_", " ").title())
        all_chunks.extend(chunks)

    if not all_chunks:
        return None, []

    index, all_chunks = build_faiss_index(all_chunks)
    return index, all_chunks


# ── Helpers ───────────────────────────────────────────────────────────────────

def unique_sources(chunks: List[dict]) -> List[str]:
    """Return deduplicated list of source names from a chunk list."""
    seen = set()
    sources = []
    for c in chunks:
        s = c.get("source", "Unknown")
        if s not in seen:
            seen.add(s)
            sources.append(s)
    return sources
