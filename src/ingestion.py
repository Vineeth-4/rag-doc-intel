"""Parse uploaded files and split them into page-aware chunks."""
import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from docx import Document

from .config import CHUNK_OVERLAP, CHUNK_SIZE

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int


def parse_file(filename: str, data: bytes) -> list[tuple[int, str]]:
    """Return a list of (page_number, text). DOCX/TXT have no pages, so they use page 1."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            return [(i + 1, page.get_text()) for i, page in enumerate(pdf)]
    if ext == ".docx":
        doc = Document(io.BytesIO(data))
        return [(1, "\n".join(p.text for p in doc.paragraphs))]
    if ext in {".txt", ".md"}:
        return [(1, data.decode("utf-8", errors="ignore"))]
    raise ValueError(f"Unsupported file type: {ext}")


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n\s*\n", text)
    # " ".join(p.split()) collapses every kind of whitespace (tabs, newlines, nbsp, em-space)
    return [" ".join(p.split()) for p in parts if p.strip()]


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Sentence-aware chunking with overlap, so ideas are rarely cut in half."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in _split_sentences(text):
        pieces = [sentence[i:i + size] for i in range(0, len(sentence), size)]
        for piece in pieces:
            if current and current_len + len(piece) > size:
                chunks.append(" ".join(current))
                keep, length = [], 0
                for prev in reversed(current):  # carry trailing sentences as overlap
                    if length + len(prev) > overlap:
                        break
                    keep.insert(0, prev)
                    length += len(prev)
                current, current_len = keep, length
            current.append(piece)
            current_len += len(piece)

    if current:
        chunks.append(" ".join(current))
    return chunks


def ingest_file(filename: str, data: bytes) -> list[Chunk]:
    """Parse + chunk one file. Chunk IDs are content-hashed, so re-uploads don't duplicate."""
    doc_hash = hashlib.sha1(data).hexdigest()[:10]
    chunks: list[Chunk] = []
    for page_num, page_text in parse_file(filename, data):
        for i, text in enumerate(chunk_text(page_text)):
            if len(text) < 30:  # skip page numbers, stray headers, etc.
                continue
            chunks.append(Chunk(f"{doc_hash}-p{page_num}-c{i}", text, filename, page_num))
    return chunks
