"""Storage + the two raw retrievers: dense vectors (ChromaDB) and keyword (BM25)."""
import re
from collections import Counter

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .config import CHROMA_DIR, EMBED_MODEL
from .ingestion import Chunk


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class DocumentStore:
    def __init__(self, persist_dir=CHROMA_DIR, embed_model: str = EMBED_MODEL):
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = SentenceTransformer(embed_model)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.col = self.client.get_or_create_collection(
            "documents", metadata={"hnsw:space": "cosine"}
        )
        self._bm25: BM25Okapi | None = None
        self._bm25_hits: list[dict] = []
        self._dirty = True

    # ---------- writing ----------
    def add_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            embeddings = self.embedder.encode(
                [c.text for c in batch], normalize_embeddings=True
            ).tolist()
            self.col.upsert(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                embeddings=embeddings,
                metadatas=[{"source": c.source, "page": c.page} for c in batch],
            )
        self._dirty = True

    def delete_document(self, source: str) -> None:
        self.col.delete(where={"source": source})
        self._dirty = True

    # ---------- reading ----------
    def list_documents(self) -> dict[str, int]:
        """Map of filename -> number of chunks."""
        metas = self.col.get(include=["metadatas"])["metadatas"]
        return dict(Counter(m["source"] for m in metas))

    @staticmethod
    def _where(sources: list[str] | None):
        return {"source": {"$in": sources}} if sources else None

    def vector_search(self, query: str, k: int, sources: list[str] | None = None) -> list[dict]:
        if self.col.count() == 0:
            return []
        emb = self.embedder.encode([query], normalize_embeddings=True).tolist()
        res = self.col.query(
            query_embeddings=emb,
            n_results=min(k, self.col.count()),
            where=self._where(sources),
        )
        return [
            {"id": id_, "text": doc, "source": meta["source"], "page": meta["page"],
             "score": 1 - dist}
            for id_, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def _build_bm25(self) -> None:
        data = self.col.get(include=["documents", "metadatas"])
        self._bm25_hits = [
            {"id": id_, "text": doc, "source": meta["source"], "page": meta["page"], "score": 0.0}
            for id_, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
        ]
        self._bm25 = (
            BM25Okapi([tokenize(h["text"]) for h in self._bm25_hits]) if self._bm25_hits else None
        )
        self._dirty = False

    def bm25_search(self, query: str, k: int, sources: list[str] | None = None) -> list[dict]:
        if self._dirty:
            self._build_bm25()
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self._bm25_hits, scores), key=lambda x: x[1], reverse=True)
        out = []
        for hit, score in ranked:
            if score <= 0:
                break
            if sources and hit["source"] not in sources:
                continue
            out.append({**hit, "score": float(score)})
            if len(out) >= k:
                break
        return out
