"""Hybrid retrieval: vector + BM25 -> Reciprocal Rank Fusion -> cross-encoder rerank."""
from sentence_transformers import CrossEncoder

from .config import RERANK_MODEL, TOP_K_FINAL, TOP_K_RETRIEVE
from .store import DocumentStore

MODES = ("vector", "hybrid", "hybrid_rerank")


def reciprocal_rank_fusion(rank_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Merge ranked lists. Each hit earns 1/(k + rank); hits found by both retrievers rise."""
    scores: dict[str, float] = {}
    by_id: dict[str, dict] = {}
    for hits in rank_lists:
        for rank, hit in enumerate(hits):
            scores[hit["id"]] = scores.get(hit["id"], 0.0) + 1.0 / (k + rank + 1)
            by_id.setdefault(hit["id"], hit)
    return [{**by_id[i], "score": s}
            for i, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


class Retriever:
    def __init__(self, store: DocumentStore, rerank_model: str = RERANK_MODEL):
        self.store = store
        self.reranker = CrossEncoder(rerank_model)

    def retrieve(self, query: str, sources: list[str] | None = None,
                 mode: str = "hybrid_rerank", top_k: int = TOP_K_FINAL) -> list[dict]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")

        vec = self.store.vector_search(query, TOP_K_RETRIEVE, sources)
        if mode == "vector":
            return vec[:top_k]

        bm25 = self.store.bm25_search(query, TOP_K_RETRIEVE, sources)
        fused = reciprocal_rank_fusion([vec, bm25])
        if mode == "hybrid":
            return fused[:top_k]

        candidates = fused[:TOP_K_RETRIEVE]
        if not candidates:
            return []
        scores = self.reranker.predict([(query, h["text"]) for h in candidates])
        for hit, score in zip(candidates, scores):
            hit["score"] = float(score)
        return sorted(candidates, key=lambda h: h["score"], reverse=True)[:top_k]
