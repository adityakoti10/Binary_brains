"""bge-reranker-v2-m3 cross-encoder reranking of the ANN candidate set."""
from __future__ import annotations

from functools import lru_cache

from config import CFG
from vectorstore.pgvector_store import RetrievedChunk


@lru_cache(maxsize=1)
def _get_reranker():
    from FlagEmbedding import FlagReranker

    dev = CFG.small_models_device
    fp16 = dev != "cpu"
    try:
        return FlagReranker(CFG.reranker_model, use_fp16=fp16, devices=dev)
    except TypeError:
        return FlagReranker(CFG.reranker_model, use_fp16=fp16, device=dev)


def rerank(query: str, candidates: list[RetrievedChunk], top_k: int = CFG.top_k_rerank) -> list[RetrievedChunk]:
    if not candidates:
        return []
    reranker = _get_reranker()
    pairs = [[query, c.text] for c in candidates]
    scores = reranker.compute_score(pairs, normalize=True)
    if not isinstance(scores, list):  # single pair returns a bare float
        scores = [scores]
    scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    ranked = []
    for c, s in scored[:top_k]:
        c.score = float(s)  # overwrite ANN score with the (more precise) rerank score
        ranked.append(c)
    return ranked
