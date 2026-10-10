"""
BGE-M3 embeddings. One model handles dense + sparse + multi-vector, but this
scaffold uses just the dense vectors for a plain pgvector column -- add the
sparse/colbert vectors later if you want hybrid search.

Runs on CFG.small_models_device (default "cpu") so the GPU stays free for
the LLM on an 8 GB card. Set SMALL_MODELS_DEVICE=cuda for faster bulk ingestion.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from config import CFG


@lru_cache(maxsize=1)
def _get_model():
    from FlagEmbedding import BGEM3FlagModel

    dev = CFG.small_models_device
    fp16 = dev != "cpu"
    try:  # FlagEmbedding >= 1.3 uses `devices`; older releases use `device`
        return BGEM3FlagModel(CFG.embedding_model, use_fp16=fp16, devices=dev)
    except TypeError:
        return BGEM3FlagModel(CFG.embedding_model, use_fp16=fp16, device=dev)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Returns an (N, CFG.embedding_dim) float32 array of dense embeddings."""
    model = _get_model()
    out = model.encode(texts, batch_size=16, max_length=1024)
    return np.asarray(out["dense_vecs"], dtype="float32")


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]
