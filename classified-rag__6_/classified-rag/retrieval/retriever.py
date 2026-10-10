"""
Wires embedding -> clearance-filtered ANN search -> per-user redaction ->
entity-aware query expansion -> rerank into one call.

Order matters for security:
  1. vectorstore.search()         hard SQL clearance filter (nothing above the
                                  caller's level ever leaves the database)
  2. security.redaction           mask spans this user may not see, BEFORE the
                                  reranker, entity resolution or the LLM read any text
  3. entity_resolution            alias discovery runs only on redacted text
"""
from __future__ import annotations

from dataclasses import dataclass, field

from config import CFG
from embedding.embedder import embed_query
from entity_resolution import resolver
from entity_resolution.resolver import ResolvedEntity
from retrieval.reranker import rerank
from security.classification import RequestContext
from security.redaction import redact_chunks
from vectorstore.pgvector_store import RetrievedChunk, search


@dataclass
class RetrievalResult:
    sources: list[RetrievedChunk]
    entities: list[ResolvedEntity] = field(default_factory=list)
    expansion: list[str] = field(default_factory=list)  # alias terms used for the 2nd search
    redaction_count: int = 0                            # redactions inside the final sources


def _search_visible(q_emb, ctx: RequestContext) -> list[RetrievedChunk]:
    hits = search(q_emb, max_clearance=ctx.max_clearance, allowed_doc_ids=ctx.allowed_doc_ids)
    visible, _ = redact_chunks(hits, ctx)
    return visible


def retrieve_with_entities(query: str, ctx: RequestContext) -> RetrievalResult:
    candidates = _search_visible(embed_query(query), ctx)

    q_mentions, expansion = [], []
    if CFG.entity_resolution:
        q_mentions = resolver.query_mentions(query)
        if q_mentions:
            expansion = resolver.discover_aliases(query, q_mentions, candidates)
            if expansion:
                # Same question, plus the other names those entities go by. Same
                # clearance filter + redaction as the first pass.
                expanded = f"{query}\n(also referred to as: {'; '.join(expansion)})"
                seen = {c.id for c in candidates}
                candidates += [c for c in _search_visible(embed_query(expanded), ctx) if c.id not in seen]

    sources = rerank(query, candidates)  # rank on the ORIGINAL question
    entities = resolver.resolve_sources(q_mentions, sources) if CFG.entity_resolution else []
    redactions = sum(int((s.extra or {}).get("redactions", 0)) for s in sources)
    return RetrievalResult(sources, entities, expansion, redactions)


def retrieve(query: str, ctx: RequestContext) -> list[RetrievedChunk]:
    """Backwards-compatible: just the ranked, redacted source passages."""
    return retrieve_with_entities(query, ctx).sources
