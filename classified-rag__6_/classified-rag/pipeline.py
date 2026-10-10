"""
End-to-end orchestration:

    ingest -> (portion-marking split) -> triage tags -> entity mentions
           -> embed -> store                       (offline / on upload)
    query  -> retrieve (clearance filter -> per-user redaction
                        -> entity alias expansion -> rerank)
           -> compare sources against each other (cross-source conflicts)
           -> synthesize (cited answer, Qwen via Ollama)
           -> decompose claims -> NLI-verify each
           -> write provenance graph + signed audit log
           -> return (answer, sources, claim verdicts, conflicts, entities, KPIs)

CLI demo (needs a user token from `python -m security.users add ...`):
    python pipeline.py "What maintenance interval does the manual specify?" <TOKEN>
"""
from __future__ import annotations

import sys
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from config import CFG
from embedding.embedder import embed_texts
from entity_resolution.resolver import annotate_chunks, entity_notes
from ingestion.doc_ingest import Chunk, ingest_directory, ingest_document
from provenance.audit import AuditRecord, write_audit_log
from retrieval.retriever import retrieve_with_entities
from routing.document_router import classify_chunk
from security.classification import RequestContext, resolve_user_clearance
from security.redaction import redact_text
from synthesis.llm_client import synthesize
from verification.claim_decompose import decompose
from verification.conflict_detector import detect_conflicts, mark_acknowledged
from verification.nli_checker import hallucination_rate, verify_answer
from vectorstore.pgvector_store import upsert_chunks


# ---------------------------------------------------------------- INGEST ---
def _store(doc_chunks: list[Chunk], log_chunks: list[Chunk], triage: bool) -> int:
    chunks = doc_chunks + log_chunks
    if not chunks:
        return 0
    if triage:  # fast local classifier (SetFit or Laya) -- never the synthesis LLM
        for c in doc_chunks:
            c.extra.update(classify_chunk(c.text, is_log=False))
        for c in log_chunks:
            c.extra.update(classify_chunk(c.text, is_log=True))
    annotate_chunks(chunks)  # entity mentions -> chunk.extra["entities"] (fail-soft)
    embeddings = embed_texts([c.text for c in chunks])
    upsert_chunks(chunks, embeddings)
    return len(chunks)


def ingest_files(
    paths: list[str | Path],
    classification_level: str,
    triage: bool = True,
    portion_marked: bool = False,
) -> int:
    """Ingest specific files (used by the web upload). *.log files go through the
    Drain3 log parser; everything else through Docling/plaintext. Returns #chunks stored.
    portion_marked=True: the (non-log) files carry (U)/(C)/(S)/(TS) paragraph marks and
    are split into chunks that each get their own classification level."""
    from ingestion.log_ingest import logs_to_chunks  # lazy: needs drain3

    doc_chunks: list[Chunk] = []
    log_chunks: list[Chunk] = []
    for p in map(Path, paths):
        if p.suffix.lower() == ".log":
            log_chunks.extend(logs_to_chunks(p, classification_level))
        else:
            doc_chunks.extend(ingest_document(p, classification_level, portion_marked))
    n = _store(doc_chunks, log_chunks, triage)
    by_level = Counter(c.classification_level for c in doc_chunks + log_chunks)
    detail = ", ".join(f"{k}: {v}" for k, v in sorted(by_level.items())) if portion_marked else classification_level
    print(f"[ingest] stored {n} chunks from {len(paths)} file(s) ({detail})")
    return n


def run_ingestion(
    doc_dir: str,
    log_dir: str | None,
    classification_level: str,
    triage: bool = True,
    portion_marked: bool = False,
) -> None:
    """Batch-load a directory of documents (and optionally *.log files)."""
    from ingestion.log_ingest import logs_to_chunks

    doc_chunks = list(ingest_directory(doc_dir, classification_level, portion_marked))
    log_chunks: list[Chunk] = []
    if log_dir:
        for log_file in Path(log_dir).glob("*.log"):
            log_chunks.extend(logs_to_chunks(log_file, classification_level))
    n = _store(doc_chunks, log_chunks, triage)
    if n == 0:
        print("[ingest] no chunks produced -- check doc_dir/log_dir")
    else:
        print(f"[ingest] stored {n} chunks from {doc_dir}" + (f" + logs from {log_dir}" if log_dir else ""))


# ---------------------------------------------------------------- QUERY ----
@dataclass
class AnswerResult:
    answer: str
    hallucination_rate: float
    supported_claims: int
    unsupported_claims: int
    contradicted_claims: int
    cited_doc_ids: list[str]
    sources: list[dict] = field(default_factory=list)  # what [S1], [S2]... refer to
    claims: list[dict] = field(default_factory=list)   # per-claim verification verdicts
    conflicts: list[dict] = field(default_factory=list)  # source-vs-source disagreements
    entities: list[dict] = field(default_factory=list)   # resolved entities across the sources
    query_expansion: list[str] = field(default_factory=list)  # alias terms added to the search
    redaction_count: int = 0        # spans masked for this user (sources + answer)
    conflict_check: str = "skipped"  # "ok" | "skipped" | "failed" (NLI model unavailable)

    def to_dict(self) -> dict:
        return asdict(self)


def answer_query(query: str, ctx: RequestContext, log_provenance: bool = True) -> AnswerResult:
    # clearance filter -> per-user redaction -> entity alias expansion -> rerank
    retrieval = retrieve_with_entities(query, ctx)
    sources = retrieval.sources
    if not sources:
        return AnswerResult(
            answer="No sources at or below your clearance level matched this query.",
            hallucination_rate=0.0, supported_claims=0, unsupported_claims=0,
            contradicted_claims=0, cited_doc_ids=[],
        )
    entities = retrieval.entities

    # Do the sources agree with EACH OTHER? (advisory: a failure here must not kill the answer,
    # but it is reported to the user instead of being silently skipped)
    conflicts, conflict_check = [], "skipped"
    if CFG.conflict_detection:
        try:
            conflicts = detect_conflicts(sources, entities)
            conflict_check = "ok"
        except Exception as e:  # noqa: BLE001
            conflict_check = "failed"
            print(f"[conflicts] cross-source check failed: {e}")

    raw_answer, sources = synthesize(query, sources, conflicts, entity_notes(entities))
    # Defence in depth: the LLM only saw redacted text, but mask the answer too.
    answer_red = redact_text(raw_answer, ctx.max_clearance, ctx.user_id)
    raw_answer = answer_red.text
    redaction_count = retrieval.redaction_count + answer_red.count

    claims = decompose(raw_answer)
    source_texts = {i + 1: s.text for i, s in enumerate(sources)}  # redacted text
    verdicts = verify_answer(claims, source_texts)
    mark_acknowledged(conflicts, claims)

    supported = sum(1 for v in verdicts if v.status == "supported")
    unsupported = sum(1 for v in verdicts if v.status == "unsupported")
    contradicted = sum(1 for v in verdicts if v.status == "contradicted")
    h_rate = hallucination_rate(verdicts)
    cited_doc_ids = sorted({s.doc_id for s in sources})

    if log_provenance:
        answer_id = str(uuid.uuid4())
        try:
            from provenance.graph_store import ProvenanceGraph  # lazy: needs neo4j

            graph = ProvenanceGraph()
            graph.record_answer(answer_id, query, sources, verdicts, conflicts, entities)
            graph.close()
        except Exception as e:  # noqa: BLE001
            print(f"[provenance] graph write failed (is Neo4j running?): {e}")

        write_audit_log(
            AuditRecord(
                query=query,
                answer=raw_answer,
                max_clearance_used=ctx.max_clearance,
                cited_doc_ids=cited_doc_ids,
                hallucination_rate=h_rate,
                unsupported_claim_count=unsupported,
                contradicted_claim_count=contradicted,
                timestamp=0.0,  # filled in by write_audit_log
                user_id=ctx.user_id,
                redaction_count=redaction_count,
                conflict_count=len(conflicts),
            )
        )

    return AnswerResult(
        answer=raw_answer,
        hallucination_rate=h_rate,
        supported_claims=supported,
        unsupported_claims=unsupported,
        contradicted_claims=contradicted,
        cited_doc_ids=cited_doc_ids,
        sources=[
            {
                "index": i + 1,
                "doc_id": s.doc_id,
                "source_path": Path(s.source_path).name,  # file name only, not server paths
                "page": s.page,
                "classification": s.classification,
                "score": round(s.score, 3),
                "snippet": s.text[:400],  # already redacted for this user
                "redactions": int((s.extra or {}).get("redactions", 0)),
            }
            for i, s in enumerate(sources)
        ],
        claims=[
            {
                "text": v.claim.text,
                "cited": v.claim.cited_indices,
                "status": v.status,
                "entailment": round(v.entailment_prob, 3),
            }
            for v in verdicts
        ],
        conflicts=[c.to_dict() for c in conflicts],
        entities=[e.to_dict() for e in entities],
        query_expansion=retrieval.expansion,
        redaction_count=redaction_count,
        conflict_check=conflict_check,
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit('usage: python pipeline.py "your question" <USER_TOKEN>')
    q, token = sys.argv[1], sys.argv[2]
    ctx = resolve_user_clearance(token)  # clearance comes from the server-side user table

    result = answer_query(q, ctx)
    print("\n=== ANSWER ===")
    print(result.answer)
    print("\n=== KPIs ===")
    print(f"hallucination_rate:   {result.hallucination_rate:.2%}")
    print(f"supported claims:     {result.supported_claims}")
    print(f"unsupported claims:   {result.unsupported_claims}")
    print(f"contradicted claims:  {result.contradicted_claims}")
    print(f"cited doc_ids:        {result.cited_doc_ids}")
    print(f"redactions applied:   {result.redaction_count}")
    print(f"source conflicts:     {len(result.conflicts)} (check: {result.conflict_check})")
    for c in result.conflicts:
        print(f"  [S{c['source_a']}] vs [S{c['source_b']}] ({c['kind']}): "
              f"{c['sentence_a'][:80]!r} <> {c['sentence_b'][:80]!r}")
    if result.query_expansion:
        print(f"query also searched:  {result.query_expansion}")
    for e in result.entities:
        if e["aliases"]:
            print(f"entity: {e['canonical']} also seen as {e['aliases']}")
