"""
pgvector-backed store. All read paths take a `max_clearance` argument and
turn it into a hard SQL predicate -- this is the "zero data spillage" control
point. Nothing downstream (LLM, reranker) ever sees a chunk above the
caller's clearance, because it's filtered out before it leaves the database.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from config import CFG, clearance_rank
from ingestion.doc_ingest import Chunk


@dataclass
class RetrievedChunk:
    id: int
    text: str
    source_path: str
    doc_id: str
    page: int | None
    classification: str
    extra: dict
    score: float  # similarity score from the ANN search (higher = closer)


def _connect():
    conn = psycopg2.connect(CFG.pg_dsn)
    register_vector(conn)
    return conn


def upsert_chunks(chunks: list[Chunk], embeddings) -> None:
    if not chunks:
        return
    with _connect() as conn, conn.cursor() as cur:
        rows = [
            (
                c.doc_id,
                c.doc_sha256,
                c.source_path,
                c.page,
                c.chunk_index,
                c.text,
                c.classification_level,
                json.dumps(c.extra),
                emb,
            )
            for c, emb in zip(chunks, embeddings)
        ]
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO chunks
                (doc_id, doc_sha256, source_path, page, chunk_index, text,
                 classification, extra, embedding)
            VALUES %s
            """,
            rows,
        )
        conn.commit()


def search(
    query_embedding,
    max_clearance: str,
    top_k: int = CFG.top_k_retrieve,
    allowed_doc_ids: list[str] | None = None,
) -> list[RetrievedChunk]:
    """
    Mandatory-access-control filter lives here, not in the prompt.
    `max_clearance` is resolved server-side from the caller's authenticated
    session -- never trust a client-supplied clearance level as-is; this
    function assumes the caller already did that lookup.
    """
    allowed_levels = CFG.classification_levels[: clearance_rank(max_clearance) + 1]

    sql = """
        SELECT id, text, source_path, doc_id, page, classification, extra,
               1 - (embedding <=> %s) AS score
        FROM chunks
        WHERE classification = ANY(%s)
    """
    params: list = [query_embedding, allowed_levels]

    if allowed_doc_ids:
        sql += " AND doc_id = ANY(%s)"
        params.append(allowed_doc_ids)

    sql += " ORDER BY embedding <=> %s LIMIT %s"
    params.extend([query_embedding, top_k])

    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    return [
        RetrievedChunk(
            id=r[0],
            text=r[1],
            source_path=r[2],
            doc_id=r[3],
            page=r[4],
            classification=r[5],
            extra=r[6],
            score=float(r[7]),
        )
        for r in rows
    ]
