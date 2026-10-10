"""
Neo4j-backed provenance graph: (Answer)-[:CITES]->(Chunk)-[:PART_OF]->(Document),
plus (Claim)-[:VERIFIED_AS {status}]->(Chunk) so a reviewer can traverse from
any sentence in the final report back to the exact source passage and its
verification verdict.

Also recorded per answer:
  (Chunk)-[:CONFLICTS_WITH {answer_id, kind, contradiction, acknowledged}]->(Chunk)
      -- two cited passages that disagree (no passage text is stored on the edge)
  (Chunk)-[:MENTIONS {classification}]->(Entity {key, label, canonical})
      -- resolved entities; `classification` is the chunk's level so a reviewer
         can filter the graph. Entities only appear if they survived the
         requesting user's redaction.

Treat this database as holding the highest level of data it references.
"""
from __future__ import annotations

from neo4j import GraphDatabase

from config import CFG
from vectorstore.pgvector_store import RetrievedChunk
from verification.nli_checker import ClaimVerdict


class ProvenanceGraph:
    def __init__(self):
        self._driver = GraphDatabase.driver(CFG.neo4j_uri, auth=(CFG.neo4j_user, CFG.neo4j_password))

    def close(self):
        self._driver.close()

    def record_answer(
        self,
        answer_id: str,
        query: str,
        sources: list[RetrievedChunk],
        verdicts: list[ClaimVerdict],
        conflicts: list | None = None,
        entities: list | None = None,
    ) -> None:
        with self._driver.session() as session:
            session.execute_write(
                self._write_tx, answer_id, query, sources, verdicts, conflicts or [], entities or []
            )

    @staticmethod
    def _write_tx(tx, answer_id, query, sources, verdicts, conflicts=(), entities=()):
        tx.run(
            "MERGE (a:Answer {id: $answer_id}) SET a.query = $query, a.ts = timestamp()",
            answer_id=answer_id, query=query,
        )
        for s in sources:
            tx.run(
                """
                MERGE (d:Document {doc_id: $doc_id}) SET d.source_path = $source_path,
                                                            d.classification = $classification
                MERGE (c:Chunk {chunk_id: $chunk_id})
                MERGE (c)-[:PART_OF]->(d)
                MERGE (a:Answer {id: $answer_id})
                MERGE (a)-[:CITES]->(c)
                """,
                doc_id=s.doc_id, source_path=s.source_path, classification=s.classification,
                chunk_id=s.id, answer_id=answer_id,
            )
        for v in verdicts:
            for idx in v.claim.cited_indices:
                if idx - 1 < len(sources):
                    s = sources[idx - 1]
                    tx.run(
                        """
                        MERGE (c:Chunk {chunk_id: $chunk_id})
                        MERGE (cl:Claim {text: $claim_text, answer_id: $answer_id})
                        MERGE (cl)-[r:VERIFIED_AS]->(c)
                        SET r.status = $status, r.entailment = $entailment
                        """,
                        chunk_id=s.id, claim_text=v.claim.text, answer_id=answer_id,
                        status=v.status, entailment=v.entailment_prob,
                    )
        for c in conflicts:
            if not (1 <= c.source_a <= len(sources) and 1 <= c.source_b <= len(sources)):
                continue
            tx.run(
                """
                MERGE (a:Chunk {chunk_id: $a})
                MERGE (b:Chunk {chunk_id: $b})
                MERGE (a)-[r:CONFLICTS_WITH {answer_id: $answer_id}]->(b)
                SET r.kind = $kind, r.contradiction = $prob, r.acknowledged = $ack
                """,
                a=sources[c.source_a - 1].id, b=sources[c.source_b - 1].id,
                answer_id=answer_id, kind=c.kind, prob=c.contradiction, ack=c.acknowledged,
            )
        for e in entities:
            key = f"{e.label}:{e.canonical.lower()}"
            for idx in e.sources:
                if not 1 <= idx <= len(sources):
                    continue
                s = sources[idx - 1]
                tx.run(
                    """
                    MERGE (c:Chunk {chunk_id: $chunk_id})
                    MERGE (en:Entity {key: $key})
                    SET en.label = $label, en.canonical = $canonical, en.aliases = $aliases
                    MERGE (c)-[m:MENTIONS]->(en)
                    SET m.classification = $classification
                    """,
                    chunk_id=s.id, key=key, label=e.label, canonical=e.canonical,
                    aliases=e.aliases, classification=s.classification,
                )
