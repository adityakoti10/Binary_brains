"""
System-log ingestion via Drain3 online log-template mining.
Turns a stream of raw log lines into (template, variable-parts) pairs so
logs become retrievable/citable just like document chunks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from ingestion.doc_ingest import Chunk, _sha256_file


@dataclass
class LogEvent:
    raw_line: str
    template: str
    template_id: int
    line_no: int


def _make_miner() -> TemplateMiner:
    cfg = TemplateMinerConfig()
    cfg.load(None)  # sensible built-in defaults; point at a .ini for custom masking rules
    return TemplateMiner(config=cfg)


def parse_log_file(path: str | Path) -> list[LogEvent]:
    miner = _make_miner()
    events = []
    with open(path, errors="ignore") as f:
        for i, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            result = miner.add_log_message(line)
            events.append(
                LogEvent(
                    raw_line=line,
                    template=result["template_mined"],
                    template_id=result["cluster_id"],
                    line_no=i,
                )
            )
    return events


def logs_to_chunks(
    path: str | Path, classification_level: str, batch_size: int = 40
) -> list[Chunk]:
    """Group consecutive log lines into small windows so each chunk still
    reads as coherent context (a single log line is rarely a useful RAG hit)."""
    path = Path(path)
    doc_sha = _sha256_file(path)
    events = parse_log_file(path)

    chunks: list[Chunk] = []
    for start in range(0, len(events), batch_size):
        window = events[start : start + batch_size]
        text = "\n".join(e.raw_line for e in window)
        chunks.append(
            Chunk(
                text=text,
                source_path=str(path),
                doc_id=doc_sha[:16],
                page=None,
                classification_level=classification_level,
                doc_sha256=doc_sha,
                chunk_index=len(chunks),
                extra={
                    "line_range": [window[0].line_no, window[-1].line_no],
                    "template_ids": sorted({e.template_id for e in window}),
                },
            )
        )
    return chunks
