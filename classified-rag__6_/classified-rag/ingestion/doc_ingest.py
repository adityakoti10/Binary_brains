"""
Document ingestion: scanned PDFs, manuals, reports -> structured chunks.

Primary path: Docling (layout + OCR + table structure in one pass).
If Docling isn't installed / the file type isn't supported, falls back
to a plain-text reader so the pipeline never hard-fails on ingestion.

Every chunk that comes out of this module carries the provenance fields
the rest of the system depends on: source_path, page, sha256 hash,
and a classification_level the caller must supply (this is NOT inferred
from content -- classification is a human/system decision, never an AI
guess).

Portion-marked documents ("(U) ...", "(S) ...", "(TS) ..." paragraph marks) can be
ingested with portion_marked=True: the text is split at the marks so each chunk
carries a single classification (see security/redaction.py). Off by default.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from security.redaction import group_portions, split_portions


@dataclass
class Chunk:
    text: str
    source_path: str
    doc_id: str
    page: int | None
    classification_level: str
    doc_sha256: str
    chunk_index: int
    extra: dict = field(default_factory=dict)  # e.g. bbox, table_id, log_template_id


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _chunk_text(text: str, max_chars: int = 1800, overlap: int = 200) -> list[str]:
    """Simple sliding-window chunker. Window edges are nudged back to whitespace so a
    word (or a name/codeword the redaction rules must match) is not cut in half.
    Swap for a semantic chunker later (docling emits structural units you can chunk on)."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            ws = text.rfind(" ", start + max_chars // 2, end)
            if ws == -1:
                ws = text.rfind("\n", start + max_chars // 2, end)
            if ws != -1:
                end = ws
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
        nxt = text.find(" ", start, end)  # start the next window on a word boundary
        if nxt != -1:
            start = nxt + 1
    return chunks


def _classified_pieces(
    text: str, doc_level: str, portion_marked: bool
) -> list[tuple[str, str, dict]]:
    """Return [(piece_text, classification_level, extra)] for one block of text.
    Ordinary documents: every piece gets the document label.  Portion-marked
    documents: pieces never mix levels -- each run of equally-marked paragraphs is
    chunked on its own and labelled with its own level."""
    if not portion_marked:
        return [(piece, doc_level, {}) for piece in _chunk_text(text)]
    out = []
    for run in group_portions(split_portions(text, doc_level)):
        extra = {"portion_marked": True, "portion_mark": run.mark}
        if run.exceeds_label:
            extra["portion_exceeds_label"] = True
        out.extend((piece, run.level, dict(extra)) for piece in _chunk_text(run.text))
    return out


def ingest_with_docling(
    path: Path, classification_level: str, portion_marked: bool = False
) -> list[Chunk]:
    from docling.document_converter import DocumentConverter  # heavy import, local

    converter = DocumentConverter()
    result = converter.convert(str(path))
    doc_sha = _sha256_file(path)
    chunks: list[Chunk] = []

    # docling exposes page-aware structured elements; export per-page markdown
    # as a pragmatic default, then sub-chunk. Swap in result.document.texts /
    # .tables directly if you want table-aware chunk boundaries.
    for page_no, page_md in enumerate(result.document.export_to_markdown_pages(), start=1):
        for piece, level, extra in _classified_pieces(page_md, classification_level, portion_marked):
            chunks.append(
                Chunk(
                    text=piece,
                    source_path=str(path),
                    doc_id=doc_sha[:16],
                    page=page_no,
                    classification_level=level,
                    doc_sha256=doc_sha,
                    chunk_index=len(chunks),
                    extra=extra,
                )
            )
    return chunks


def ingest_plaintext_fallback(
    path: Path, classification_level: str, portion_marked: bool = False
) -> list[Chunk]:
    doc_sha = _sha256_file(path)
    text = path.read_text(errors="ignore")
    return [
        Chunk(
            text=piece,
            source_path=str(path),
            doc_id=doc_sha[:16],
            page=None,
            classification_level=level,
            doc_sha256=doc_sha,
            chunk_index=i,
            extra=extra,
        )
        for i, (piece, level, extra) in enumerate(
            _classified_pieces(text, classification_level, portion_marked)
        )
    ]


def ingest_document(
    path: str | Path, classification_level: str, portion_marked: bool = False
) -> list[Chunk]:
    """Single entry point the pipeline calls. classification_level MUST be
    supplied by the caller (from a manifest / tagging system), never inferred.
    portion_marked=True means the caller asserts the file carries (U)/(C)/(S)/(TS)
    paragraph marks (classification_level is then the document's overall level)."""
    path = Path(path)
    try:
        return ingest_with_docling(path, classification_level, portion_marked)
    except Exception as e:  # noqa: BLE001 - deliberate broad fallback for a demo scaffold
        print(f"[ingest] docling unavailable/failed for {path} ({e}); using plaintext fallback")
        return ingest_plaintext_fallback(path, classification_level, portion_marked)


def ingest_directory(
    dir_path: str | Path, classification_level: str, portion_marked: bool = False
) -> Iterable[Chunk]:
    for p in Path(dir_path).rglob("*"):
        if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md", ".docx", ".png", ".jpg", ".jpeg"}:
            yield from ingest_document(p, classification_level, portion_marked)
