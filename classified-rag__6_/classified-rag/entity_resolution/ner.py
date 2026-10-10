"""Zero-shot NER with GLiNER -- extracts entities without retraining per
domain, useful across heterogeneous doc types (manuals, logs, reports).

GLiNER truncates long inputs, and our chunks run ~1800 characters, so long text
is processed in overlapping windows and the results are merged."""
from __future__ import annotations

from functools import lru_cache

from config import CFG

DEFAULT_ENTITY_TYPES = [
    "person", "organization", "location", "system", "device",
    "date", "identifier", "document_reference",
]

_WINDOW_CHARS = 1000
_WINDOW_OVERLAP = 120


@lru_cache(maxsize=1)
def _get_model():
    from gliner import GLiNER

    return GLiNER.from_pretrained(CFG.ner_model)


def _windows(text: str, size: int = _WINDOW_CHARS, overlap: int = _WINDOW_OVERLAP):
    """Yield (offset, piece) windows that end on whitespace where possible."""
    if len(text) <= size:
        yield 0, text
        return
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            ws = text.rfind(" ", start + size // 2, end)
            if ws != -1:
                end = ws
        yield start, text[start:end]
        if end >= len(text):
            return
        start = max(end - overlap, start + 1)


def extract_entities(
    text: str, entity_types: list[str] | None = None, threshold: float | None = None
) -> list[dict]:
    """Each result: {"text", "label", "start", "end", "score"} (offsets into `text`)."""
    model = _get_model()
    labels = entity_types or DEFAULT_ENTITY_TYPES
    thr = CFG.entity_min_score if threshold is None else threshold

    best: dict[tuple, dict] = {}
    for offset, piece in _windows(text):
        for e in model.predict_entities(piece, labels, threshold=thr):
            e = dict(e, start=e["start"] + offset, end=e["end"] + offset)
            key = (e["start"], e["end"], e["label"])
            if key not in best or e["score"] > best[key]["score"]:
                best[key] = e
    return sorted(best.values(), key=lambda e: e["start"])
