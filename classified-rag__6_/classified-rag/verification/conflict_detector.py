"""
Cross-source conflict detection: do the retrieved passages disagree WITH EACH OTHER?

(verification/nli_checker.py checks each answer claim against the source it cites.
That can't notice that S2 says "500 hours" and S5 says "250 hours" if the answer only
cites one of them -- this module compares the sources themselves.)

Pipeline, cheap stages first so the NLI model only sees a handful of pairs:

  1. split every source into sentences
  2. canonicalise entity aliases ("J. Smith" -> "John Smith")     [entity resolution]
  3. block: keep sentence pairs from DIFFERENT documents that share enough content
     words (or share a resolved entity plus some words); rank; keep the best
     CFG.conflict_max_pairs
  4. batched NLI on those pairs; a pair is a conflict when P(contradiction) is high
     in BOTH directions (average >= CFG.conflict_contradiction_threshold)
  5. numeric-mismatch rule: nearly identical sentences whose numbers differ
     ("500 hours" vs "250 hours", "12 May" vs "14 May") -- generic MNLI models are
     weak on numbers, so this catches what NLI misses. Reported as kind="numeric"
     (a *possible* conflict).

Everything operates on the already-redacted source text. Conflicts are advisory:
they go to the LLM prompt, the UI, the provenance graph and the audit log -- they
never delete or rewrite a source.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from config import CFG
from entity_resolution.resolver import ResolvedEntity, canonicalize, entities_in
from verification.claim_decompose import split_sentences as _split


@dataclass
class Conflict:
    source_a: int                 # 1-based [Sx] indices
    source_b: int
    sentence_a: str
    sentence_b: str
    kind: str                     # "contradiction" (NLI) | "numeric" (number mismatch)
    contradiction: float | None   # NLI probability; None for the numeric rule
    acknowledged: bool = False    # does the final answer cite both sides together?

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------- helpers ---
_STOP = {
    "a", "an", "the", "of", "to", "in", "on", "at", "by", "for", "with", "from", "and", "or",
    "is", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that", "these",
    "those", "as", "than", "then", "has", "have", "had", "will", "shall", "should", "must",
    "can", "may", "not", "no", "per", "into", "over", "under", "after", "before", "which",
}
_WORD = re.compile(r"[^\W_]+(?:[-.][^\W_]+)*", re.UNICODE)
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)*(?![\w])")
_ID_TOKEN = re.compile(r"\b(?=[A-Za-z0-9-]*\d)(?=[A-Za-z0-9-]*[A-Za-z])[A-Za-z0-9-]+\b")
_REDACTED = re.compile(r"\[REDACTED:[^\]]*\]")


def split_sentences(text: str, max_sentences: int) -> list[str]:
    out = []
    for line in text.split("\n"):
        for s in _split(line):
            s = s.strip(" \t|#>*-")
            if len(s) >= 20 and len(s.split()) >= 4:
                out.append(s)
            if len(out) >= max_sentences:
                return out
    return out


def _stem(w: str) -> str:
    w = w.lower()
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _content_words(s: str, alpha_only: bool = False) -> set[str]:
    s = _REDACTED.sub(" ", s)
    words = {_stem(w) for w in _WORD.findall(s) if w.lower() not in _STOP and len(w) > 1}
    if alpha_only:
        words = {w for w in words if w.isalpha()}
    return words


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _numbers(s: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER.findall(_REDACTED.sub(" ", s))}


def _id_tokens(s: str) -> set[str]:
    return {t.lower() for t in _ID_TOKEN.findall(s)}


def numeric_mismatch(a: str, b: str) -> bool:
    """Same wording, different numbers -- and each sentence has a number the other lacks
    (so "500 hours (see Table 4)" vs "500 hours" is NOT flagged)."""
    if _jaccard(_content_words(a, alpha_only=True), _content_words(b, alpha_only=True)) < CFG.conflict_numeric_overlap:
        return False
    if _id_tokens(a) != _id_tokens(b):  # "M8 bolts" vs "M10 bolts": different objects
        return False
    na, nb = _numbers(a), _numbers(b)
    return bool(na - nb) and bool(nb - na)


# ----------------------------------------------------------------- main ---
def detect_conflicts(
    sources: list, entities: list[ResolvedEntity] | None = None
) -> list[Conflict]:
    """`sources` are the (redacted, ranked) RetrievedChunk objects behind [S1], [S2]...
    Raises if the NLI model can't run -- the caller decides how to degrade."""
    if not CFG.conflict_detection or len(sources) < 2:
        return []
    entities = entities or []

    # 1-2. sentences, alias-canonicalised
    sents: list[dict] = []
    for i, src in enumerate(sources, start=1):
        for raw in split_sentences(src.text, CFG.conflict_max_sentences_per_source):
            canon = canonicalize(raw, entities)
            sents.append({
                "src": i, "doc": src.doc_id, "raw": raw, "canon": canon,
                "words": _content_words(canon), "ents": entities_in(canon, entities),
            })

    # 3. blocking + ranking
    cands: list[tuple[float, dict, dict]] = []
    for x in range(len(sents)):
        for y in range(x + 1, len(sents)):
            a, b = sents[x], sents[y]
            if a["src"] == b["src"]:
                continue
            if CFG.conflict_cross_doc_only and a["doc"] == b["doc"]:
                continue
            ov = _jaccard(a["words"], b["words"])
            shared_entity = bool(a["ents"] & b["ents"])
            if ov >= CFG.conflict_min_overlap or (shared_entity and ov >= CFG.conflict_min_overlap * 0.6):
                cands.append((ov + (0.1 if shared_entity else 0.0), a, b))
    cands.sort(key=lambda t: -t[0])
    cands = cands[: CFG.conflict_max_pairs]
    if not cands:
        return []

    # 4. NLI, both directions on anything that looks like a contradiction
    from verification.nli_checker import nli_batch

    fwd = nli_batch([(a["canon"], b["canon"]) for _, a, b in cands], max_length=256)
    flagged = [k for k, (_, contra, _) in enumerate(fwd) if contra >= CFG.conflict_contradiction_threshold * 0.6]
    rev = dict(zip(
        flagged,
        nli_batch([(cands[k][2]["canon"], cands[k][1]["canon"]) for k in flagged], max_length=256),
    ))

    conflicts: list[Conflict] = []
    seen_pairs: set[tuple] = set()

    def _add(a: dict, b: dict, kind: str, prob: float | None) -> None:
        pair = (a["src"], b["src"], a["raw"], b["raw"])
        if pair in seen_pairs:
            return
        seen_pairs.add(pair)
        conflicts.append(Conflict(a["src"], b["src"], a["raw"], b["raw"], kind, prob))

    for k, (_, a, b) in enumerate(cands):
        if k in rev:
            avg = (fwd[k][1] + rev[k][1]) / 2
            if avg >= CFG.conflict_contradiction_threshold:
                _add(a, b, "contradiction", round(avg, 3))
                continue
        # 5. numbers the NLI model tends to miss
        if numeric_mismatch(a["canon"], b["canon"]):
            _add(a, b, "numeric", None)

    conflicts.sort(key=lambda c: (c.kind != "contradiction", -(c.contradiction or 0.0)))
    return conflicts[: CFG.conflict_max_reported]


def mark_acknowledged(conflicts: list[Conflict], claims: list) -> None:
    """A conflict is 'acknowledged' when some sentence of the final answer cites BOTH
    sides -- i.e. the answer says the sources disagree rather than silently picking one."""
    for c in conflicts:
        c.acknowledged = any({c.source_a, c.source_b} <= set(cl.cited_indices) for cl in claims)
