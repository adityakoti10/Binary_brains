"""
Entity resolution wired into the pipeline.  Three touch points:

  INGEST   annotate_chunks()   store the entities mentioned in each chunk
                               (chunk.extra["entities"]) so query time is cheap.

  RETRIEVE discover_aliases()  find the entities named in the QUESTION, look at how
                               the already-retrieved (clearance-filtered, redacted)
                               passages refer to them, and return the other name
                               forms.  retrieval/retriever.py uses them to run a
                               second search, so a question about "John Smith" also
                               finds the passage that only says "J. Smith".

  ANSWER   resolve_sources()   cluster the entities across the final source passages.
           canonicalize()      -> shown in the UI, hinted to the LLM, written to the
                                  provenance graph, and used by the conflict detector
                                  so "J. Smith reported 500 h" and "John Smith reported
                                  250 h" are recognised as talking about the same person.

Security: every function here only ever sees text that has ALREADY passed the
clearance filter and per-user redaction; mentions that no longer appear
verbatim in the (redacted) text are dropped.

Everything is fail-soft -- if GLiNER isn't installed or fails, resolution quietly
switches itself off (one warning) and the rest of the pipeline is unaffected.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from config import CFG
from entity_resolution.dedup import EntityCluster, Mention, cluster_mentions

_warned = False


def _warn_once(msg: str) -> None:
    global _warned
    if not _warned:
        _warned = True
        print(f"[entities] {msg}")


# ------------------------------------------------------------------ types ---
@dataclass
class ResolvedEntity:
    canonical: str
    label: str
    aliases: list[str] = field(default_factory=list)  # other surface forms
    sources: list[int] = field(default_factory=list)  # 1-based [Sx] indices
    in_query: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# -------------------------------------------------------------- extraction ---
def _extract(text: str) -> list[dict]:
    """GLiNER call that never raises."""
    if not CFG.entity_resolution or not text.strip():
        return []
    try:
        from entity_resolution.ner import extract_entities

        return extract_entities(text)
    except Exception as e:  # noqa: BLE001
        _warn_once(f"entity extraction unavailable, continuing without it: {e}")
        return []


def compact_entities(raw: list[dict]) -> list[dict]:
    """Keep what's worth storing: confident, sanely sized, de-duplicated, capped."""
    seen, out = set(), []
    for e in sorted(raw, key=lambda e: -e.get("score", 0)):
        text = (e.get("text") or "").strip()
        label = (e.get("label") or "").lower()
        if e.get("score", 0) < CFG.entity_min_score or not 2 <= len(text) <= 80:
            continue
        key = (text.lower(), label)
        if key in seen:
            continue
        seen.add(key)
        out.append({"text": text, "label": label, "score": round(float(e.get("score", 0)), 3)})
        if len(out) >= CFG.entity_max_per_chunk:
            break
    return out


def annotate_chunks(chunks: list) -> None:
    """Ingest time: attach extra['entities'] to every chunk. Never raises."""
    if not CFG.entity_resolution:
        return
    for c in chunks:
        c.extra["entities"] = compact_entities(_extract(c.text))


def query_mentions(query: str) -> list[Mention]:
    return [
        Mention(e["text"], e["label"], e["score"], source=None)
        for e in compact_entities(_extract(query))
    ]


def chunk_mentions(chunks: list, allow_live: bool = False) -> list[Mention]:
    """Mentions per chunk (Mention.source = 1-based position in `chunks`).
    Uses stored entities; chunks that have none stored at all (ingested before
    entity extraction was on) may be NER'd live, up to CFG.entity_live_fallback_max."""
    out: list[Mention] = []
    live_budget = CFG.entity_live_fallback_max if allow_live else 0
    for i, c in enumerate(chunks, start=1):
        stored = (c.extra or {}).get("entities")
        if stored is None:
            if live_budget <= 0:
                continue
            live_budget -= 1
            stored = compact_entities(_extract(c.text))  # c.text is already redacted
        for e in stored:
            if e.get("text") and e["text"] in c.text:  # drop anything redaction removed
                out.append(Mention(e["text"], e.get("label", ""), e.get("score", 1.0), source=i))
    return out


# ---------------------------------------------------------------- resolving ---
def _to_resolved(c: EntityCluster) -> ResolvedEntity:
    return ResolvedEntity(
        canonical=c.canonical,
        label=c.label,
        aliases=[m for m in c.members if m != c.canonical],
        sources=c.sources,
        in_query=c.in_query,
    )


def discover_aliases(query: str, q_mentions: list[Mention], candidates: list) -> list[str]:
    """RETRIEVE stage: other name forms, seen in the visible passages, for entities
    named in the question. Returns at most CFG.entity_max_aliases_in_query terms."""
    if not q_mentions or not candidates:
        return []
    clusters = cluster_mentions(q_mentions + chunk_mentions(candidates, allow_live=False))
    q_low = query.lower()
    aliases: list[str] = []
    for c in clusters:
        if not c.in_query:
            continue
        for m in c.members:
            if m.lower() not in q_low and m.lower() not in {a.lower() for a in aliases}:
                aliases.append(m)
    return aliases[: CFG.entity_max_aliases_in_query]


def resolve_sources(q_mentions: list[Mention], sources: list) -> list[ResolvedEntity]:
    """ANSWER stage: entity clusters across the final sources. Only 'notable' clusters
    are kept -- named in the question, having more than one name form, or appearing in
    more than one source."""
    if not CFG.entity_resolution:
        return []
    clusters = cluster_mentions(q_mentions + chunk_mentions(sources, allow_live=True))
    ents = [
        _to_resolved(c)
        for c in clusters
        if c.in_query or len(c.members) > 1 or len(c.sources) > 1
    ]
    ents.sort(key=lambda e: (not e.in_query, -len(e.sources), -len(e.aliases), e.canonical))
    return ents[:15]


# ------------------------------------------------------------ using results ---
def canonicalize(text: str, entities: list[ResolvedEntity]) -> str:
    """Rewrite every known alias in `text` to its entity's canonical name (single pass,
    longest alias first, so a short alias inside a long one is never double-replaced)."""
    mapping = {
        a.lower(): e.canonical for e in entities for a in e.aliases if a.lower() != e.canonical.lower()
    }
    if not mapping:
        return text
    alts = "|".join(re.escape(a) for a in sorted(mapping, key=len, reverse=True))
    pattern = re.compile(rf"(?<!\w)(?:{alts})(?!\w)", re.IGNORECASE)
    return pattern.sub(lambda m: mapping[m.group(0).lower()], text)


def entities_in(text: str, entities: list[ResolvedEntity]) -> set[str]:
    """Canonical names of the notable entities that occur in `text`."""
    low = text.lower()
    found = set()
    for e in entities:
        for form in [e.canonical, *e.aliases]:
            if re.search(rf"(?<!\w){re.escape(form.lower())}(?!\w)", low):
                found.add(e.canonical)
                break
    return found


def entity_notes(entities: list[ResolvedEntity], limit: int = 5) -> list[str]:
    """One line per entity that has several name forms (for the LLM prompt)."""
    notes = []
    for e in entities:
        if e.aliases:
            forms = ", ".join(f'"{a}"' for a in [e.canonical, *e.aliases])
            notes.append(f"{forms} refer to the same {e.label or 'entity'}.")
        if len(notes) >= limit:
            break
    return notes
