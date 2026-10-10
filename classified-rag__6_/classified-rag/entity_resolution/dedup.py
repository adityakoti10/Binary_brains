"""
Entity resolution: decide which surface forms ("J. Smith", "John Smith",
"Gen. Smith") are the same real-world entity.

Design rule: a WRONG merge is worse than a missed one -- merged entities feed
conflict detection and query expansion. So the matching is deliberately
conservative:
  * only mentions with the same NER label are ever merged;
  * dates and identifiers must match exactly (after normalisation);
  * any digit-bearing token must match exactly ("Server-14A" != "Server-14B",
    "Unit 12" != "Unit 21");
  * people: same surname AND compatible given names/initials; a bare surname
    joins a cluster only if exactly ONE cluster could own it (else it stays apart);
  * organisations additionally match acronyms ("DoD" <-> "Department of Defense").

RapidFuzz is used when installed; a difflib fallback keeps everything working
without it. For a labelled entity-resolution benchmark, swap `cluster_mentions`
for Zingg / Dedupe once you have labelled pairs -- callers only rely on
EntityCluster.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - exercised only when rapidfuzz is absent
    fuzz = None


def _token_sort_ratio(a: str, b: str) -> float:
    if fuzz is not None:
        return fuzz.token_sort_ratio(a, b)
    sa, sb = " ".join(sorted(a.split())), " ".join(sorted(b.split()))
    return 100.0 * difflib.SequenceMatcher(None, sa, sb).ratio()


# ------------------------------------------------------------------ types ---
@dataclass
class Mention:
    text: str
    label: str = ""
    score: float = 1.0
    source: int | None = None  # 1-based [Sx] index; None = the user's query


@dataclass
class EntityCluster:
    canonical: str
    members: list[str]
    label: str = ""
    sources: list[int] = field(default_factory=list)  # 1-based source indices
    in_query: bool = False


# ------------------------------------------------------------ normalising ---
_HONORIFICS = {"mr", "mrs", "ms", "miss", "dr", "prof", "sir", "gen", "col", "maj", "capt",
               "cpt", "lt", "ltc", "cdr", "adm", "sgt", "ssgt", "cpl", "pvt", "amb", "hon"}
_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "phd", "md"}
_ORG_NOISE = {"inc", "corp", "corporation", "ltd", "llc", "co", "company", "plc", "gmbh", "the"}
_ACRONYM_STOP = {"of", "the", "and", "for", "on", "in"}
_EXACT_ONLY_LABELS = {"date", "identifier", "document_reference"}


def normalize(text: str, label: str = "") -> str:
    t = unicodedata.normalize("NFKD", text)
    t = "".join(ch for ch in t if not unicodedata.combining(ch)).lower()
    t = re.sub(r"[./'’`]", "", t)          # "M/V" -> "mv", "J." -> "j", "O'Brien" -> "obrien"
    t = re.sub(r"[\W_]+", " ", t).strip()   # unicode-aware: keeps Cyrillic, CJK, ...
    tokens = t.split()
    if label == "person":
        kept = [x for x in tokens if x not in _HONORIFICS and x not in _NAME_SUFFIXES]
        tokens = kept or tokens
    elif label == "organization":
        kept = [x for x in tokens if x not in _ORG_NOISE]
        tokens = kept or tokens
    return " ".join(tokens)


def _digit_tokens(s: str) -> set[str]:
    return {t for t in s.split() if any(ch.isdigit() for ch in t)}


# ---------------------------------------------------------- compatibility ---
def _given_compatible(ga: list[str], gb: list[str]) -> bool:
    for x, y in zip(ga, gb):
        if x == y:
            continue
        if (len(x) == 1 and y.startswith(x)) or (len(y) == 1 and x.startswith(y)):
            continue
        return False
    return True


def _person_compatible(a: str, b: str) -> bool:
    ta, tb = a.split(), b.split()
    if not ta or not tb or ta[-1] != tb[-1]:  # surnames must match exactly
        return False
    ga, gb = ta[:-1], tb[:-1]
    if not ga or not gb:                       # a bare surname: caller enforces uniqueness
        return True
    return _given_compatible(ga, gb)


def _acronym_match(a: str, b: str) -> bool:
    short, long_ = (a, b) if len(a.split()) < len(b.split()) else (b, a)
    if len(short.split()) != 1 or not (2 <= len(short) <= 6) or len(long_.split()) < 2:
        return False
    words = long_.split()
    with_stop = "".join(w[0] for w in words)
    without = "".join(w[0] for w in words if w not in _ACRONYM_STOP)
    return short in (with_stop, without)


def compatible(a: str, b: str, label: str, threshold: float = 90.0) -> bool:
    """Could normalised forms `a` and `b` (same NER label) be the same entity?"""
    if a == b:
        return True
    if _digit_tokens(a) != _digit_tokens(b):
        return False
    if label in _EXACT_ONLY_LABELS:
        return False
    if label == "person":
        return _person_compatible(a, b)
    if label == "organization" and _acronym_match(a, b):
        return True
    return _token_sort_ratio(a, b) >= threshold


# ------------------------------------------------------------- clustering ---
def cluster_mentions(mentions: list[Mention], threshold: float = 90.0) -> list[EntityCluster]:
    """Group mentions into entities. Most specific (longest) forms are placed first so
    they become the cluster head that shorter forms are compared against."""
    by_label: dict[str, list[Mention]] = {}
    for m in mentions:
        if m.text and m.text.strip():
            by_label.setdefault(m.label.lower(), []).append(m)

    out: list[EntityCluster] = []
    for label, group in by_label.items():
        forms: dict[str, dict] = {}  # normalised form -> aggregate
        for m in group:
            key = normalize(m.text, label)
            if not key:
                continue
            f = forms.setdefault(key, {"surfaces": Counter(), "sources": set(), "in_query": False})
            f["surfaces"][m.text.strip()] += 1
            if m.source is None:
                f["in_query"] = True
            else:
                f["sources"].add(m.source)

        clusters: list[dict] = []  # {"head": key, "keys": [key,...]}
        for key in sorted(forms, key=lambda k: (-len(k.split()), -len(k), k)):
            owners = [c for c in clusters if compatible(key, c["head"], label, threshold)]
            if len(owners) == 1:
                owners[0]["keys"].append(key)
            else:  # no owner -> new entity; several possible owners -> ambiguous, keep apart
                clusters.append({"head": key, "keys": [key]})

        for c in clusters:
            head = forms[c["head"]]
            canonical = head["surfaces"].most_common(1)[0][0]
            members: list[str] = []
            for k in c["keys"]:
                for s, _ in forms[k]["surfaces"].most_common():
                    if s.lower() not in {x.lower() for x in members}:
                        members.append(s)
            out.append(
                EntityCluster(
                    canonical=canonical,
                    members=members,
                    label=label,
                    sources=sorted(set().union(*(forms[k]["sources"] for k in c["keys"]))),
                    in_query=any(forms[k]["in_query"] for k in c["keys"]),
                )
            )
    return out


# ------------------------------------------------- legacy string-only API ---
def cluster_entities(mentions: list[str], threshold: float = 88.0) -> list[EntityCluster]:
    """Original zero-training baseline (kept for existing callers): greedy fuzzy
    clustering of bare strings. Prefer cluster_mentions()."""
    clusters: list[EntityCluster] = []
    for m in mentions:
        best_i, best_score = -1, -1.0
        for i, c in enumerate(clusters):
            s = _token_sort_ratio(m, c.canonical)
            if s > best_score:
                best_i, best_score = i, s
        if best_i >= 0 and best_score >= threshold:
            clusters[best_i].members.append(m)
        else:
            clusters.append(EntityCluster(canonical=m, members=[m]))
    return clusters
