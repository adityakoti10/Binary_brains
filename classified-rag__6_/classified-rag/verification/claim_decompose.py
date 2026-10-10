"""
Breaks a synthesized answer into atomic claims tied to the [Sx] citation
they came from, so each claim can be checked independently against ONLY
the source it cited (not the whole retrieved set -- that would let a claim
"pass" by matching an unrelated passage).
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Claim:
    text: str
    cited_indices: list[int]  # 1-based indices into the source list, e.g. [3] for [S3]


_CITATION_RE = re.compile(r"\[S(\d+)\]")

# A "sentence" that ends in an initial or abbreviation ("J.", "Dr.", "U.S.") is not
# finished -- otherwise "J. Smith reported X" is cut into "J." + "Smith reported X".
_TITLES = ("Dr|Mr|Mrs|Ms|Gen|Col|Lt|Cdr|Capt|Maj|Sgt|Prof|Sr|Jr|St|No|Nos|vs|approx|Fig|Figs|Eq|Inc|Corp|Ltd|Co")
_LEAD = r"(?:^|[\s(\[\"'])"
_ABBREV_END = re.compile(
    rf"{_LEAD}(?:[A-Z]|{_TITLES})\.$|{_LEAD}(?:[A-Za-z]\.){{2,}}$"
)


def split_sentences(text: str) -> list[str]:
    """Split on . ! ? followed by whitespace, without breaking after initials/abbreviations."""
    out: list[str] = []
    for part in re.split(r"(?<=[.!?])\s+", text.strip()):
        if out and _ABBREV_END.search(out[-1]):
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return out


def decompose(answer: str) -> list[Claim]:
    """Sentence-level split + citation extraction. This is a pragmatic
    heuristic; for higher recall swap in an LLM-based claim decomposer
    (ask the synthesis model itself, or a dedicated model like MiniCheck's
    claim splitter) and keep this as the cheap default path."""
    claims = []
    for s in split_sentences(answer):
        if not s.strip():
            continue
        cited = [int(m) for m in _CITATION_RE.findall(s)]
        claims.append(Claim(text=_CITATION_RE.sub("", s).strip(), cited_indices=cited))
    return claims
