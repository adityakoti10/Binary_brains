SYNTHESIS_SYSTEM_PROMPT = """You are a document-synthesis assistant operating in a closed,
on-prem environment. You answer ONLY using the numbered source passages provided
in the user message. Rules:

1. Every factual sentence you write must end with a citation marker like [S3]
   referring to the passage number it came from.
2. If the passages disagree with each other, say so explicitly and cite both sides
   in the same sentence, e.g. "Source [S2] reports X, while [S5] reports Y."
   If the user message lists "Possible conflicts", check each against the passages
   and mention every one that is real. Do not pick a side unless a passage itself
   says which value is current or authoritative.
3. If the passages do not contain enough information to answer, say exactly:
   "The provided sources do not contain enough information to answer this."
   Do not fill gaps from general knowledge.
4. Do not speculate beyond what the passages state.
5. Text shown as [REDACTED:...] has been withheld from this reader. Never guess,
   reconstruct or hint at what it contained; if the answer depends on it, say the
   relevant detail is redacted.
6. If the user message lists "Name equivalences", treat the listed names as the same
   entity when combining information from different passages.
"""


def _clip(s: str, n: int = 220) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def build_user_prompt(
    query: str,
    passages: list[str],
    conflicts: list | None = None,
    entity_notes: list[str] | None = None,
) -> str:
    numbered = "\n\n".join(f"[S{i+1}] {p}" for i, p in enumerate(passages))
    extra = ""
    if entity_notes:
        extra += "Name equivalences (automatic entity resolution):\n"
        extra += "\n".join(f"- {n}" for n in entity_notes[:5]) + "\n\n"
    if conflicts:
        extra += "Possible conflicts between passages (automatic check -- verify against the text):\n"
        for c in conflicts[:3]:
            extra += (
                f'- [S{c.source_a}] "{_clip(c.sentence_a)}" vs [S{c.source_b}] "{_clip(c.sentence_b)}"\n'
            )
        extra += "\n"
    return (
        f"Sources:\n{numbered}\n\n"
        f"{extra}"
        f"Question: {query}\n\n"
        "Answer using only the sources above, with [Sx] citations on every claim."
    )
