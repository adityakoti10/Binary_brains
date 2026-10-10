"""
Redaction *inside* documents.  Two independent layers, both enforced on the
server before any text reaches the reranker or the LLM:

1. PORTION MARKINGS (ingest time, opt-in per upload)
   Government-style paragraph marks -- "(U) ...", "(C) ...", "(S) ...",
   "(TS//NF) ..." -- are parsed when an upload is flagged `portion_marked`.
   The text is split at the marks so every chunk carries ONE classification
   (the level of the portions inside it).  The existing SQL filter in
   vectorstore/pgvector_store.search() then keeps a SECRET-cleared user away
   from the (TS) chunks of the same file while still letting them read its
   (U)/(S) chunks.
     * Unmarked paragraphs inherit the document's label   (fail closed).
     * A mark HIGHER than the document label wins         (fail closed).
     * Marks are only honoured if the uploader says the file is portion
       marked -- a stray "(U)" in an ordinary SECRET file must never
       downgrade text.

2. RULE-BASED REDACTION (query time, per user)
   Sensitive spans (SSNs, codewords, names, coordinates ...) inside chunks the
   user IS allowed to read are replaced by [REDACTED:<RULE>] unless the user's
   clearance is at least the rule's `min_clearance` (or the user is on the
   rule's `allow_users` need-to-know list).  Rules are defined in a JSON file
   (CFG.redaction_rules_file, see redaction_rules.example.json).

   Fail-closed behaviour:
     * a rule with no `min_clearance` is masked for everybody except `allow_users`;
     * a malformed rules file raises RedactionConfigError (the query fails
       rather than answering unredacted);
     * built-in rules (SSN, payment card) always apply unless a file rule with
       the same name overrides them.

Check your rules from the command line:

    python -m security.redaction "SSN 123-45-6789, call NIGHTHAWK" SECRET
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, NamedTuple

from config import CFG, clearance_rank


class RedactionConfigError(RuntimeError):
    """The redaction rules could not be loaded. Callers must NOT continue unredacted."""


# =============================================================== portions ===
@dataclass
class Portion:
    text: str
    level: str
    mark: str | None = None          # raw mark as written, e.g. "TS"; None = unmarked
    exceeds_label: bool = False      # mark was higher than the document label


# optional markdown prefix (heading / bullet / quote / "1."), then "(XX)" or "(XX//CAVEATS)"
_MARK_RE = re.compile(
    r"^(?P<prefix>[ \t]*(?:#{1,6}[ \t]+|[-*+][ \t]+|>[ \t]*|\d+[.)][ \t]+)?)"
    r"\((?P<mark>[A-Z]{1,2})(?://[^)\n]{0,40})?\)[ \t]*"
)


def split_portions(text: str, doc_level: str) -> list[Portion]:
    """Split `text` into paragraphs and classify each by its leading portion mark.
    The mark itself is stripped from the stored text; it is kept in Portion.mark."""
    doc_rank = clearance_rank(doc_level)  # raises ValueError on an unknown label
    portions: list[Portion] = []
    for para in re.split(r"\n\s*\n", text):
        if not para.strip():
            continue
        m = _MARK_RE.match(para)
        level_for_mark = CFG.portion_marks.get(m.group("mark")) if m else None
        if m and level_for_mark:
            # The mark's level is used even when it is HIGHER than the document label
            # (fail closed: more restrictive); such portions are flagged.
            level = level_for_mark
            exceeds = clearance_rank(level) > doc_rank
            body = m.group("prefix") + para[m.end():]
            portions.append(Portion(body.strip(), level, m.group("mark"), exceeds))
        else:
            portions.append(Portion(para.strip(), doc_level, None))
    return portions


def group_portions(portions: list[Portion]) -> list[Portion]:
    """Merge consecutive portions that share a level, so chunking works on runs of
    equally-classified text.  A run is 'exceeds_label' if any part of it was."""
    runs: list[Portion] = []
    for p in portions:
        if runs and runs[-1].level == p.level:
            runs[-1].text += "\n\n" + p.text
            runs[-1].exceeds_label = runs[-1].exceeds_label or p.exceeds_label
        else:
            runs.append(Portion(p.text, p.level, p.mark, p.exceeds_label))
    return runs


# ================================================================== rules ===
@dataclass(frozen=True)
class Rule:
    name: str
    regex: re.Pattern
    min_rank: int | None                     # None = never visible (except allow_users)
    allow_users: frozenset = frozenset()
    validator: Callable[[str], bool] | None = None


def _luhn_ok(s: str) -> bool:
    digits = [int(c) for c in s if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


def _builtin_rules() -> list[Rule]:
    return [
        Rule("SSN", re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"), None),
        Rule("CARD", re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), None, validator=_luhn_ok),
    ]


def _parse_rule(spec: dict, i: int) -> Rule:
    where = f"rule #{i + 1}"
    try:
        name = str(spec["name"]).strip().upper()
        if not name or not re.fullmatch(r"[A-Z0-9_ -]{1,40}", name):
            raise ValueError("name must be 1-40 chars of A-Z 0-9 _ - space")
        has_regex, has_terms = "regex" in spec, "terms" in spec
        if has_regex == has_terms:
            raise ValueError("give exactly one of 'regex' or 'terms'")
        flags = re.IGNORECASE if spec.get("ignore_case", has_terms) else 0
        if has_regex:
            pattern = re.compile(spec["regex"], flags)
        else:
            terms = [str(t).strip() for t in spec["terms"] if str(t).strip()]
            if not terms:
                raise ValueError("'terms' is empty")
            alts = "|".join(
                r"\s+".join(re.escape(w) for w in t.split()) for t in sorted(terms, key=len, reverse=True)
            )
            pattern = re.compile(rf"(?<!\w)(?:{alts})(?!\w)", flags)
        mc = spec.get("min_clearance")
        min_rank = None if mc in (None, "", "NEVER") else clearance_rank(str(mc).upper())
        allow = frozenset(str(u) for u in spec.get("allow_users", []))
        return Rule(name, pattern, min_rank, allow)
    except (KeyError, ValueError, re.error, TypeError) as e:
        raise RedactionConfigError(f"redaction {where}: {e}") from e


_cache: dict = {"key": None, "rules": None}


def load_rules() -> list[Rule]:
    """Built-ins + rules file. Re-read automatically when the file changes."""
    path = Path(CFG.redaction_rules_file)
    key = (str(path), path.stat().st_mtime_ns if path.exists() else None)
    if _cache["key"] == key and _cache["rules"] is not None:
        return _cache["rules"]
    rules = {r.name: r for r in _builtin_rules()}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            specs = data["rules"] if isinstance(data, dict) else data
            if not isinstance(specs, list):
                raise ValueError("'rules' must be a list")
        except (OSError, ValueError, KeyError) as e:
            raise RedactionConfigError(f"cannot read {path}: {e}") from e
        for i, spec in enumerate(specs):
            r = _parse_rule(spec, i)
            rules[r.name] = r  # a file rule overrides a built-in of the same name
    _cache.update(key=key, rules=list(rules.values()))
    return _cache["rules"]


# ============================================================== redacting ===
class RedactionResult(NamedTuple):
    text: str
    count: int
    kinds: list


def _visible(rule: Rule, rank: int, user_id: str) -> bool:
    if user_id and user_id in rule.allow_users:
        return True
    return rule.min_rank is not None and rank >= rule.min_rank


def redact_text(text: str, max_clearance: str, user_id: str = "") -> RedactionResult:
    """Mask every span this user may not see. All rules are matched against the
    ORIGINAL text and merged, so one rule can never re-match another's marker."""
    if not CFG.redaction_enabled or not text:
        return RedactionResult(text, 0, [])
    rank = clearance_rank(max_clearance)

    spans: list[tuple[int, int, str]] = []
    for rule in load_rules():
        if _visible(rule, rank, user_id):
            continue
        for m in rule.regex.finditer(text):
            if m.end() == m.start():
                continue
            if rule.validator and not rule.validator(m.group(0)):
                continue
            spans.append((m.start(), m.end(), rule.name))
    if not spans:
        return RedactionResult(text, 0, [])

    spans.sort(key=lambda s: (s[0], -s[1]))
    merged: list[list] = []  # [start, end, [names]]
    for s, e, name in spans:
        if merged and s < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
            if name not in merged[-1][2]:
                merged[-1][2].append(name)
        else:
            merged.append([s, e, [name]])

    out, pos, kinds = [], 0, []
    for s, e, names in merged:
        out.append(text[pos:s])
        out.append(f"[REDACTED:{names[0]}]")
        pos = e
        kinds.extend(n for n in names if n not in kinds)
    out.append(text[pos:])
    return RedactionResult("".join(out), len(merged), kinds)


def redact_chunks(chunks: list, ctx) -> tuple[list, int]:
    """Return redacted COPIES of retrieved chunks (the originals are untouched) and
    the total number of redactions.  This is the single enforcement point for the
    query path: retrieval/retriever.py calls it before rerank/synthesis.

    Stored entity mentions (extra['entities']) are re-validated against the redacted
    text so a masked name can't leak back out through entity resolution."""
    out, total = [], 0
    for c in chunks:
        res = redact_text(c.text, ctx.max_clearance, ctx.user_id)
        if res.count == 0:
            out.append(c)
            continue
        total += res.count
        extra = dict(c.extra or {})
        extra["redactions"] = res.count
        extra["redacted_kinds"] = res.kinds
        if extra.get("entities"):
            extra["entities"] = [e for e in extra["entities"] if e.get("text", "") in res.text]
        out.append(replace(c, text=res.text, extra=extra))
    return out, total


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit('usage: python -m security.redaction "text to test" <CLEARANCE>')
    r = redact_text(sys.argv[1], sys.argv[2].upper())
    print(r.text)
    print(f"[{r.count} redaction(s): {', '.join(r.kinds) or 'none'}]")
