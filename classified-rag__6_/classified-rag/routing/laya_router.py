"""
Laya backend for chunk triage (optional; enable with TRIAGE_BACKEND=laya).

Laya (Apache-2.0, ModernBERT-based) answers typed questions -- choice / score /
noul -- in one forward pass with no text generation. Same triage labels as the
SetFit backend, so the rest of the pipeline doesn't care which one is used.

HONEST CAVEATS (from Laya's own docs):
  * The base checkpoints are near chance zero-shot. The "typed-decisions"
    checkpoint is fine-tuned, but on business workflows (invoices, security
    incidents, support tickets), NOT on your manuals/logs. Measure it on a
    labeled sample of YOUR chunks before trusting it, and fine-tune with
    Laya's notebook if it falls short.
  * English checkpoints read ~512-1024 tokens, so text is truncated below.
  * Keep choice questions to <= ~20 options (we use 4).
"""
from __future__ import annotations

from functools import lru_cache

from config import CFG

MAX_CHARS = 1200  # keep well inside Laya's context window

_URGENCY_LEVELS = ["routine", "priority", "urgent"]

_BASE_QUESTIONS = {
    "doc_type": {
        "type": "choice",
        "instructions": "What kind of document excerpt is this?",
        "criteria": {
            "manual": "technical manual, procedure or installation instructions",
            "report": "analysis, findings or status report",
            "conversation_log": "chat, call or email conversation between people",
            "system_log": "machine-generated system log lines",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this text?",
        "criteria": [
            "routine, no time pressure",
            "priority, needs attention soon",
            "urgent, immediate action required",
        ],
    },
}

_LOG_QUESTION = {
    "log_severity": {
        "type": "choice",
        "instructions": "What is the severity of these log lines?",
        "criteria": {
            "info": "normal operation messages",
            "warning": "something may go wrong soon",
            "error": "an operation failed",
            "critical": "system-threatening failure or security breach",
        },
    }
}


@lru_cache(maxsize=1)
def _router():
    from laya import Router

    return Router(device=CFG.small_models_device)  # checkpoints load lazily


def classify_chunk_laya(text: str, is_log: bool = False) -> dict:
    questions = dict(_BASE_QUESTIONS)
    if is_log:
        questions.update(_LOG_QUESTION)

    result = _router().predict({"body": text[:MAX_CHARS]}, questions, model=CFG.laya_checkpoint)
    answers = result["answers"]

    doc = answers["doc_type"]
    urg = answers["urgency"]
    level = int(round(float(urg["score"])))
    level = max(0, min(level, len(_URGENCY_LEVELS) - 1))

    out = {
        "triage_backend": "laya",
        "doc_type": doc["choice"],
        "doc_type_confidence": round(float(doc.get("confidence", 0.0)), 3),
        "urgency": _URGENCY_LEVELS[level],
        "urgency_confidence": round(float(urg.get("confidence", 0.0)), 3),
        "log_severity": None,
    }
    confs = [out["doc_type_confidence"], out["urgency_confidence"]]

    if is_log and "log_severity" in answers:
        sev = answers["log_severity"]
        out["log_severity"] = sev["choice"]
        confs.append(round(float(sev.get("confidence", 0.0)), 3))

    out["triage_needs_review"] = min(confs) < CFG.laya_min_confidence
    return out
