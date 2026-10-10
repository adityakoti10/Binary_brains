"""
Three ready-to-train typed decisions, built on FastClassifier:

  1. doc_type   -- Choice: manual / report / conversation_log / system_log
  2. urgency    -- Score:  routine / priority / urgent
  3. log_severity -- Choice: info / warning / error / critical

These run on every ingested chunk in a fraction of a second on CPU, so you
get useful triage metadata (stored in Chunk.extra) WITHOUT ever calling the
70B synthesis LLM for it. Expand the example lists below with your own real
data before relying on this -- 10-20 examples per label is the realistic
minimum for SetFit to learn anything useful; the seed examples here are just
enough to get the pipeline running end to end.
"""
from __future__ import annotations

from config import CFG
from routing.fast_classifier import Decision, FastClassifier

# --------------------------------------------------------------------- seed training data ---
# Replace/extend these with real labeled snippets from your own corpus.

_DOC_TYPE_EXAMPLES = {
    "manual": [
        "Step 3: Remove the access panel using a Phillips screwdriver before servicing the unit.",
        "This chapter describes installation procedures for the cooling subsystem.",
        "Warning: disconnect power before performing maintenance on this component.",
        "Table 4.2 lists the torque specifications for all fasteners on the assembly.",
    ],
    "report": [
        "Executive summary: Q3 performance declined 4% relative to forecast due to supply delays.",
        "The investigation concluded that the failure was caused by a defective sensor batch.",
        "This report presents findings from the site inspection conducted on the 14th.",
        "Recommendations: increase inspection frequency and revise the maintenance schedule.",
    ],
    "conversation_log": [
        "User: can you check the status of ticket 4471? Agent: looking into it now.",
        "10:32 - Analyst: confirmed the anomaly, escalating to on-call. 10:34 - Oncall: acknowledged.",
        "Caller stated the device had been unresponsive since Tuesday morning.",
        "Chat transcript: 'is the server back up yet' / 'yes, restored at 14:02'",
    ],
    "system_log": [
        "2026-09-12T03:14:02Z ERROR db-connector timeout after 30000ms retries=3",
        "WARN [auth-service] failed login attempt from 10.0.4.12 user=admin",
        "INFO service restarted pid=8821 uptime_reset=true",
        "CRITICAL disk usage at 97% on /var/log, rotation triggered",
    ],
}

_URGENCY_EXAMPLES = {
    "routine": [
        "Please review this document at your convenience before next month's audit.",
        "Routine maintenance completed as scheduled, no issues found.",
        "Quarterly summary attached for your records.",
    ],
    "priority": [
        "This needs review before Friday's release -- please prioritize.",
        "Escalating: customer impact confirmed, needs a fix this week.",
        "Please respond within 24 hours, this is affecting the pilot team.",
    ],
    "urgent": [
        "URGENT: production system down, all hands needed immediately.",
        "Critical failure detected, requires immediate attention -- do not wait for morning review.",
        "Emergency: safety incident reported, escalate now.",
    ],
}

_LOG_SEVERITY_EXAMPLES = {
    "info": [
        "INFO service started successfully on port 8080",
        "INFO scheduled backup completed",
        "INFO user session ended normally",
    ],
    "warning": [
        "WARN memory usage above 80% threshold",
        "WARN deprecated API endpoint called, migrate before v3",
        "WARN retrying failed request, attempt 2 of 3",
    ],
    "error": [
        "ERROR failed to connect to database after 3 retries",
        "ERROR null pointer exception in request handler",
        "ERROR authentication failed for user token",
    ],
    "critical": [
        "CRITICAL disk full, service cannot write logs",
        "CRITICAL out of memory, process killed",
        "CRITICAL security breach detected, lockdown initiated",
    ],
}


def _flatten(examples: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    for label, snippets in examples.items():
        texts.extend(snippets)
        labels.extend([label] * len(snippets))
    return texts, labels


# --------------------------------------------------------------------- public API ---
doc_type_classifier = FastClassifier("doc_type")
urgency_classifier = FastClassifier("urgency")
log_severity_classifier = FastClassifier("log_severity")


def train_all() -> None:
    """Run this ONCE to train and save all three classifiers locally.
    Re-run any time you add more labeled examples above."""
    for clf, examples in [
        (doc_type_classifier, _DOC_TYPE_EXAMPLES),
        (urgency_classifier, _URGENCY_EXAMPLES),
        (log_severity_classifier, _LOG_SEVERITY_EXAMPLES),
    ]:
        texts, labels = _flatten(examples)
        print(f"[routing] training '{clf.name}' on {len(texts)} examples...")
        clf.train(texts, labels)
    print("[routing] done. Models saved under routing/trained_models/")


def _classify_setfit(text: str, is_log: bool = False) -> dict:
    """SetFit backend: cheap, local, trained on the seed examples above."""
    out: dict[str, Decision | None] = {
        "doc_type": doc_type_classifier.predict(text),
        "urgency": urgency_classifier.predict(text),
        "log_severity": log_severity_classifier.predict(text) if is_log else None,
    }
    return {
        "triage_backend": "setfit",
        "doc_type": out["doc_type"].label,
        "doc_type_confidence": round(out["doc_type"].confidence, 3),
        "urgency": out["urgency"].label,
        "urgency_confidence": round(out["urgency"].confidence, 3),
        "log_severity": out["log_severity"].label if out["log_severity"] else None,
    }


_warned: set[str] = set()


def classify_chunk(text: str, is_log: bool = False) -> dict:
    """Called once per chunk during ingestion. Returns triage metadata to
    merge into Chunk.extra. Backend chosen by CFG.triage_backend
    ("setfit" | "laya" | "off"). If the backend isn't ready (e.g. classifiers
    not trained yet) ingestion carries on without triage tags instead of
    failing -- a one-time warning is printed."""
    backend = CFG.triage_backend.lower()
    if backend == "off":
        return {}
    try:
        if backend == "laya":
            from routing.laya_router import classify_chunk_laya

            return classify_chunk_laya(text, is_log)
        return _classify_setfit(text, is_log)
    except Exception as e:  # noqa: BLE001
        if backend not in _warned:
            _warned.add(backend)
            print(f"[routing] triage skipped, backend '{backend}' not ready: {e}")
        return {}


if __name__ == "__main__":
    train_all()
    print(classify_chunk("CRITICAL disk usage at 98%, immediate action required", is_log=True))
    print(classify_chunk("Step 5: torque the bolts to spec before reassembly."))
