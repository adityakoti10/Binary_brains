"""
Hashing/signing + an append-only audit trail entry for every answer produced.
Uses HMAC-SHA256 for integrity (swap in a real PKI/detached digital
signature -- e.g. cryptography's Ed25519 -- for production non-repudiation).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import asdict, dataclass

_AUDIT_KEY = os.getenv("AUDIT_HMAC_KEY", "CHANGE-ME-load-from-a-secrets-manager").encode()


@dataclass
class AuditRecord:
    query: str
    answer: str
    max_clearance_used: str
    cited_doc_ids: list[str]
    hallucination_rate: float
    unsupported_claim_count: int
    contradicted_claim_count: int
    timestamp: float
    user_id: str = ""
    redaction_count: int = 0   # spans masked inside the sources shown to this user
    conflict_count: int = 0    # cross-source conflicts detected between the sources
    signature: str = ""

    # Fields added after the first release. They are left OUT of the signed payload
    # while they hold their default value, so log lines written before they existed
    # still verify; a non-default value is always covered by the signature.
    _LATER_FIELDS = {"redaction_count": 0, "conflict_count": 0}

    def _payload(self) -> bytes:
        d = asdict(self) | {"signature": ""}
        for k, default in self._LATER_FIELDS.items():
            if d.get(k) == default:
                d.pop(k, None)
        return json.dumps(d, sort_keys=True).encode()

    def sign(self) -> "AuditRecord":
        self.signature = hmac.new(_AUDIT_KEY, self._payload(), hashlib.sha256).hexdigest()
        return self

    def verify(self) -> bool:
        expected = hmac.new(_AUDIT_KEY, self._payload(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(self.signature, expected)


def write_audit_log(record: AuditRecord, path: str = "audit_log.jsonl") -> None:
    record.timestamp = record.timestamp or time.time()
    record.sign()
    with open(path, "a") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def write_event(kind: str, payload: dict, path: str = "events_log.jsonl") -> None:
    """Signed append-only log for non-answer events (e.g. document uploads)."""
    body = {"kind": kind, "timestamp": time.time(), **payload}
    body["signature"] = hmac.new(
        _AUDIT_KEY, json.dumps(body, sort_keys=True).encode(), hashlib.sha256
    ).hexdigest()
    with open(path, "a") as f:
        f.write(json.dumps(body) + "\n")
