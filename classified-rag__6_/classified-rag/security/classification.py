"""
Classification/clearance resolution. Deliberately tiny and boring -- this
is the module an auditor should be able to read in 30 seconds and trust.

The clearance ALWAYS comes from the server-side user table (security/users.py)
keyed by the caller's token -- never from anything the client sends about
itself. To move to a real identity system (LDAP, SSO, OPA policy engine),
replace `resolve_user_clearance` and keep its return type; nothing else in the
pipeline computes a clearance level.
"""
from __future__ import annotations

from dataclasses import dataclass

from config import CFG, clearance_rank
from security import users


@dataclass
class RequestContext:
    user_id: str
    max_clearance: str
    allowed_doc_ids: list[str] | None = None  # extra per-document ACL, if used


def resolve_user_clearance(auth_token: str) -> RequestContext:
    """Look the token up in the server-side user table.
    Raises PermissionError for a missing/unknown token or a corrupt entry."""
    entry = users.lookup(auth_token)
    if not entry:
        raise PermissionError("invalid or missing token")
    try:
        clearance_rank(entry["clearance"])
    except (ValueError, KeyError) as e:
        raise PermissionError("user entry has an invalid clearance level") from e
    return RequestContext(user_id=entry["user_id"], max_clearance=entry["clearance"])


def assert_valid_level(level: str) -> None:
    clearance_rank(level)  # raises ValueError on an unknown level


def levels_up_to(max_clearance: str) -> list[str]:
    """All levels a user with `max_clearance` may read (and label uploads with)."""
    return CFG.classification_levels[: clearance_rank(max_clearance) + 1]
