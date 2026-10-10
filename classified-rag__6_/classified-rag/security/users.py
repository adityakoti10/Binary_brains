"""
Minimal local user table for the web interface: API token -> (user, clearance).

Only a SHA-256 hash of each token is stored (in USERS_FILE, default users.json),
so the file alone can't be used to log in. Tokens are shown ONCE when created.

    python -m security.users add alice SECRET
    python -m security.users list

This is fine for a single-machine demo / small team. For a real deployment,
replace security.classification.resolve_user_clearance with a call to your
identity provider / ABAC policy engine -- nothing else needs to change.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sys
from pathlib import Path

from config import CFG, clearance_rank


def _path() -> Path:
    return Path(CFG.users_file)


def _load() -> dict:
    p = _path()
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _save(data: dict) -> None:
    p = _path()
    p.write_text(json.dumps(data, indent=2))
    try:
        os.chmod(p, 0o600)  # best effort; a no-op on some Windows setups
    except OSError:
        pass


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def add_user(user_id: str, clearance: str) -> str:
    clearance_rank(clearance)  # raises ValueError on an unknown level
    data = _load()
    token = secrets.token_urlsafe(32)
    data[hash_token(token)] = {"user_id": user_id, "clearance": clearance}
    _save(data)
    return token


def lookup(token: str) -> dict | None:
    if not token:
        return None
    return _load().get(hash_token(token))


def list_users() -> list[dict]:
    return list(_load().values())


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 3 and args[0] == "add":
        tok = add_user(args[1], args[2].upper())
        print(f"User '{args[1]}' created with clearance {args[2].upper()}.")
        print(f"Login token (shown once, store it safely):\n\n  {tok}\n")
    elif args == ["list"]:
        for u in list_users():
            print(f"{u['user_id']:<20} {u['clearance']}")
    else:
        print("usage: python -m security.users add <user_id> <CLEARANCE>\n"
              "       python -m security.users list")
        print("levels:", ", ".join(CFG.classification_levels))
        sys.exit(1)
