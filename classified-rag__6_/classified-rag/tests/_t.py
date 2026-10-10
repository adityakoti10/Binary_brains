"""Tiny shared helpers for the standalone test scripts (no pytest needed)."""
import os
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_failed = 0


def check(name, cond):
    global _failed
    print(("PASS  " if cond else "FAIL  ") + name)
    if not cond:
        _failed += 1


def finish():
    print(f"\n{'ALL PASSED' if not _failed else str(_failed) + ' FAILED'}")
    sys.exit(1 if _failed else 0)


def scratch_env(**env):
    """Point config at a throw-away directory BEFORE `config` is imported."""
    tmp = tempfile.mkdtemp()
    os.chdir(tmp)
    os.environ.update({"USERS_FILE": os.path.join(tmp, "users.json"),
                       "REDACTION_RULES_FILE": os.path.join(tmp, "redaction_rules.json"),
                       "AUDIT_HMAC_KEY": "test-key", **env})
    return tmp


def stub_db_drivers():
    """Let vectorstore.pgvector_store import without psycopg2/pgvector installed."""
    for name in ("psycopg2", "psycopg2.extras", "pgvector", "pgvector.psycopg2"):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.register_vector = lambda conn: None
            m.connect = lambda *a, **k: None
            sys.modules[name] = m
    sys.modules["psycopg2"].extras = sys.modules["psycopg2.extras"]
