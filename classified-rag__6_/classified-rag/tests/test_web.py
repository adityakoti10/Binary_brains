"""
Smoke tests for the web layer with the heavy pipeline stubbed out (no GPU,
databases or model downloads needed):

    python tests/test_web.py

Checks: auth, server-side clearance, upload-level enforcement, filename
sanitising, file-type + size limits, and the response shapes the UI relies on.
"""
import os
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

tmp = tempfile.mkdtemp()
os.environ["USERS_FILE"] = os.path.join(tmp, "users.json")
os.environ["UPLOAD_DIR"] = os.path.join(tmp, "uploads")
os.environ["MAX_UPLOAD_MB"] = "1"
os.chdir(tmp)  # audit/event logs land here, not in the project

# --- stub the heavy pipeline module -----------------------------------------
calls = {}
fake = types.ModuleType("pipeline")


class _R:
    def to_dict(self):
        return {
            "answer": "The interval is 500 hours [S1].", "hallucination_rate": 0.0,
            "supported_claims": 1, "unsupported_claims": 0, "contradicted_claims": 0,
            "cited_doc_ids": ["abc"],
            "sources": [{"index": 1, "doc_id": "abc", "source_path": "m.pdf", "page": 3,
                         "classification": "SECRET", "score": 0.9, "snippet": "..."}],
            "claims": [{"text": "The interval is 500 hours.", "cited": [1],
                        "status": "supported", "entailment": 0.97}],
            "conflicts": [], "entities": [], "query_expansion": [],
            "redaction_count": 0, "conflict_check": "ok",
        }


def _answer_query(q, ctx):
    calls["ctx"] = ctx
    return _R()


def _ingest_files(paths, level, portion_marked=False):
    calls["ingest"] = ([Path(p).name for p in paths], level)
    calls["portion_marked"] = portion_marked
    return 7


fake.answer_query, fake.ingest_files = _answer_query, _ingest_files
sys.modules["pipeline"] = fake

from fastapi.testclient import TestClient  # noqa: E402

from security import users  # noqa: E402
from web.app import app  # noqa: E402

client = TestClient(app)
secret_tok = users.add_user("alice", "SECRET")
uncl_tok = users.add_user("bob", "UNCLASSIFIED")
H = lambda t: {"Authorization": f"Bearer {t}"}  # noqa: E731


def check(name, cond):
    print(("PASS  " if cond else "FAIL  ") + name)
    if not cond:
        check.failed = True


check.failed = False

# auth
check("no token -> 401", client.get("/api/me").status_code == 401)
check("bad token -> 401", client.get("/api/me", headers=H("nope")).status_code == 401)
r = client.get("/api/me", headers=H(secret_tok)).json()
check("me returns clearance + levels up to it", r["clearance"] == "SECRET" and r["upload_levels"] == ["UNCLASSIFIED", "CONFIDENTIAL", "SECRET"])
check("low-clearance user sees only own level", client.get("/api/me", headers=H(uncl_tok)).json()["upload_levels"] == ["UNCLASSIFIED"])
check("token hash only in users file", secret_tok not in Path(os.environ["USERS_FILE"]).read_text())

# ask: clearance comes from the server, not the client
resp = client.post("/api/ask", json={"query": "interval?", "clearance": "TOP_SECRET"}, headers=H(uncl_tok))
check("ask ok + shape", resp.status_code == 200 and resp.json()["sources"][0]["classification"] == "SECRET")
check("client-supplied clearance ignored", calls["ctx"].max_clearance == "UNCLASSIFIED")
check("empty question rejected", client.post("/api/ask", json={"query": "  "}, headers=H(secret_tok)).status_code == 400)
check("ask needs auth", client.post("/api/ask", json={"query": "x"}).status_code == 401)

# upload
def up(tok, level, name="m.pdf", data=b"x", ctype="application/pdf", **extra):
    return client.post("/api/upload", headers=H(tok), data={"classification": level, **extra},
                       files=[("files", (name, data, ctype))])

check("cannot label above clearance", up(uncl_tok, "SECRET").status_code == 403)
check("unknown level rejected", up(secret_tok, "BOGUS").status_code == 403)
ok = up(secret_tok, "CONFIDENTIAL")
check("upload ok", ok.status_code == 200 and ok.json()["chunks_stored"] == 7)
check("ingest got the chosen level", calls["ingest"][1] == "CONFIDENTIAL")
evil = up(secret_tok, "SECRET", name="../../etc/passwd.pdf")
saved = calls["ingest"][0][0]
check("path traversal neutralised", evil.status_code == 200 and "/" not in saved and ".." not in saved)
check("disallowed file type rejected", up(secret_tok, "SECRET", name="run.exe").status_code == 400)
check("oversize rejected", up(secret_tok, "SECRET", data=b"0" * (2 * 1024 * 1024)).status_code == 413)
leftovers = [p for p in Path(os.environ["UPLOAD_DIR"]).rglob("*") if p.is_file() and p.stat().st_size > 1024 * 1024]
check("oversize file not left on disk", not leftovers)
check("upload events are logged", Path("events_log.jsonl").exists() and "upload" in Path("events_log.jsonl").read_text())

# portion markings
up(secret_tok, "SECRET")
check("portion_marked defaults to off", calls["portion_marked"] is False)
pm = up(secret_tok, "SECRET", portion_marked="true")
check("portion_marked=true is passed to ingestion and echoed", pm.status_code == 200 and calls["portion_marked"] is True and pm.json()["portion_marked"] is True)
check("portion-marked upload is still capped at the uploader's clearance", up(uncl_tok, "SECRET", portion_marked="true").status_code == 403)
check("upload event records portion_marked", '"portion_marked": true' in Path("events_log.jsonl").read_text())

# new answer fields + feature flags reach the client
ans = client.post("/api/ask", json={"query": "interval?"}, headers=H(secret_tok)).json()
check("answer carries conflicts / entities / redaction fields the UI renders",
      all(k in ans for k in ("conflicts", "entities", "query_expansion", "redaction_count", "conflict_check")))
me = client.get("/api/me", headers=H(secret_tok)).json()
check("/api/me reports feature switches", all(k in me for k in ("redaction", "conflict_detection", "entity_resolution")))

# static + headers
page = client.get("/")
check("index served with CSP", page.status_code == 200 and "Content-Security-Policy" in page.headers)
check("history endpoint works", client.get("/api/history", headers=H(secret_tok)).status_code == 200)

sys.exit(1 if check.failed else 0)
