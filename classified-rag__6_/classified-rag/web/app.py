"""
Web interface + JSON API for the pipeline.

    uvicorn web.app:app --host 127.0.0.1 --port 8080
    -> open http://127.0.0.1:8080

Security notes
  * The caller's clearance is ALWAYS looked up server-side from their token.
    Nothing the browser sends can raise it.
  * Uploads can only be labelled at a level the user is cleared for, and every
    upload is written to a signed events log (events_log.jsonl).
  * Binds to localhost by default. Put it behind TLS + your real auth before
    exposing it to a network.
"""
from __future__ import annotations

import re
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from config import CFG
from security.classification import RequestContext, levels_up_to, resolve_user_clearance

STATIC_DIR = Path(__file__).parent / "static"
ALLOWED_SUFFIXES = {".pdf", ".txt", ".md", ".docx", ".png", ".jpg", ".jpeg", ".log"}
MAX_QUERY_CHARS = 2000

app = FastAPI(title="Classified Document Fusion", docs_url=None, redoc_url=None)


@app.middleware("http")
async def security_headers(request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'"
    )
    if request.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


def current_ctx(authorization: str = Header(default="")) -> RequestContext:
    token = authorization.removeprefix("Bearer ").strip()
    try:
        return resolve_user_clearance(token)
    except PermissionError:
        raise HTTPException(status_code=401, detail="Invalid or missing token")


def _pipeline():
    """Imported lazily so the server starts (and shows a clear error) even if
    heavy model dependencies aren't installed yet."""
    import pipeline

    return pipeline


# ------------------------------------------------------------------ pages --
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


# -------------------------------------------------------------------- API --
@app.get("/api/me")
def me(ctx: RequestContext = Depends(current_ctx)):
    return {
        "user_id": ctx.user_id,
        "clearance": ctx.max_clearance,
        "levels": CFG.classification_levels,
        "upload_levels": levels_up_to(ctx.max_clearance),
        "allowed_suffixes": sorted(ALLOWED_SUFFIXES),
        "max_upload_mb": CFG.max_upload_mb,
        "llm_model": CFG.llm_model,
        "triage_backend": CFG.triage_backend,
        "redaction": CFG.redaction_enabled,
        "conflict_detection": CFG.conflict_detection,
        "entity_resolution": CFG.entity_resolution,
    }


@app.get("/api/health")
def health(ctx: RequestContext = Depends(current_ctx)):
    """Quick check of the three services the pipeline depends on."""
    out: dict[str, dict] = {}

    try:
        import psycopg2

        psycopg2.connect(CFG.pg_dsn, connect_timeout=2).close()
        out["postgres"] = {"ok": True}
    except Exception as e:  # noqa: BLE001
        out["postgres"] = {"ok": False, "error": str(e)[:160]}

    try:
        from neo4j import GraphDatabase

        drv = GraphDatabase.driver(CFG.neo4j_uri, auth=(CFG.neo4j_user, CFG.neo4j_password))
        drv.verify_connectivity()
        drv.close()
        out["neo4j"] = {"ok": True}
    except Exception as e:  # noqa: BLE001
        out["neo4j"] = {"ok": False, "error": str(e)[:160]}

    try:
        import requests

        requests.get(f"{CFG.llm_base_url}/models", timeout=2).raise_for_status()
        out["llm"] = {"ok": True, "model": CFG.llm_model}
    except Exception as e:  # noqa: BLE001
        out["llm"] = {"ok": False, "error": str(e)[:160]}

    return out


def _safe_name(name: str) -> str:
    base = Path(name or "file").name  # strips any directory components
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._") or "file"
    return base[:120]


@app.post("/api/upload")
async def upload(
    files: list[UploadFile] = File(...),
    classification: str = Form(...),
    portion_marked: bool = Form(False),
    ctx: RequestContext = Depends(current_ctx),
):
    """`classification` is the document's OVERALL level. If `portion_marked` is set, the
    files carry (U)/(C)/(S)/(TS) paragraph marks and are split so each chunk gets its own
    level (see security/redaction.py). Unmarked paragraphs keep the overall level; a mark
    above the overall level wins (fail closed)."""
    if classification not in levels_up_to(ctx.max_clearance):
        raise HTTPException(status_code=403, detail="You cannot label documents above your clearance")
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    target_dir = Path(CFG.upload_dir) / classification
    target_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = CFG.max_upload_mb * 1024 * 1024

    saved: list[Path] = []
    try:
        for f in files:
            name = _safe_name(f.filename or "")
            if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
                raise HTTPException(status_code=400, detail=f"File type not allowed: {name}")
            dest = target_dir / f"{uuid.uuid4().hex[:8]}_{name}"
            size = 0
            with open(dest, "wb") as out:
                while chunk := await f.read(1 << 20):
                    size += len(chunk)
                    if size > max_bytes:
                        out.close()
                        dest.unlink(missing_ok=True)
                        raise HTTPException(status_code=413, detail=f"{name} exceeds {CFG.max_upload_mb} MB")
                    out.write(chunk)
            saved.append(dest)
    except HTTPException:
        for p in saved:
            p.unlink(missing_ok=True)
        raise

    try:
        n_chunks = _pipeline().ingest_files(saved, classification, portion_marked=portion_marked)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)[:300]}")

    from provenance.audit import write_event

    write_event(
        "upload",
        {
            "user_id": ctx.user_id,
            "classification": classification,
            "portion_marked": portion_marked,
            "files": [p.name for p in saved],
            "chunks": n_chunks,
        },
    )
    return {
        "files": [p.name for p in saved],
        "chunks_stored": n_chunks,
        "classification": classification,
        "portion_marked": portion_marked,
    }


class AskBody(BaseModel):
    query: str


@app.post("/api/ask")
def ask(body: AskBody, ctx: RequestContext = Depends(current_ctx)):
    q = body.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty question")
    if len(q) > MAX_QUERY_CHARS:
        raise HTTPException(status_code=400, detail=f"Question too long (max {MAX_QUERY_CHARS} chars)")
    try:
        result = _pipeline().answer_query(q, ctx)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)[:300]}")
    return result.to_dict()


@app.get("/api/history")
def history(limit: int = 20, ctx: RequestContext = Depends(current_ctx)):
    """This user's recent answers from the signed audit log."""
    import json

    path = Path("audit_log.jsonl")
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines()[-500:]:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("user_id") == ctx.user_id:
            rows.append(
                {
                    "query": rec["query"],
                    "answer": rec["answer"],
                    "timestamp": rec["timestamp"],
                    "hallucination_rate": rec["hallucination_rate"],
                }
            )
    return rows[-max(1, min(limit, 100)):][::-1]


@app.exception_handler(Exception)
async def unhandled(_, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": "Internal error"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web.app:app", host="127.0.0.1", port=8080)
