# Classified Multi-Source Document Fusion Pipeline

Ingest scanned PDFs, manuals, reports and logs at different classification
levels, then ask questions through a web page and get **one cited, fact-checked
answer**. Everything runs locally; nothing calls an outside API.

This is a working **scaffold**, not a finished product. See "Not done yet" at the bottom.

## What runs where (sized for an RTX 4060, 8 GB)

| Job | Tool | Runs on |
|---|---|---|
| Writes the cited answer | **Qwen2.5-7B-Instruct** (4-bit) via Ollama | GPU (~5 GB) |
| OCR / layout / tables | Docling | CPU |
| Log parsing | Drain3 | CPU |
| Embeddings, reranking | BGE-M3, bge-reranker-v2-m3 | CPU by default (`SMALL_MODELS_DEVICE`) |
| Claim checking + cross-source conflict check | DeBERTa-v3 NLI | CPU by default |
| Entity extraction | GLiNER | CPU by default |
| Triage tags (doc type / urgency / log severity) | **SetFit** (default) or **Laya** (optional) | CPU by default |
| Storage | Postgres + pgvector | Docker |
| Provenance graph | Neo4j | Docker |
| Web page + API | FastAPI | CPU |

Why CPU for the small models: the 7B LLM needs most of the 8 GB of VRAM. For a
big first ingestion, set `SMALL_MODELS_DEVICE=cuda` while the LLM is idle, then switch back.

## Setup (Windows PowerShell; Linux/macOS commands are the same apart from activation)

```powershell
# 1. Python 3.10+ environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. PyTorch with CUDA -- pick the command from pytorch.org/get-started/locally, then:
pip install -r requirements.txt

# 3. Postgres+pgvector and Neo4j via Docker Desktop
docker run -d --name pg -e POSTGRES_USER=rag -e POSTGRES_PASSWORD=rag -e POSTGRES_DB=classified_rag -p 5432:5432 pgvector/pgvector:pg16
docker exec -i pg psql -U rag -d classified_rag < schema.sql
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/changeme neo4j:5

# 4. The LLM: install Ollama (ollama.com), then
ollama pull qwen2.5:7b-instruct
```

**Give Ollama a bigger context window.** The answer prompt holds ~6 source
passages and can exceed Ollama's small default context, which silently cuts
sources off. In recent Ollama versions set the environment variable
`OLLAMA_CONTEXT_LENGTH=8192` (Windows: System Properties -> Environment
Variables), then restart Ollama.

Set the environment variables listed in `.env.example` (at minimum `AUDIT_HMAC_KEY`).

## Create a user, start the web page

```powershell
python -m security.users add alice SECRET      # prints a login token ONCE -- copy it
python -m routing.document_router              # one-time: trains the SetFit triage classifiers
uvicorn web.app:app --host 127.0.0.1 --port 8080
```

Open http://127.0.0.1:8080 and paste the token.

1. **Add documents** -- choose files and a classification label (only up to your clearance).
   Tick **portion markings** if paragraphs start with `(U)`, `(C)`, `(S)` or `(TS)` (see below).
2. **Ask a question** -- you get the answer with clickable `[S1]` citations, a
   per-claim check (supported / unsupported / contradicted), any **conflicts between sources**,
   the source passages (with `[REDACTED:...]` where your account may not see something),
   the **entities** found across sources, and KPI tiles.
3. The dots at the top show whether Database, Provenance (Neo4j) and LLM are reachable.

Users: `python -m security.users list`. Clearance is **always** looked up on the
server from the token; nothing the browser sends can raise it.

## Where things are saved

| What | Where |
|---|---|
| Uploaded files | `data/uploads/<LEVEL>/` |
| Indexed text + vectors + labels | Postgres table `chunks` |
| Signed record of every answer | `audit_log.jsonl` |
| Signed record of every upload | `events_log.jsonl` |
| Answer -> claim -> source graph | Neo4j (browse at http://localhost:7474) |

## Triage backends: SetFit vs Laya

Both tag every chunk with `doc_type`, `urgency` and (for logs) `log_severity`
using a small local model -- never the LLM.

* **SetFit (default).** Trains on your own labeled examples in seconds.
  The seed examples in `routing/document_router.py` are tiny placeholders; add
  10-20 real snippets per label and re-run `python -m routing.document_router`.
* **Laya** (`TRIAGE_BACKEND=laya`). Apache-2.0, ~421M parameters, ModernBERT-based,
  one forward pass, no text generation. Caveats from its own docs: base
  checkpoints are near chance zero-shot, and the `typed-decisions` checkpoint is
  fine-tuned on business workflows (invoices, incidents, tickets), not on your
  manuals or logs. **Test it on a labeled sample of your chunks before relying on
  it** -- chunks flagged `triage_needs_review` had low confidence. Keep it under ~20 options per question.

Switching is one environment variable; the rest of the pipeline doesn't change.

## Where "no spillage" is enforced

`vectorstore/pgvector_store.search()` turns the user's clearance into a hard SQL
filter (`classification = ANY(allowed_levels)`) *before* any text reaches the
reranker or the LLM. Uploads can only be labelled at or below the uploader's
clearance. Label every file truthfully: a document labelled lower than its real
level would be readable by people who shouldn't see it.

## Redaction inside documents

Two layers, both enforced on the server before any text reaches the reranker or the LLM
(`security/redaction.py`):

1. **Portion markings (opt-in per upload).** Tick the checkbox when a file's paragraphs start
   with `(U)`, `(C)`, `(S)`, `(TS)` (caveats like `(S//NF)` are fine; abbreviations are in
   `config.portion_marks`). The label you pick is the document's *overall* level. The text is split
   at the marks so every chunk has **one** classification and the existing SQL clearance filter does
   the rest: a SECRET user can read a TOP_SECRET file's `(U)`/`(S)` chunks but never its `(TS)` ones.
   Unmarked paragraphs keep the overall level, and a mark *above* the overall level wins (both fail
   closed). Marks are ignored unless you tick the box, so a stray `(U)` can't downgrade text.
2. **Per-user rules.** Copy `redaction_rules.example.json` to `redaction_rules.json`. Each rule
   (regex or term list) masks matches as `[REDACTED:<NAME>]` for users **below** `min_clearance`,
   with an optional `allow_users` need-to-know list. Omitting `min_clearance` masks for everyone.
   SSNs and Luhn-valid card numbers are masked out of the box. Edits apply without a restart.
   Test rules with `python -m security.redaction "text" SECRET`.

Fail closed: a malformed rules file makes queries error out instead of answering unredacted. The answer
itself is also redacted, entities that were masked are dropped, and the audit record stores how many
redactions were applied.

## Conflicts between sources

`verification/conflict_detector.py` compares the retrieved passages **with each other** (the NLI claim
check only compares a claim with the source it cites). It splits passages into sentences, pairs up
sentences from *different documents* that talk about the same thing, and runs batched NLI on the best
`conflict_max_pairs` (24, in `config.py`) pairs; a conflict needs high contradiction probability in **both** directions.
A second rule catches near-identical sentences whose numbers differ ("500 hours" vs "250 hours"), which
generic NLI models often miss; these are reported as *possible number mismatch*.
Conflicts are shown in the UI (and whether the answer actually mentions both sides), hinted to the LLM, written to
Neo4j as `CONFLICTS_WITH` edges and counted in the signed audit log. If the check can't run the answer
is still returned, but marked "not cross-checked". Different *versions* of a document will legitimately conflict,
so the UI shows both sentences and sources for a human to judge.

## Entity resolution in the query flow

`entity_resolution/` is now used end to end:

* **Ingest:** GLiNER stores the entities mentioned in each chunk (`chunk.extra["entities"]`).
* **Retrieve:** entities in the *question* are looked up in the (clearance-filtered, redacted) candidates;
  if they appear under other names (`J. Smith`, `Gen. John Smith`), the search is repeated with those aliases
  so passages using the other name are found too. Ranking still uses the original question.
* **Answer:** names are clustered across the final sources and shown in the UI, given to the LLM as
  "name equivalences", used to line up sentences for conflict detection, and stored in Neo4j
  (`MENTIONS`, with the chunk's classification on the edge).

Merging is deliberately conservative: same NER label only; dates/identifiers must match exactly; tokens with digits
must match exactly (`Server-14A` != `14B`); an ambiguous initial or bare surname (`J. Smith` with both a John and a Jane
Smith present) is left unmerged. Everything is fail-soft: if GLiNER is missing, expansion is skipped and the pipeline carries on.
Existing chunks have no stored entities until re-ingested (the top few sources are analysed live meanwhile).
Neo4j holds entity names and queries, so treat it as holding the highest level of data it references.

## Tests

```powershell
python tests/run_all.py      # everything below; no GPU, models or databases needed
python tests/test_redaction.py   # portion markings + rules engine + fail-closed behaviour
python tests/test_conflicts.py   # conflict detector (fake NLI)
python tests/test_entities.py    # clustering + query expansion (fake NER)
python tests/test_pipeline.py    # whole ingest + query flow with only models/DB/LLM faked
python tests/test_web.py         # auth, clearance, upload limits (needs fastapi httpx python-multipart)
```

## Not done yet (honest list)

* **Real identity system.** Login is a local token table (`users.json`, hashed).
  Fine for a demo or a small team; replace `security/classification.resolve_user_clearance`
  with your IdP / ABAC engine for real use. No TLS, rate limiting or lockout yet -- keep it on localhost or behind a proper gateway.
* **Redaction limits.** Rules are text patterns you maintain; redaction can't find sensitive facts nobody wrote a rule for.
  Embeddings are computed on the original text, so a determined user could probe similarity for redacted terms.
  Portion marks are read at paragraph starts only (not inside table cells or images/scans without OCR text).
* **Conflict detection limits.** It compares the top ~6 reranked passages, sentence to sentence. Conflicts that need
  several sentences, unit conversions, or reasoning are missed; MNLI is weak on numbers (hence the numeric rule).
  Measure precision/recall on labelled pairs before trusting it.
* **Entity resolution limits.** Rule-based, English-leaning name logic; no cross-label merging; needs a labelled
  benchmark (Zingg/Dedupe are the upgrade path). GLiNER model must be downloaded ahead of time on an offline machine.
* **Chunking / claim splitting** are simple (character windows, sentence regex).
* **Evaluation.** No benchmark harness yet for the KPIs (F1, citation correctness, hallucination rate) -- build a small labeled test set first.
* **Scale.** No batching/retry for very large corpora.
