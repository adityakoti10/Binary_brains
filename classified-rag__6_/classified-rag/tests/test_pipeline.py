"""
End-to-end test of the query + ingest flow with ONLY the models / DB / LLM faked.
The real retriever, redaction, entity resolution, conflict detection, claim
verification wiring, audit log and ingestion chunking all run:   python tests/test_pipeline.py
"""
import json
import os
import re
import sys
import types
from pathlib import Path

from _t import check, finish, scratch_env, stub_db_drivers

tmp = scratch_env()
RULES = os.environ["REDACTION_RULES_FILE"]
stub_db_drivers()
# ingestion.log_ingest needs drain3; not under test here
lg = types.ModuleType("ingestion.log_ingest"); lg.logs_to_chunks = lambda *a, **k: []
sys.modules["ingestion.log_ingest"] = lg

from config import CFG, clearance_rank  # noqa: E402
from entity_resolution import ner  # noqa: E402
from security import redaction as R  # noqa: E402
from security.classification import RequestContext  # noqa: E402
from vectorstore.pgvector_store import RetrievedChunk  # noqa: E402
from verification import nli_checker  # noqa: E402
import pipeline as P  # noqa: E402
from retrieval import retriever as RT  # noqa: E402
from provenance.audit import AuditRecord  # noqa: E402

json.dump({"rules": [{"name": "CODEWORD", "terms": ["NIGHTHAWK"], "min_clearance": "TOP_SECRET"}]}, open(RULES, "w"))

# --------------------------------------------------------------- fake world ---
def E(t, l): return {"text": t, "label": l, "score": 0.9}

DB = [  # (keys that make the fake ANN search return it, chunk)
    (["pump 12", "interval"], RetrievedChunk(1, "J. Smith reported that the scheduled maintenance interval for Pump 12 is 500 operating hours.", "/srv/a.pdf", "docA", 3, "SECRET", {"entities": [E("J. Smith", "person"), E("Pump 12", "system")]}, 0.9)),
    (["john smith", "pump 12", "interval"], RetrievedChunk(2, "John Smith stated that the scheduled maintenance interval for Pump 12 is 250 operating hours. Contact code NIGHTHAWK for details.", "/srv/b.pdf", "docB", 1, "SECRET", {"entities": [E("John Smith", "person"), E("Pump 12", "system"), E("NIGHTHAWK", "identifier")]}, 0.8)),
    (["pump 12"], RetrievedChunk(3, "Top secret: Pump 12 is scheduled for covert replacement.", "/srv/c.pdf", "docC", 1, "TOP_SECRET", {"entities": [E("Pump 12", "system")]}, 0.7)),
    (["cafeteria"], RetrievedChunk(4, "The cafeteria opens at eight every morning for staff.", "/srv/d.pdf", "docD", 1, "UNCLASSIFIED", {"entities": []}, 0.6)),
    (["j. smith"], RetrievedChunk(5, "J. Smith inspected the northern depot on 3 June without incident.", "/srv/e.pdf", "docE", 2, "SECRET", {"entities": [E("J. Smith", "person")]}, 0.5)),
]
searches = []


def fake_search(q_emb, max_clearance, top_k=25, allowed_doc_ids=None):
    """Stands in for pgvector: same clearance rule as the SQL WHERE clause."""
    searches.append((q_emb, max_clearance))
    allowed = CFG.classification_levels[: clearance_rank(max_clearance) + 1]
    q = q_emb.lower()
    return [RetrievedChunk(**{**c.__dict__, "extra": dict(c.extra)}) for keys, c in DB
            if c.classification in allowed and any(k in q for k in keys)]


def fake_ner(text):
    lex = {"John Smith": "person", "J. Smith": "person", "Pump 12": "system", "NIGHTHAWK": "identifier"}
    return [{"text": n, "label": l, "start": text.find(n), "end": text.find(n) + len(n), "score": 0.9}
            for n, l in lex.items() if n in text]


def fake_nli(pairs, max_length=512, batch_size=8):
    nums = lambda t: set(re.findall(r"\d+", t))  # noqa: E731
    return [(0.9, 0.02, 0.08) if nums(h) <= nums(p) else (0.02, 0.95, 0.03) for p, h in pairs]


llm_calls = []


def fake_synth(query, sources, conflicts=None, entity_notes=None):
    llm_calls.append({"sources": [s.text for s in sources], "conflicts": conflicts, "notes": entity_notes})
    i500 = next((i for i, s in enumerate(sources, 1) if "500" in s.text), None)
    i250 = next((i for i, s in enumerate(sources, 1) if "250" in s.text), None)
    if i500 is None or i250 is None:
        return "The cafeteria opens at eight [S1].", sources
    return (f"Source [S{i500}] reports 500 hours, while [S{i250}] reports 250 hours. Contact NIGHTHAWK [S{i250}].", sources)


graph_calls = []
gm = types.ModuleType("provenance.graph_store")
class FakeGraph:
    def record_answer(self, *a): graph_calls.append(a)
    def close(self): pass
gm.ProvenanceGraph = FakeGraph
sys.modules["provenance.graph_store"] = gm

RT.embed_query = lambda q: q                 # "embedding" = the text itself
RT.search = fake_search
RT.rerank = lambda q, cands, top_k=6: cands[:top_k]
ner.extract_entities = fake_ner
nli_checker.nli_batch = fake_nli
P.synthesize = fake_synth

secret = RequestContext("alice", "SECRET")
ts = RequestContext("tom", "TOP_SECRET")
Q = "What did John Smith say about the Pump 12 maintenance interval?"

# ------------------------------------------------------------------- query ---
print("-- SECRET user")
r = P.answer_query(Q, secret)
docs = {s["doc_id"] for s in r.sources}
check("clearance: TOP_SECRET chunk never reaches the sources", "docC" not in docs and all(clearance_rank(s["classification"]) <= 2 for s in r.sources))
check("clearance: TOP_SECRET text never reaches the LLM", not any("covert" in t for t in llm_calls[-1]["sources"]))
check("entity expansion pulled in the passage that only says 'J. Smith'", "docE" in docs and r.query_expansion == ["J. Smith"])
check("expansion used the same clearance (2 searches, both SECRET)", len(searches) == 2 and {c for _, c in searches} == {"SECRET"})
check("original question is what the reranker/LLM answer", searches[0][0] == Q and "also referred to as: J. Smith" in searches[1][0])

leak = json.dumps(r.to_dict()) + " ".join(llm_calls[-1]["sources"])
check("redaction: codeword absent from LLM input, answer, snippets, entities", "NIGHTHAWK" not in leak)
check("redaction: marker present and counted", "[REDACTED:CODEWORD]" in leak and r.redaction_count >= 2)  # source + answer
check("source flagged with its redaction count", any(s["redactions"] == 1 for s in r.sources))

check("conflict: 500 vs 250 detected across documents", len(r.conflicts) == 1 and r.conflicts[0]["kind"] == "contradiction")
c = r.conflicts[0]
check("conflict points at the right sources, keeps original wording", {c["source_a"], c["source_b"]} == {1, 2} and c["sentence_a"].startswith("J. Smith"))
check("conflict acknowledged: answer cites both sides in one sentence", c["acknowledged"] is True)
check("conflict_check ok; conflicts + name notes were handed to the LLM prompt",
      r.conflict_check == "ok" and llm_calls[-1]["conflicts"] and any("J. Smith" in n for n in llm_calls[-1]["notes"]))
js = next(e for e in r.entities if e["canonical"] == "John Smith")
check("entities: John Smith resolved with alias, in_query", js["aliases"] == ["J. Smith"] and js["in_query"])
check("entities: masked codeword never appears as an entity", all(e["canonical"] != "NIGHTHAWK" for e in r.entities))

rec = json.loads(Path("audit_log.jsonl").read_text().splitlines()[-1])
check("audit log has redaction + conflict counts and a valid signature",
      rec["redaction_count"] == r.redaction_count and rec["conflict_count"] == 1 and AuditRecord(**rec).verify())
check("provenance graph got conflicts + entities", graph_calls and len(graph_calls[-1][4]) == 1 and graph_calls[-1][5])

print("-- TOP_SECRET user (same question, different view)")
searches.clear()
t = P.answer_query(Q, ts)
check("cleared user sees the codeword and the TS passage", "NIGHTHAWK" in t.answer and any(s["doc_id"] == "docC" for s in t.sources) and t.redaction_count == 0)

print("-- few / no sources")
none = P.answer_query("something unrelated to the corpus", secret, log_provenance=False)
check("nothing retrieved -> clean 'no sources' result", none.answer.startswith("No sources") and none.conflicts == [] and none.entities == [] and none.conflict_check == "skipped")
one = P.answer_query("cafeteria hours", secret, log_provenance=False)
check("single source -> no conflicts, check reported ok", one.conflicts == [] and len(one.sources) == 1 and one.conflict_check == "ok")

print("-- degradation rules")
def nli_boom(*a, **k): raise RuntimeError("model OOM")
real_nli = nli_checker.nli_batch
# conflict check uses nli_batch first; verify_answer uses it later -> make ONLY the conflict pass fail
import verification.conflict_detector as CDm  # noqa: E402
orig_detect = P.detect_conflicts
P.detect_conflicts = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("model OOM"))
f = P.answer_query(Q, secret, log_provenance=False)
check("conflict check failure is fail-soft AND visible", f.conflict_check == "failed" and f.answer and f.conflicts == [])
P.detect_conflicts = orig_detect

ner.extract_entities = lambda text: (_ for _ in ()).throw(RuntimeError("no gliner"))
g = P.answer_query(Q, secret, log_provenance=False)
check("live NER failure is fail-soft: answer still produced, only query expansion is lost", bool(g.answer) and g.query_expansion == [])
check("...and entities stored at ingest are still used to cluster names across sources", any(e["canonical"] == "John Smith" for e in g.entities))
ner.extract_entities = fake_ner

CFG.entity_resolution = False
h = P.answer_query(Q, secret, log_provenance=False)
check("ENTITY_RESOLUTION=off: no expansion, still answers + redacts", h.query_expansion == [] and "NIGHTHAWK" not in json.dumps(h.to_dict()))
CFG.entity_resolution = True

open(RULES, "w").write("{broken")
os.utime(RULES, ns=(1, 2_000_000_000))
try:
    P.answer_query(Q, secret, log_provenance=False); ok = False
except R.RedactionConfigError:
    ok = True
check("broken redaction rules -> query FAILS (never answers unredacted)", ok)

# ------------------------------------------------------------------ ingest ---
print("-- ingest")
stored = {}
P.embed_texts = lambda texts: [[0.0] * 4 for _ in texts]
P.upsert_chunks = lambda chunks, emb: stored.setdefault("chunks", chunks)
ner.extract_entities = fake_ner
doc = Path(tmp) / "brief.txt"
doc.write_text("(U) John Smith visited Pump 12.\n\n(S) Codeword NIGHTHAWK is active.\n\n(TS) Source identity withheld.\n\nUnmarked closing paragraph.")
n = P.ingest_files([doc], "TOP_SECRET", triage=False, portion_marked=True)
ch = stored["chunks"]
levels = [c.classification_level for c in ch]
check("portion-marked upload -> chunks at U / S / TS", levels[0] == "UNCLASSIFIED" and "SECRET" in levels and levels.count("TOP_SECRET") >= 1 and n == len(ch))
check("unmarked trailing paragraph kept at the document level", "Unmarked closing paragraph." in ch[-1].text and ch[-1].classification_level == "TOP_SECRET")
check("entity mentions stored on every chunk at ingest", all("entities" in c.extra for c in ch) and ch[0].extra["entities"][0]["text"] in ("John Smith", "Pump 12"))
stored.clear()
P.ingest_files([doc], "SECRET", triage=False)
check("without the flag the whole file keeps the document label", {c.classification_level for c in stored["chunks"]} == {"SECRET"})

finish()
