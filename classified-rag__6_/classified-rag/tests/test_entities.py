"""
Entity-resolution tests (fake NER, no models):   python tests/test_entities.py
"""
from dataclasses import dataclass, field

from _t import check, finish, scratch_env

scratch_env()

from config import CFG  # noqa: E402
from entity_resolution import ner, resolver as RS  # noqa: E402
from entity_resolution.dedup import Mention, cluster_entities, cluster_mentions, normalize  # noqa: E402


def M(t, l, s=None):
    return Mention(t, l, 1.0, s)


def find(cs, name):
    return next((c for c in cs if c.canonical == name), None)


print("-- clustering: merges that SHOULD happen")
cs = cluster_mentions([M("John Smith", "person", 1), M("J. Smith", "person", 2), M("Gen. John Smith", "person", 3)])
check("initial + honorific variants join the full name", len(cs) == 1 and set(cs[0].members) == {"John Smith", "J. Smith", "Gen. John Smith"} and cs[0].sources == [1, 2, 3])
check("canonical is the most specific form", cs[0].canonical == "John Smith")
cs = cluster_mentions([M("John Smith", "person", 1), M("Smith", "person", 2)])
check("a bare surname joins when only ONE person could own it", len(cs) == 1)
cs = cluster_mentions([M("Department of Defense", "organization", 1), M("DoD", "organization", 2), M("Acme Corp.", "organization", 3), M("ACME Corporation", "organization", 4)])
check("acronyms and legal-suffix variants merge", len(cs) == 2)
cs = cluster_mentions([M("M/V Aurora", "system", 1), M("MV Aurora", "system", 2)])
check("punctuation variants merge", len(cs) == 1)
check("normalisation keeps non-Latin scripts", normalize("Иван Петров", "person") == "иван петров")

print("-- clustering: merges that must NOT happen")
cs = cluster_mentions([M("John Smith", "person", 1), M("Jane Smith", "person", 2), M("J. Smith", "person", 3), M("Smith", "person", 4)])
check("ambiguous initial / bare surname is left separate", len(cs) == 4)
check("different people stay different", find(cs, "John Smith").members == ["John Smith"])
cs = cluster_mentions([M("Server-14A", "system", 1), M("Server-14B", "system", 2)])
check("digit-bearing tokens must match exactly (14A != 14B)", len(cs) == 2)
cs = cluster_mentions([M("Unit 12", "system", 1), M("Unit 21", "system", 2)])
check("Unit 12 != Unit 21", len(cs) == 2)
cs = cluster_mentions([M("12 May 2026", "date", 1), M("12 May 2025", "date", 2), M("12 May 2026", "date", 3)])
check("dates only merge when identical", len(cs) == 2 and find(cs, "12 May 2026").sources == [1, 3])
cs = cluster_mentions([M("Aurora", "system", 1), M("Aurora", "location", 2)])
check("different NER labels never merge", len(cs) == 2)
cs = cluster_mentions([M("John Smith", "person", None), M("J. Smith", "person", 2)])
check("in_query propagates to the merged cluster", cs[0].in_query and cs[0].sources == [2])
check("legacy cluster_entities(list[str]) still works",
      [c.members for c in cluster_entities(["Acme Corp", "Acme Corp.", "Globex"])] == [["Acme Corp", "Acme Corp."], ["Globex"]])

print("-- NER windowing")
class FakeGliner:
    calls = 0
    def predict_entities(self, piece, labels, threshold=0.5):
        FakeGliner.calls += 1
        out, i = [], piece.find("Aurora")
        while i != -1:
            out.append({"text": "Aurora", "label": "system", "start": i, "end": i + 6, "score": 0.9})
            i = piece.find("Aurora", i + 1)
        return out

ner._get_model = lambda: FakeGliner()
long_text = ("filler " * 140) + "Aurora " + ("filler " * 200) + "Aurora end " + "x " * 10
found = ner.extract_entities(long_text)
check("long text is processed in several windows", FakeGliner.calls > 1)
check("offsets are mapped back to the full text", all(long_text[e["start"]:e["end"]] == "Aurora" for e in found))
check("mentions in the window overlap are not duplicated", len(found) == 2)

print("-- resolver: storing / filtering")
raw = [{"text": "Acme", "label": "Organization", "score": 0.9}, {"text": "Acme", "label": "organization", "score": 0.8},
       {"text": "x", "label": "person", "score": 0.99}, {"text": "Low", "label": "person", "score": 0.2},
       {"text": "Z" * 90, "label": "person", "score": 0.9}]
ce = RS.compact_entities(raw)
check("drops low-score, tiny, huge and duplicate mentions; lowercases labels", ce == [{"text": "Acme", "label": "organization", "score": 0.9}])


@dataclass
class Ch:
    text: str
    extra: dict = field(default_factory=dict)
    id: int = 0


def fake_extract(text):
    names = {"John Smith": "person", "J. Smith": "person", "Gen. John Smith": "person", "Pump 12": "system", "NIGHTHAWK": "identifier"}
    out = []
    for n, l in names.items():
        i = text.find(n)
        if i != -1:
            out.append({"text": n, "label": l, "start": i, "end": i + len(n), "score": 0.9})
    return out

ner.extract_entities = fake_extract
chunks = [Ch("J. Smith checked Pump 12 today."), Ch("Nothing here.")]
RS.annotate_chunks(chunks)
check("annotate_chunks stores entities (empty list when none)", [e["text"] for e in chunks[0].extra["entities"]] == ["J. Smith", "Pump 12"] and chunks[1].extra["entities"] == [])

def boom(text):
    raise RuntimeError("no gliner")
ner.extract_entities = boom
c2 = [Ch("J. Smith checked Pump 12.")]
RS.annotate_chunks(c2)
check("extractor failure is fail-soft (ingestion continues, no entities)", c2[0].extra["entities"] == [] and RS.query_mentions("John Smith?") == [])
ner.extract_entities = fake_extract

print("-- resolver: chunk mentions")
stored = Ch("[REDACTED:PERSON] met J. Smith at Pump 12.", {"entities": [
    {"text": "John Smith", "label": "person", "score": 0.9}, {"text": "J. Smith", "label": "person", "score": 0.9}]})
cm = RS.chunk_mentions([stored])
check("stored mention no longer present in (redacted) text is dropped", [m.text for m in cm] == ["J. Smith"])
nolive = Ch("John Smith at Pump 12.")
check("no live NER at candidate stage", RS.chunk_mentions([nolive], allow_live=False) == [])
CFG.entity_live_fallback_max = 1
live = RS.chunk_mentions([Ch("John Smith here."), Ch("Pump 12 there."), Ch("J. Smith there.", {"entities": []})], allow_live=True)
check("live fallback only for chunks with NO stored key, and capped", [m.text for m in live] == ["John Smith"])
CFG.entity_live_fallback_max = 6

print("-- resolver: alias discovery (query expansion)")
cands = [Ch("J. Smith reported 500 hours for Pump 12.", {"entities": [{"text": "J. Smith", "label": "person", "score": 0.9}, {"text": "Pump 12", "label": "system", "score": 0.9}]}),
         Ch("John Smith reported 250 hours for Pump 12.", {"entities": [{"text": "John Smith", "label": "person", "score": 0.9}]}),
         Ch("Gen. John Smith inspected it.", {"entities": [{"text": "Gen. John Smith", "label": "person", "score": 0.9}]})]
q = "What did John Smith say about Pump 12?"
qm = RS.query_mentions(q)
check("query entities extracted", {m.text for m in qm} == {"John Smith", "Pump 12"} and all(m.source is None for m in qm))
al = RS.discover_aliases(q, qm, cands)
check("aliases = other forms of the query's entities seen in visible passages", al == ["J. Smith", "Gen. John Smith"] or set(al) == {"J. Smith", "Gen. John Smith"})
check("forms already in the question are not repeated", "John Smith" not in al and "Pump 12" not in al)
check("no aliases when nothing is retrieved / no entities", RS.discover_aliases(q, qm, []) == [] and RS.discover_aliases("hello", [], cands) == [])
CFG.entity_max_aliases_in_query = 1
check("alias count capped", len(RS.discover_aliases(q, qm, cands)) == 1)
CFG.entity_max_aliases_in_query = 5
amb = cands + [Ch("Jane Smith approved.", {"entities": [{"text": "Jane Smith", "label": "person", "score": 0.9}]})]
al2 = RS.discover_aliases("What did J. Smith say?", [Mention("J. Smith", "person", 0.9, None)], amb)
check("ambiguous initial ('J. Smith' could be John or Jane) yields no expansion", al2 == [])

print("-- resolver: resolve + use")
ents = RS.resolve_sources(qm, cands)
js = next(e for e in ents if e.canonical == "John Smith")
check("resolved entity carries aliases, sources, in_query", set(js.aliases) == {"J. Smith", "Gen. John Smith"} and js.sources == [1, 2, 3] and js.in_query)
check("in-question entities sort first", ents[0].in_query)
check("un-notable entities are omitted", not RS.resolve_sources([], [Ch("x", {"entities": [{"text": "Lonely Corp", "label": "organization", "score": 0.9}]})]))
check("canonicalize maps aliases to the canonical name, case-insensitively",
      RS.canonicalize("j. smith and Gen. John Smith met.", [js]) == "John Smith and John Smith met.")
check("canonicalize is single-pass (no 'John John Smith')", RS.canonicalize("John Smith", [js]) == "John Smith")
check("entities_in finds the canonical name for an alias occurrence", RS.entities_in("Then J. Smith left.", [js]) == {"John Smith"})
notes = RS.entity_notes(ents)
check("prompt notes list name equivalences", notes and '"J. Smith"' in notes[0] and "same person" in notes[0])
CFG.entity_resolution = False
check("ENTITY_RESOLUTION=off -> nothing happens", RS.resolve_sources(qm, cands) == [] and RS.query_mentions(q) == [])
CFG.entity_resolution = True

finish()
