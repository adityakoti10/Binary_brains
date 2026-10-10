"""
Redaction tests (pure Python -- no GPU, DB or models):   python tests/test_redaction.py
Covers: portion-marking split, fail-closed defaults, per-user rules, need-to-know
allow-lists, overlap handling, hot-reload, malformed config, chunk copies + entity scrub.
"""
import json
import os
import time
from dataclasses import dataclass, field

from _t import check, finish, scratch_env

scratch_env()
RULES = os.environ["REDACTION_RULES_FILE"]

from config import CFG  # noqa: E402
from ingestion.doc_ingest import _classified_pieces  # noqa: E402
from security.classification import RequestContext, levels_up_to  # noqa: E402
from security import redaction as R  # noqa: E402


def write_rules(rules):
    with open(RULES, "w") as f:
        json.dump({"rules": rules}, f)
    st = os.stat(RULES)  # make sure mtime changes even on coarse filesystems
    os.utime(RULES, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000))


# ---------------------------------------------------------------- portions ---
print("-- portion markings")
doc = ("(U) Public overview of the pump.\n\n"
       "(S//NF) Secret failure analysis.\n\n"
       "- (TS) Top secret source name.\n\n"
       "## (C) Confidential heading\n\n"
       "No mark on this paragraph.\n\n"
       "(s) lowercase is not a mark, (X) unknown is not a mark.\n\n"
       "(S) Another secret paragraph.")
P = R.split_portions(doc, "TOP_SECRET")
lv = [p.level for p in P]
check("U/S/TS/C marks recognised", lv[:4] == ["UNCLASSIFIED", "SECRET", "TOP_SECRET", "CONFIDENTIAL"])
check("caveat after // is tolerated", P[1].mark == "S" and P[1].text == "Secret failure analysis.")
check("markdown bullet/heading prefix kept, mark stripped", P[2].text == "- Top secret source name." and P[3].text == "## Confidential heading")
check("unmarked paragraph inherits document level (fail closed)", P[4].level == "TOP_SECRET" and P[4].mark is None)
check("lowercase / unknown marks are NOT marks", P[5].level == "TOP_SECRET" and P[5].text.startswith("(s)"))
over = R.split_portions("(TS) too high", "SECRET")[0]
check("mark above doc label wins and is flagged", over.level == "TOP_SECRET" and over.exceeds_label)
runs = R.group_portions(R.split_portions("(S) a\n\n(S) b\n\n(U) c", "SECRET"))
check("consecutive same-level portions merge", [(r.level, r.text) for r in runs] == [("SECRET", "a\n\nb"), ("UNCLASSIFIED", "c")])
try:
    R.split_portions("x", "BOGUS"); ok = False
except ValueError:
    ok = True
check("unknown document label rejected", ok)

pieces = _classified_pieces(doc, "TOP_SECRET", portion_marked=True)
check("portion-marked ingest: every piece has exactly one level, several levels present",
      len({lvl for _, lvl, _ in pieces}) >= 3)
plain = _classified_pieces("(U) Downgrade attempt inside a SECRET file.", "SECRET", portion_marked=False)
check("marks ignored unless upload is flagged portion_marked (no silent downgrade)", plain[0][1] == "SECRET")
secret_user_sees = [t for t, lvl, _ in pieces if lvl in levels_up_to("SECRET")]
check("SECRET user's filter excludes TS chunks of the same file",
      secret_user_sees and not any("Top secret source name" in t for t in secret_user_sees))
check("...but keeps the U/S/C chunks of that file", any("Public overview" in t for t in secret_user_sees))

# ------------------------------------------------------------------- rules ---
print("-- built-in rules")
r = R.redact_text("SSN 123-45-6789 and card 4111 1111 1111 1111 and junk 1234 5678 9012 3456.", "TOP_SECRET")
check("SSN masked even for the highest clearance (built-in = mask all)", "123-45-6789" not in r.text and "[REDACTED:SSN]" in r.text)
check("Luhn-valid card masked", "4111" not in r.text and "[REDACTED:CARD]" in r.text)
check("Luhn-invalid digit run left alone", "1234 5678 9012 3456" in r.text)
check("invalid SSN area (000/666/9xx) left alone", R.redact_text("id 000-12-3456 / 666-12-3456", "SECRET").count == 0)

print("-- rules file")
write_rules([
    {"name": "CODEWORD", "terms": ["NIGHTHAWK", "Night Hawk"], "min_clearance": "TOP_SECRET", "allow_users": ["carol"]},
    {"name": "GRID", "regex": r"GRID-\d{6}", "min_clearance": "SECRET"},
    {"name": "SOURCE", "terms": ["Agent Falcon"]},  # no min_clearance -> masked for everyone (fail closed)
])
txt = "Operation NIGHTHAWK (aka night   hawk) at GRID-123456 per Agent Falcon."
u = R.redact_text(txt, "UNCLASSIFIED", "bob")
s = R.redact_text(txt, "SECRET", "bob")
t = R.redact_text(txt, "TOP_SECRET", "bob")
c = R.redact_text(txt, "SECRET", "carol")
check("UNCLASSIFIED: everything masked", u.text.count("[REDACTED:") == 4 and "NIGHTHAWK" not in u.text and "GRID" not in u.text.replace("[REDACTED:GRID]", ""))
check("multi-word term matches with flexible whitespace + case", "night   hawk" not in u.text)
check("SECRET: grid visible, codeword masked", "GRID-123456" in s.text and "NIGHTHAWK" not in s.text)
check("TOP_SECRET: codeword+grid visible, no-min_clearance source STILL masked", "NIGHTHAWK" in t.text and "Agent Falcon" not in t.text)
check("allow_users (need-to-know) overrides clearance", "NIGHTHAWK" in c.text and "night   hawk" in c.text and "Agent Falcon" not in c.text)
check("word boundaries respected", R.redact_text("NIGHTHAWKS and XNIGHTHAWK", "SECRET").count == 0)

write_rules([{"name": "ID9", "regex": r"\d{3}-\d{2}-\d{4}", "min_clearance": "TOP_SECRET"}])
o = R.redact_text("x 123-45-6789 y", "SECRET")
check("overlapping rules merge into ONE marker", o.text == "x [REDACTED:SSN] y" and o.count == 1 and set(o.kinds) == {"SSN", "ID9"})
check("built-ins still apply when the file does not override them", R.redact_text("x 123-45-6789", "TOP_SECRET").count == 1)  # ID9 visible at TS, but SSN built-in masks
write_rules([{"name": "SSN", "regex": r"\b\d{3}-\d{2}-\d{4}\b", "min_clearance": "SECRET"}])
check("file rule named SSN replaces the built-in (visible at SECRET)", R.redact_text("x 123-45-6789", "SECRET").count == 0)
check("...and still masks below that", R.redact_text("x 123-45-6789", "CONFIDENTIAL").count == 1)

write_rules([{"name": "T", "terms": ["alpha"], "min_clearance": "SECRET"}])
check("editing the rules file takes effect without restart", R.redact_text("alpha", "CONFIDENTIAL").count == 1)
write_rules([{"name": "T", "terms": ["beta"], "min_clearance": "SECRET"}])
check("...hot reload picked up the new rule", R.redact_text("alpha beta", "CONFIDENTIAL").text == "alpha [REDACTED:T]")

print("-- fail closed")
def raises(fn):
    try:
        fn(); return False
    except R.RedactionConfigError:
        return True
for label, bad in [
    ("bad regex", [{"name": "X", "regex": "(", "min_clearance": "SECRET"}]),
    ("unknown clearance", [{"name": "X", "terms": ["a"], "min_clearance": "COSMIC"}]),
    ("both regex and terms", [{"name": "X", "regex": "a", "terms": ["a"]}]),
    ("neither regex nor terms", [{"name": "X"}]),
    ("empty terms", [{"name": "X", "terms": []}]),
    ("bad name", [{"name": "x!!", "terms": ["a"]}]),
]:
    write_rules(bad)
    check(f"malformed rules file ({label}) raises instead of answering unredacted", raises(lambda: R.redact_text("a", "SECRET")))
with open(RULES, "w") as f:
    f.write("{not json")
os.utime(RULES, ns=(1, time.time_ns()))
check("invalid JSON raises", raises(lambda: R.redact_text("a", "SECRET")))
os.remove(RULES)
check("no rules file -> built-ins only", R.redact_text("123-45-6789", "SECRET").count == 1)
CFG.redaction_enabled = False
check("REDACTION=off disables masking", R.redact_text("123-45-6789", "SECRET").count == 0)
CFG.redaction_enabled = True

# ------------------------------------------------------------------ chunks ---
print("-- redact_chunks")
@dataclass
class Chunk:
    id: int
    text: str
    extra: dict = field(default_factory=dict)

write_rules([{"name": "PERSON", "terms": ["John Smith"], "min_clearance": "TOP_SECRET"}])
orig = Chunk(1, "John Smith met Acme Corp on 12 May.",
             {"entities": [{"text": "John Smith", "label": "person"}, {"text": "Acme Corp", "label": "organization"}]})
clean = Chunk(2, "Nothing sensitive here.", {"entities": [{"text": "Nothing", "label": "x"}]})
ctx = RequestContext("alice", "SECRET")
out, n = R.redact_chunks([orig, clean], ctx)
check("returns redacted COPIES; original untouched", out[0] is not orig and "John Smith" in orig.text and "John Smith" not in out[0].text)
check("redaction count + kinds recorded on the copy", n == 1 and out[0].extra["redactions"] == 1 and out[0].extra["redacted_kinds"] == ["PERSON"])
check("masked name scrubbed from stored entities (no leak via entity resolution)",
      [e["text"] for e in out[0].extra["entities"]] == ["Acme Corp"] and len(orig.extra["entities"]) == 2)
check("untouched chunk passes through unchanged", out[1] is clean)
out2, n2 = R.redact_chunks([orig], RequestContext("dave", "TOP_SECRET"))
check("cleared user sees original text", n2 == 0 and out2[0].text == orig.text)

finish()
