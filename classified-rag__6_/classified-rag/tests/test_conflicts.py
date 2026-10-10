"""
Cross-source conflict detection tests. The NLI model is replaced by a tiny fake so the
logic AROUND it (sentence split, blocking, both-direction confirmation, numeric rule,
entity canonicalisation, budget) is what's under test:   python tests/test_conflicts.py
"""
import re
from dataclasses import dataclass

from _t import check, finish, scratch_env

scratch_env()

from config import CFG  # noqa: E402
from entity_resolution.resolver import ResolvedEntity  # noqa: E402
from verification import conflict_detector as CD  # noqa: E402
from verification import nli_checker  # noqa: E402
from verification.claim_decompose import Claim  # noqa: E402


@dataclass
class Src:
    text: str
    doc_id: str


nli_calls = []          # every (premise, hypothesis) pair the fake model was asked about
batches = []            # size of each call
fake_mode = {"mode": "numbers"}


def fake_nli(pairs, max_length=512, batch_size=8):
    nli_calls.extend(pairs)
    batches.append(len(pairs))
    out = []
    for p, h in pairs:
        nums = lambda t: set(re.findall(r"\d+", t))  # noqa: E731
        if fake_mode["mode"] == "neutral":
            out.append((0.1, 0.1, 0.8))
        elif fake_mode["mode"] == "forward_only":  # contradicts only when premise has the smaller number
            a, b = sorted(nums(p))[:1], sorted(nums(h))[:1]
            out.append((0.0, 0.95 if a and b and a < b else 0.05, 0.05))
        elif nums(p) != nums(h):
            out.append((0.02, 0.95, 0.03))
        else:
            out.append((0.9, 0.02, 0.08))
    return out


nli_checker.nli_batch = fake_nli
A = "The scheduled maintenance interval is 500 operating hours for the main pump."
B = "The scheduled maintenance interval is 250 operating hours for the main pump."

print("-- basics")
fake_mode["mode"] = "numbers"
cs = CD.detect_conflicts([Src(A, "docA"), Src(B, "docB")])
check("different numbers in two documents -> contradiction found", len(cs) == 1 and cs[0].kind == "contradiction")
check("indices + sentences reported", (cs[0].source_a, cs[0].source_b) == (1, 2) and "500" in cs[0].sentence_a and "250" in cs[0].sentence_b)
check("confidence is the two-direction average", cs[0].contradiction == 0.95)
check("agreeing sources -> no conflict", CD.detect_conflicts([Src(A, "docA"), Src(A + " Confirmed.", "docB")]) == [])
check("single source -> nothing, no NLI call", (nli_calls.clear() or CD.detect_conflicts([Src(A, "d")]) == []) and not nli_calls)

print("-- scope")
check("same document is not compared (cross-doc only)", CD.detect_conflicts([Src(A, "same"), Src(B, "same")]) == [])
CFG.conflict_cross_doc_only = False
check("...unless cross_doc_only is switched off", len(CD.detect_conflicts([Src(A, "same"), Src(B, "same")])) == 1)
CFG.conflict_cross_doc_only = True
nli_calls.clear()
CD.detect_conflicts([Src("Completely unrelated text about the cafeteria menu today.", "a"),
                     Src("Torque values for the flange bolts are listed below in newton metres.", "b")])
check("unrelated sentences never reach the NLI model", not nli_calls)
CFG.conflict_detection = False
check("CONFLICT_DETECTION=off -> []", CD.detect_conflicts([Src(A, "a"), Src(B, "b")]) == [])
CFG.conflict_detection = True

print("-- numeric rule (NLI says neutral)")
fake_mode["mode"] = "neutral"
cs = CD.detect_conflicts([Src(A, "a"), Src(B, "b")])
check("numeric mismatch caught even when NLI misses it", len(cs) == 1 and cs[0].kind == "numeric" and cs[0].contradiction is None)
check("extra reference number is not a mismatch",
      CD.detect_conflicts([Src("The interval is 500 operating hours for the main pump (see Table 4).", "a"),
                           Src("The interval is 500 operating hours for the main pump as specified.", "b")]) == [])
check("different part sizes (M8 vs M10) are different objects, not a conflict",
      CD.detect_conflicts([Src("Torque the M8 flange bolts to 25 Nm during assembly.", "a"),
                           Src("Torque the M10 flange bolts to 45 Nm during assembly.", "b")]) == [])
check("date mismatch flagged", len(CD.detect_conflicts([Src("The pump was replaced on 12 May 2026 by the crew.", "a"),
                                                       Src("The pump was replaced on 14 May 2026 by the crew.", "b")])) == 1)

print("-- both-direction confirmation")
fake_mode["mode"] = "forward_only"
cs = CD.detect_conflicts([Src("Pump interval is 250 hours for the main assembly unit.", "a"),
                          Src("Pump interval is 500 hours for the main assembly unit.", "b")])
# forward NLI = 0.95, reverse = 0.05 -> average 0.5 < 0.70 -> not an NLI conflict; numeric rule still reports it
check("one-directional NLI signal is not enough for kind=contradiction", all(c.kind != "contradiction" for c in cs))

print("-- entities")
fake_mode["mode"] = "numbers"
nli_calls.clear()
ents = [ResolvedEntity("John Smith", "person", ["J. Smith"], [1, 2])]
sa = "J. Smith reported the pump interval as 500 hours during the audit."
sb = "John Smith reported the pump interval as 250 hours during the audit."
cs = CD.detect_conflicts([Src(sa, "a"), Src(sb, "b")], ents)
check("conflict found between alias and canonical form", len(cs) == 1)
check("NLI saw canonical names, not the alias", nli_calls and all("J. Smith" not in p and "J. Smith" not in h for p, h in nli_calls))
check("original wording is what gets reported", cs[0].sentence_a.startswith("J. Smith"))
wa = "Smith said the coolant pump reached 500 hours after commissioning last spring."
wb = "Smith noted the coolant pump failed at 250 hours following overhaul recently."
ov = CD._jaccard(CD._content_words(wa), CD._content_words(wb))
check("(fixture) word overlap sits between the entity bar and the normal bar",
      CFG.conflict_min_overlap * 0.6 < ov < CFG.conflict_min_overlap)
smith = [ResolvedEntity("Smith", "person", [], [1, 2])]
check("low-overlap pair is NOT compared without a shared entity", CD.detect_conflicts([Src(wa, "a"), Src(wb, "b")]) == [])
check("...but IS compared (and flagged) when both mention the same resolved entity",
      len(CD.detect_conflicts([Src(wa, "a"), Src(wb, "b")], smith)) == 1)

print("-- budget, ordering, redaction markers")
many = [Src(f"The interval for the main pump is {100 + i} hours in the manual. Extra sentence number {i} about the interval." , f"d{i}") for i in range(12)]
nli_calls.clear(); batches.clear()
CFG.conflict_max_pairs = 10
CD.detect_conflicts(many)
check("NLI budget: first (forward) pass never exceeds max_pairs", batches and batches[0] <= 10)
CFG.conflict_max_pairs = 24
CFG.conflict_max_reported = 2
check("reported list capped", len(CD.detect_conflicts(many)) <= 2)
CFG.conflict_max_reported = 5
red = CD.detect_conflicts([Src("The interval is [REDACTED:CODEWORD] hours for the main pump assembly.", "a"),
                           Src("The interval is 250 hours for the main pump assembly here.", "b")])
check("redaction markers don't crash or create numeric conflicts", isinstance(red, list))

print("-- acknowledged")
c1 = CD.Conflict(1, 2, "x", "y", "contradiction", 0.9)
c2 = CD.Conflict(1, 3, "x", "z", "contradiction", 0.9)
CD.mark_acknowledged([c1, c2], [Claim("Source 1 says X while 2 says Y", [1, 2]), Claim("Other", [3])])
check("acknowledged only when one answer sentence cites both sides", c1.acknowledged and not c2.acknowledged)

print("-- sentence splitter")
sp = CD.split_sentences("Short. This is a proper sentence with enough words in it. | table | row |\nAnother reasonable full line follows here.", 40)
check("drops fragments, splits on newlines", len(sp) == 2)

print("-- abbreviation-aware splitting (shared with claim_decompose)")
sp = CD.split_sentences("Dr. Jones and J. Smith met U.S. Navy staff on Monday. The pump failed after inspection.", 40)
check("initials, titles and U.S. don't end a sentence", len(sp) == 2 and sp[0].startswith("Dr. Jones and J. Smith"))
from verification.claim_decompose import decompose  # noqa: E402
cl = decompose("J. Smith reported 500 hours [S1]. The logs agree [S2].")
check("claims are not cut inside a name", len(cl) == 2 and cl[0].text.startswith("J. Smith reported") and cl[0].cited_indices == [1])

finish()
