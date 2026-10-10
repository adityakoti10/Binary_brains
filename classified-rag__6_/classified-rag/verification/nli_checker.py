"""
Entailment-based claim verification with a DeBERTa MNLI model -- the
mechanism behind your "hallucination rate" and "citation correctness" KPIs.

For each claim: premise = the cited source passage(s), hypothesis = the claim.
- high P(entailment)      -> claim is supported, keep it
- high P(contradiction)   -> flag as a cross-source conflict / hallucination
- otherwise (neutral)     -> flag as unsupported, route to human review

Swap `_get_nli` for Vectara's HHEM or MiniCheck for a purpose-built,
lighter-weight alternative to generic MNLI.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from config import CFG
from verification.claim_decompose import Claim


@dataclass
class ClaimVerdict:
    claim: Claim
    entailment_prob: float
    contradiction_prob: float
    neutral_prob: float
    status: str  # "supported" | "contradicted" | "unsupported"


@lru_cache(maxsize=1)
def _get_nli():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CFG.nli_model)
    model = AutoModelForSequenceClassification.from_pretrained(CFG.nli_model)
    model.to(CFG.small_models_device)
    model.eval()
    return tok, model


def _label_order(model) -> dict[str, int]:
    # MoritzLaurer's checkpoints expose id2label; normalize to a fixed dict.
    id2label = {k: v.lower() for k, v in model.config.id2label.items()}
    return {v: k for k, v in id2label.items()}


def nli_batch(
    pairs: list[tuple[str, str]], max_length: int = 512, batch_size: int = 8
) -> list[tuple[float, float, float]]:
    """Batched NLI. Each pair is (premise, hypothesis); returns
    (entailment, contradiction, neutral) probabilities in the same order."""
    if not pairs:
        return []
    import torch  # lazy: keeps this module importable (and testable) without torch

    tok, model = _get_nli()
    labels = _label_order(model)
    out: list[tuple[float, float, float]] = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        inputs = tok(
            [p for p, _ in batch], [h for _, h in batch],
            truncation=True, max_length=max_length, padding=True, return_tensors="pt",
        )
        inputs = {k: v.to(CFG.small_models_device) for k, v in inputs.items()}
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1).tolist()
        out.extend((p[labels["entailment"]], p[labels["contradiction"]], p[labels["neutral"]]) for p in probs)
    return out


def verify_claim(claim: Claim, cited_passages: list[str]) -> ClaimVerdict:
    premise = " ".join(cited_passages) if cited_passages else ""

    if not premise:
        # Claim cited nothing -- automatic fail, no need to run the model.
        return ClaimVerdict(claim, 0.0, 0.0, 1.0, "unsupported")

    entail, contra, neutral = nli_batch([(premise, claim.text)], max_length=512)[0]

    if entail >= CFG.entailment_accept_threshold:
        status = "supported"
    elif contra >= CFG.contradiction_flag_threshold:
        status = "contradicted"
    else:
        status = "unsupported"

    return ClaimVerdict(claim, entail, contra, neutral, status)


def verify_answer(claims: list[Claim], source_texts: dict[int, str]) -> list[ClaimVerdict]:
    """source_texts maps 1-based [Sx] index -> passage text."""
    verdicts = []
    for c in claims:
        cited = [source_texts[i] for i in c.cited_indices if i in source_texts]
        verdicts.append(verify_claim(c, cited))
    return verdicts


def hallucination_rate(verdicts: list[ClaimVerdict]) -> float:
    if not verdicts:
        return 0.0
    bad = sum(1 for v in verdicts if v.status != "supported")
    return bad / len(verdicts)
