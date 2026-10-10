"""
Talks to an on-prem OpenAI-compatible LLM server (Ollama, vLLM, TGI, etc.)
Point CFG.llm_base_url at whichever one is running Llama 3.3 / Qwen2.5 /
Mistral -- nothing else in this module cares which.
"""
from __future__ import annotations

import requests

from config import CFG
from synthesis.prompts import SYNTHESIS_SYSTEM_PROMPT, build_user_prompt
from vectorstore.pgvector_store import RetrievedChunk


def synthesize(
    query: str,
    sources: list[RetrievedChunk],
    conflicts: list | None = None,
    entity_notes: list[str] | None = None,
) -> tuple[str, list[RetrievedChunk]]:
    """Returns (answer_text, sources_in_citation_order) so callers can map
    [S3] back to the actual RetrievedChunk for the provenance graph / audit log.
    `conflicts` (verification.conflict_detector.Conflict) and `entity_notes`
    (entity_resolution.resolver.entity_notes) are optional hints added to the prompt."""
    passages = [s.text for s in sources]
    resp = requests.post(
        f"{CFG.llm_base_url}/chat/completions",
        headers={"Authorization": f"Bearer {CFG.llm_api_key}"},
        json={
            "model": CFG.llm_model,
            "messages": [
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(query, passages, conflicts, entity_notes)},
            ],
            "temperature": 0.1,  # low temperature: this is a synthesis task, not creative writing
        },
        timeout=120,
    )
    resp.raise_for_status()
    answer = resp.json()["choices"][0]["message"]["content"]
    return answer, sources
