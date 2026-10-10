"""
Central config. Everything that talks to real infrastructure (Postgres,
Neo4j, an LLM server) reads its connection info from here / env vars.
Swap any of these without touching pipeline logic.
"""
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Postgres + pgvector ---
    pg_dsn: str = os.getenv(
        "PG_DSN", "postgresql://rag:rag@localhost:5432/classified_rag"
    )
    embedding_dim: int = 1024  # bge-m3 output dim

    # --- Neo4j (provenance graph) ---
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "changeme")

    # --- On-prem LLM server (Ollama / vLLM, OpenAI-compatible /v1/chat/completions) ---
    # Sized for an 8 GB GPU (RTX 4060): a 7B model at 4-bit is ~5 GB.
    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
    llm_api_key: str = os.getenv("LLM_API_KEY", "not-needed-on-prem")

    # --- Where the small models (embedder, reranker, NLI, Laya) run ---
    # "cpu" leaves the whole GPU to the LLM. Use "cuda" for bulk ingestion
    # (e.g. set SMALL_MODELS_DEVICE=cuda while the LLM is idle).
    small_models_device: str = os.getenv("SMALL_MODELS_DEVICE", "cpu")

    # --- Fast triage of chunks: doc type / urgency / log severity ---
    #   "setfit" = few-shot classifier trained on your examples (default, safest)
    #   "laya"   = Laya typed-decision model (Jev-style, needs validation on your data)
    #   "off"    = skip triage
    triage_backend: str = os.getenv("TRIAGE_BACKEND", "setfit")
    laya_checkpoint: str = os.getenv("LAYA_CHECKPOINT", "typed-decisions")
    laya_min_confidence: float = 0.60  # below this a triage label is flagged for review

    # --- Models ---
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    nli_model: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    ner_model: str = "urchade/gliner_multi-v2.1"

    # --- Classification levels, low -> high. Extend to match your program's scheme. ---
    classification_levels: list = field(
        default_factory=lambda: ["UNCLASSIFIED", "CONFIDENTIAL", "SECRET", "TOP_SECRET"]
    )

    # --- Retrieval / verification thresholds ---
    top_k_retrieve: int = 25
    top_k_rerank: int = 6
    entailment_accept_threshold: float = 0.55  # NLI "entailment" prob to accept a claim
    contradiction_flag_threshold: float = 0.40  # NLI "contradiction" prob to flag a conflict

    # --- Redaction (security/redaction.py) ---
    # Rules file is JSON; see redaction_rules.example.json. Built-in rules (SSN,
    # payment card) always apply. Set REDACTION=off ONLY for local debugging.
    redaction_enabled: bool = os.getenv("REDACTION", "on").lower() != "off"
    redaction_rules_file: str = os.getenv("REDACTION_RULES_FILE", "redaction_rules.json")
    # Portion-marking abbreviations recognised at the start of a paragraph when an
    # upload is flagged "portion marked", e.g. "(S) The pump ..." -> SECRET.
    portion_marks: dict = field(
        default_factory=lambda: {
            "U": "UNCLASSIFIED", "C": "CONFIDENTIAL", "S": "SECRET", "TS": "TOP_SECRET",
        }
    )

    # --- Cross-source conflict detection (verification/conflict_detector.py) ---
    conflict_detection: bool = os.getenv("CONFLICT_DETECTION", "on").lower() != "off"
    conflict_contradiction_threshold: float = 0.70  # stricter than the per-claim 0.40
    conflict_min_overlap: float = 0.30       # word overlap needed before two sentences are compared
    conflict_numeric_overlap: float = 0.60   # (letters-only) overlap needed for the numeric-mismatch rule
    conflict_max_pairs: int = 24             # NLI budget per question (CPU-friendly)
    conflict_max_sentences_per_source: int = 40
    conflict_max_reported: int = 5
    conflict_cross_doc_only: bool = True     # only compare passages from different documents

    # --- Entity resolution (entity_resolution/) ---
    #   ENTITY_RESOLUTION=off disables both ingest-time extraction and query-time use.
    entity_resolution: bool = os.getenv("ENTITY_RESOLUTION", "on").lower() != "off"
    entity_min_score: float = 0.50           # GLiNER confidence to keep a mention
    entity_max_per_chunk: int = 40
    entity_live_fallback_max: int = 6        # chunks with no stored entities NER'd at query time
    entity_max_aliases_in_query: int = 5     # cap on alias terms used to expand a query

    # --- Web interface ---
    users_file: str = os.getenv("USERS_FILE", "users.json")
    upload_dir: str = os.getenv("UPLOAD_DIR", "data/uploads")
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "50"))


CFG = Config()


def clearance_rank(level: str, cfg: Config = CFG) -> int:
    """Higher index = higher clearance. Used for the mandatory retrieval filter."""
    try:
        return cfg.classification_levels.index(level)
    except ValueError:
        raise ValueError(f"Unknown classification level: {level}")
