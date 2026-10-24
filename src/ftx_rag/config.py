"""Runtime configuration, read from the environment (and `.env` if present)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
# Everything the pipeline reads or writes lives under one directory.
DATA_DIR = Path(os.getenv("FTX_RAG_DATA_DIR", "data"))
METADATA_FILE = DATA_DIR / "metadata.jsonl"      # input: one docket entry per line
CLASSIFIED_FILE = DATA_DIR / "classified.jsonl"  # output of `process`
PARSED_DIR = DATA_DIR / "parsed"                 # Docling JSON, one file per PDF
EXTRACTS_DIR = DATA_DIR / "extracts"             # Phase-1 output, one file per chunk
LIGHTRAG_DIR = DATA_DIR / "lightrag"             # LightRAG working dir (graph)
EVAL_DIR = DATA_DIR / "eval"

# ── LLM (any OpenAI-compatible endpoint) ───────────────────────────────
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:8b")
# Provider-specific request fields, as JSON. Reasoning models are worth
# switching off for extraction: they spend thousands of tokens thinking about a
# small JSON answer and often truncate it. How to switch it off differs by
# server, e.g. '{"chat_template_kwargs": {"enable_thinking": false}}' for
# vLLM / llama.cpp, or '{"thinking": {"type": "disabled"}}' for some gateways.
LLM_EXTRA_BODY: dict = json.loads(os.getenv("LLM_EXTRA_BODY") or "{}")


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    api_key: str
    model: str
    extra_body: dict = field(default_factory=dict)

    @property
    def chat_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    @property
    def headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h


def llm_settings() -> LLMSettings:
    return LLMSettings(LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_EXTRA_BODY)


# ── Embedding & reranking ──────────────────────────────────────────────
# Qwen3-Embedding-4B emits 2560 dims. pgvector's `vector` type caps HNSW
# indexes at 2000, so the default MRL-truncates to 1536. To keep all 2560,
# set POSTGRES_VECTOR_INDEX_TYPE=HNSW_HALFVEC and EMBEDDING_DIM=2560
# (halfvec indexes up to 4000 dims, and fp16 at 2560 is smaller than fp32 at 1536).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
EMBEDDING_MAX_SEQ_LEN = 384   # chunk tokens: p95 ≈ 428, so 384 truncates ~5%
EMBEDDING_BATCH_SIZE = 16

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Qwen/Qwen3-Reranker-4B")
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "1") == "1"
RAG_TOP_K = 5

# ── PostgreSQL + pgvector ──────────────────────────────────────────────
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_USER = os.getenv("PG_USER", "ftxrag")
PG_PASSWORD = os.getenv("PG_PASSWORD", "ftxrag_dev")
PG_DATABASE = os.getenv("PG_DATABASE", "ftxrag")
PG_DSN = os.getenv(
    "DATABASE_URL",
    f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}",
)

# ── Graph extraction ───────────────────────────────────────────────────
ENTITY_TYPES = [
    "LEGAL_PARTY",
    "CASE_NUMBER",
    "CLAIM_AMOUNT",
    "DATE",
    "JUDGE",
    "LAW_FIRM",
    "CREDITOR",
    "BUYER",
]

# Only the head of a document goes into the graph. The largest filings are
# omnibus claim objections: the ruling is on the first pages, followed by tens
# of thousands of rows of creditor schedules. Extracting entities from those
# rows is the most expensive part of a run and floods the graph with
# near-duplicate CREDITOR nodes. 0 disables the cap.
MAX_GRAPH_DOC_CHARS = int(os.getenv("MAX_GRAPH_DOC_CHARS", "400000"))
CHUNK_CHARS = 1000


def torch_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
