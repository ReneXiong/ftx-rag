"""Hybrid retrieval over LightRAG's chunk table: pgvector + Postgres full-text.

Dense embeddings smear rare literals — dollar amounts, dates, party names —
and those are exactly what a docket lookup turns on. Full-text search matches
them exactly but misses paraphrase. On the eval set each retriever finds
filings the other never surfaces, so both run and their rankings are merged
with reciprocal rank fusion. No LLM is involved.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from . import config

logger = logging.getLogger(__name__)

MODES = ("dense", "lexical", "hybrid")
DEPTH = 50
RRF_K = 60
FTS_CONFIG = "english"
TSV_COLUMN = "content_tsv"
TABLE_DIM = re.compile(r"_(\d+)d$")


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    file_path: str
    content: str


def rrf(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> list[str]:
    """Reciprocal rank fusion: score each id by sum(1 / (k + rank)) over rankings.

    Only ranks are used, so a cosine distance and a ts_rank never have to be
    put on the same scale. Every id from every ranking is kept.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, 1):
            scores[key] = scores.get(key, 0.0) + 1 / (k + rank)
    return sorted(scores, key=lambda key: -scores[key])


def connect():
    import psycopg2

    return psycopg2.connect(config.PG_DSN)


def chunk_tables(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT tablename FROM pg_tables WHERE schemaname='public' "
                    "AND tablename LIKE 'lightrag_vdb_chunks%' ORDER BY tablename")
        return [r[0] for r in cur.fetchall()]


def resolve_table(conn, table: str | None = None) -> str:
    tables = chunk_tables(conn)
    if not tables:
        raise SystemExit("no lightrag_vdb_chunks* tables found — run `ftx-rag assemble` first")
    if table and table not in tables:
        raise SystemExit(f"{table} not found; available: {', '.join(tables)}")
    table = table or tables[0]
    # LightRAG names vector tables after the embedding dimension. A query
    # vector from a different model is meaningless against them, so refuse
    # rather than compare across embedding spaces.
    m = TABLE_DIM.search(table)
    if m and int(m.group(1)) != config.EMBEDDING_DIM:
        raise SystemExit(f"{table} holds {m.group(1)}-dim vectors but EMBEDDING_DIM is "
                         f"{config.EMBEDDING_DIM}; set EMBEDDING_MODEL/EMBEDDING_DIM to match")
    return table


def ensure_fts(conn, table: str) -> None:
    """Add a stored tsvector column and a GIN index to LightRAG's chunk table.

    Idempotent. A generated column, rather than an expression index, means
    ts_rank reads a precomputed vector instead of re-parsing every matching
    chunk on each query, and LightRAG's own inserts keep it current.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM information_schema.columns "
                    "WHERE table_name = %s AND column_name = %s", (table, TSV_COLUMN))
        if cur.fetchone():
            return
        logger.info("building full-text index on %s (one-off)", table)
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {TSV_COLUMN} tsvector GENERATED ALWAYS "
                    f"AS (to_tsvector('{FTS_CONFIG}', coalesce(content, ''))) STORED")
        cur.execute(f"CREATE INDEX IF NOT EXISTS {table}_fts ON {table} "
                    f"USING gin ({TSV_COLUMN})")
    conn.commit()


def load_embedder() -> Callable[[str], np.ndarray]:
    import torch
    from sentence_transformers import SentenceTransformer

    from .kb import mrl_normalize

    device = config.torch_device()
    model = SentenceTransformer(config.EMBEDDING_MODEL, device=device,
                                model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = config.EMBEDDING_MAX_SEQ_LEN

    def embed(text: str) -> np.ndarray:
        raw = model.encode([text], convert_to_numpy=True, normalize_embeddings=False)
        return mrl_normalize(raw, config.EMBEDDING_DIM)[0]

    return embed


class Searcher:
    def __init__(self, conn, table: str, embed: Callable[[str], np.ndarray] | None = None):
        self.conn = conn
        self.table = table
        self._embed = embed

    def _rows(self, sql: str, params: tuple) -> list[Hit]:
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return [Hit(*r) for r in cur.fetchall()]

    def dense(self, query: str, n: int = DEPTH) -> list[Hit]:
        if self._embed is None:
            self._embed = load_embedder()
        literal = "[" + ",".join(f"{x:.7f}" for x in self._embed(query)) + "]"
        return self._rows(f"SELECT id, file_path, content FROM {self.table} "
                          f"ORDER BY content_vector <=> %s::vector LIMIT %s", (literal, n))

    def lexical(self, query: str, n: int = DEPTH) -> list[Hit]:
        # plainto_tsquery ANDs every term, and a question rarely shares all of
        # its words with the passage that answers it. Rewriting the operators
        # to OR keeps Postgres's own tokenising, stemming and stop words.
        return self._rows(
            f"SELECT id, file_path, content FROM {self.table}, "
            f"CAST(replace(plainto_tsquery('{FTS_CONFIG}', %s)::text, '&', '|') AS tsquery) q "
            f"WHERE {TSV_COLUMN} @@ q ORDER BY ts_rank({TSV_COLUMN}, q) DESC LIMIT %s",
            (query, n))

    def hybrid(self, query: str, n: int = DEPTH) -> list[Hit]:
        """Up to 2n hits: both candidate lists in full, ordered by RRF."""
        by_id = {}
        rankings = []
        for hits in (self.dense(query, n), self.lexical(query, n)):
            by_id.update((h.chunk_id, h) for h in hits)
            rankings.append([h.chunk_id for h in hits])
        return [by_id[i] for i in rrf(rankings)]

    def search(self, query: str, mode: str = "hybrid", n: int = DEPTH) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        return getattr(self, mode)(query, n)


def open_searcher(table: str | None = None) -> Searcher:
    conn = connect()
    table = resolve_table(conn, table)
    ensure_fts(conn, table)
    return Searcher(conn, table)
