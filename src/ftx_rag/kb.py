"""LightRAG knowledge base: local Qwen3 embedder + reranker, pgvector storage."""

from __future__ import annotations

import asyncio
import logging
import os

import numpy as np

from . import config

logger = logging.getLogger(__name__)


def mrl_normalize(raw: np.ndarray, dim: int) -> np.ndarray:
    """Matryoshka truncation to `dim`, then L2-normalise — in that order.

    Normalising first and truncating second leaves vectors shorter than 1,
    which quietly distorts every cosine distance computed from them.
    """
    cut = raw[:, :dim]
    return cut / np.linalg.norm(cut, axis=1, keepdims=True)


class KnowledgeBase:
    def __init__(self, working_dir: str | os.PathLike = config.LIGHTRAG_DIR):
        import torch
        from lightrag import LightRAG
        from lightrag.llm.openai import openai_complete_if_cache
        from lightrag.utils import EmbeddingFunc
        from sentence_transformers import SentenceTransformer

        self.device = config.torch_device()
        logger.info("loading %s on %s", config.EMBEDDING_MODEL, self.device)
        self._embedder = SentenceTransformer(
            config.EMBEDDING_MODEL, device=self.device,
            model_kwargs={"torch_dtype": torch.float16})
        self._embedder.max_seq_length = config.EMBEDDING_MAX_SEQ_LEN
        # The reranker is only used at query time. Loading it here would hold
        # ~8 GB through graph assembly, which never calls it.
        self._reranker = None

        async def embed(texts: list[str]) -> np.ndarray:
            raw = self._embedder.encode(
                texts, batch_size=config.EMBEDDING_BATCH_SIZE, convert_to_numpy=True,
                normalize_embeddings=False, show_progress_bar=False)
            return mrl_normalize(raw, config.EMBEDDING_DIM)

        async def rerank(query: str, documents: list[str], **_) -> list[dict]:
            if not config.RERANKER_ENABLED or not documents:
                return []
            scores = self._get_reranker().predict(
                [(query, d) for d in documents], show_progress_bar=False)
            return [{"index": i, "relevance_score": float(s)} for i, s in enumerate(scores)]

        llm = config.llm_settings()
        llm_slots = asyncio.Semaphore(4)

        async def complete(prompt, system_prompt=None, history_messages=None, **kwargs):
            if llm.extra_body:
                kwargs["extra_body"] = {**llm.extra_body, **kwargs.get("extra_body", {})}
            async with llm_slots:
                return await openai_complete_if_cache(
                    llm.model, prompt, system_prompt=system_prompt,
                    history_messages=history_messages or [],
                    api_key=llm.api_key or "not-needed", base_url=llm.base_url, **kwargs)

        # LightRAG's Postgres storages read their connection from the environment.
        os.environ.setdefault("POSTGRES_HOST", config.PG_HOST)
        os.environ.setdefault("POSTGRES_PORT", str(config.PG_PORT))
        os.environ.setdefault("POSTGRES_USER", config.PG_USER)
        os.environ.setdefault("POSTGRES_PASSWORD", config.PG_PASSWORD)
        os.environ.setdefault("POSTGRES_DATABASE", config.PG_DATABASE)

        os.makedirs(working_dir, exist_ok=True)
        self.rag = LightRAG(
            working_dir=str(working_dir),
            llm_model_func=complete,
            embedding_func=EmbeddingFunc(embedding_dim=config.EMBEDDING_DIM, func=embed,
                                         model_name=config.EMBEDDING_MODEL),
            rerank_model_func=rerank,
            graph_storage="NetworkXStorage",
            vector_storage="PGVectorStorage",
            kv_storage="PGKVStorage",
            doc_status_storage="PGDocStatusStorage",
            chunk_token_size=1200,
        )

    def _get_reranker(self):
        if self._reranker is None:
            import torch
            from sentence_transformers import CrossEncoder

            logger.info("loading %s on %s", config.RERANKER_MODEL, self.device)
            self._reranker = CrossEncoder(config.RERANKER_MODEL, device=self.device,
                                          model_kwargs={"torch_dtype": torch.float16})
            # Qwen3-Reranker ships without pad_token_id on the model config,
            # which breaks batched scoring.
            self._reranker.model.config.pad_token_id = self._reranker.tokenizer.pad_token_id
        return self._reranker

    async def initialize(self) -> None:
        await self.rag.initialize_storages()

    async def finalize(self) -> None:
        await self.rag.finalize_storages()

    async def query(self, question: str, mode: str = "hybrid",
                    top_k: int = config.RAG_TOP_K) -> str:
        from lightrag import QueryParam

        return await self.rag.aquery(question, param=QueryParam(
            mode=mode, top_k=top_k, max_total_tokens=8192,
            max_entity_tokens=2000, max_relation_tokens=2000))
