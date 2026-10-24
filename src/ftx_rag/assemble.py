"""Phase 2: build the LightRAG graph from saved extractions. No LLM calls.

Each extraction file becomes one chunk plus its entities and relationships, fed
to `LightRAG.ainsert_custom_kg`, which embeds them and writes the graph.

Two things make this fast and robust:

- Merged batches. Every `ainsert_custom_kg` call ends by persisting the whole
  graph, so the cost of a call grows with the graph, and one call per chunk is
  quadratic overall. Merging ~1,000 extractions per call pays that cost once
  per batch. It is safe: duplicate node/edge declarations within a call merge
  the same way they would across calls.
- Sanitising first. LightRAG validates a merged batch as a whole, so a single
  bad entity name rejects all 1,000 extractions in it. Names are normalised
  with LightRAG's own function up front, and anything it would reject is
  dropped here instead.
"""

from __future__ import annotations

import json
import logging
import time

from . import config

logger = logging.getLogger(__name__)

# LightRAG's Postgres schema stores entity names as VARCHAR(512), and the
# vector upsert is one transaction per flush, so one longer name silently
# loses the whole flush. Names that long are table rows the model read as one
# entity anyway.
MAX_NAME_CHARS = 256


def normalize_name(name) -> str:
    """LightRAG's entity-name normalisation; '' for anything it would reject."""
    from lightrag.utils import normalize_entity_name

    if not isinstance(name, str) or not name.strip():
        return ""
    name = normalize_entity_name(name)
    return name if len(name) <= MAX_NAME_CHARS else ""


def to_custom_kg(extraction: dict) -> dict:
    """One extraction file -> the dict shape `ainsert_custom_kg` expects.

    Every entity and relationship carries the chunk's content hash as
    `source_id`, which is how LightRAG links them back to the chunk, and the
    chunk keeps `doc_id` as `file_path` so a retrieved passage can be traced to
    its filing.
    """
    ch = extraction["content_hash"]
    doc_id = extraction.get("doc_id") or "unknown"
    # A small fraction of model replies put a scalar or null where a list
    # belongs; treat those as empty instead of failing the batch.
    raw_entities = extraction.get("entities")
    raw_rels = extraction.get("relationships")

    entities = []
    for e in raw_entities if isinstance(raw_entities, list) else []:
        if not isinstance(e, dict) or not (name := normalize_name(e.get("entity_name"))):
            continue
        entities.append({
            "entity_name": name,
            "entity_type": e.get("entity_type") or "LEGAL_PARTY",
            "description": e.get("description") or "No description provided",
            "source_id": ch,
        })

    relationships = []
    for r in raw_rels if isinstance(raw_rels, list) else []:
        if not isinstance(r, dict):
            continue
        src, tgt = normalize_name(r.get("src_id")), normalize_name(r.get("tgt_id"))
        if not src or not tgt:
            continue
        relationships.append({
            "src_id": src,
            "tgt_id": tgt,
            "description": r.get("description") or "",
            "keywords": src,
            "source_id": ch,
            "weight": 1.0,
            "file_path": doc_id,
        })

    return {
        "chunks": [{"content": extraction["text"], "source_id": ch, "file_path": doc_id}],
        "entities": entities,
        "relationships": relationships,
    }


def merge(kgs: list[dict]) -> dict:
    merged: dict = {"chunks": [], "entities": [], "relationships": []}
    for kg in kgs:
        for key in merged:
            merged[key].extend(kg[key])
    return merged


def load_extractions() -> list[dict]:
    out = []
    for f in sorted(config.EXTRACTS_DIR.glob("*.json")):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("skipping %s: %s", f.name, e)
    return out


async def run(batch_size: int = 1000) -> dict:
    from .kb import KnowledgeBase

    extractions = load_extractions()
    if not extractions:
        return {"assembled": 0, "failed": 0}
    kb = KnowledgeBase()
    await kb.initialize()

    stats = {"assembled": 0, "failed": 0}
    t0 = time.monotonic()
    for i in range(0, len(extractions), batch_size):
        kgs = [to_custom_kg(e) for e in extractions[i:i + batch_size]]
        try:
            await kb.rag.ainsert_custom_kg(merge(kgs))
            stats["assembled"] += len(kgs)
        except Exception as e:
            # Fall back to one call per extraction so a single poison record
            # cannot keep the rest of its batch out of the graph.
            logger.warning("batch at %d failed (%s); retrying one by one", i, e)
            for kg in kgs:
                try:
                    await kb.rag.ainsert_custom_kg(kg)
                    stats["assembled"] += 1
                except Exception as e2:
                    stats["failed"] += 1
                    logger.warning("insert failed: %s", e2)
        logger.info("%d/%d assembled in %.0fs", i + len(kgs), len(extractions),
                    time.monotonic() - t0)
    await kb.finalize()
    return stats
