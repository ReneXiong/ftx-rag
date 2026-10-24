"""Docket classification: title rules first, LLM only for what they miss."""

from __future__ import annotations

import logging

import httpx

from . import llm
from .config import llm_settings
from .rules import match_bulk_rule

logger = logging.getLogger(__name__)

CATEGORIES = (
    "Claims Trading|Financial Reports|Procedural|Financial Disclosures|"
    "Disputes/Litigation|Hearing Notices|Court Rulings|Exhibits|Settlements|"
    "Plan/Reorganization|Other"
)

PROMPT = """Classify this FTX bankruptcy docket entry. Return ONLY valid JSON.

Document Name: {title}
Date Filed: {date_filed}

Return JSON with these keys:
{{
    "category": "pick ONE from: {categories}",
    "subcategory": "specific sub-type (2-6 words)",
    "entity": "primary entity name or empty string",
    "one_line_summary": "one sentence summary in English"
}}

Rules:
- Pick exactly one category; never combine them with | or /.
- If unsure, use "Other".

Examples:
- Monthly Operating Report -> "Financial Reports" / "Monthly Operating Report" / "FTX Japan K.K." / "FTX Japan K.K. filed its March 2026 monthly operating report"
- Objection to Motion -> "Disputes/Litigation" / "Objection" / "<objecting party>" / "<summary>"
"""


def _fallback(title: str) -> dict:
    return {"category": "Other", "subcategory": "unknown", "entity": "",
            "one_line_summary": title[:100]}


class DocketClassifier:
    def __init__(self, timeout: float = 60):
        self.settings = llm_settings()
        self.client = httpx.Client(timeout=timeout)

    def classify_with_llm(self, title: str, date_filed: str, retries: int = 2) -> dict:
        messages = [{"role": "user", "content": PROMPT.format(
            title=title, date_filed=date_filed, categories=CATEGORIES)}]
        for attempt in range(retries + 1):
            try:
                result = llm.parse_json_reply(llm.chat(
                    self.client, self.settings, messages, max_tokens=256, temperature=0.1))
                # Models sometimes return "A|B" despite the instruction.
                result["category"] = str(result.get("category", "Other")).split("|")[0].strip()
                return result
            except Exception as e:
                logger.warning("classification attempt %d failed for %r: %s",
                               attempt + 1, title[:60], e)
        return _fallback(title)

    def classify(self, entry: dict) -> dict:
        """Return a classification dict. `bulk=True` marks rule-routed filings."""
        title = entry.get("document_name", "")
        rule = match_bulk_rule(title)
        if rule:
            category, subcategory = rule
            return {"category": category, "subcategory": subcategory, "entity": "",
                    "one_line_summary": title[:100], "bulk": True}
        return {**self.classify_with_llm(title, entry.get("date_filed", "")), "bulk": False}
