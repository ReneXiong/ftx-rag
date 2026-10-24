"""Minimal OpenAI-compatible chat client.

Plain HTTP rather than an SDK: every call site here sends one prompt and wants
one string back, and talking to the endpoint directly keeps provider-specific
fields (see `config.LLM_EXTRA_BODY`) a dict merge instead of an SDK workaround.
"""

from __future__ import annotations

import json

import httpx

from .config import LLMSettings


def _payload(settings: LLMSettings, messages: list[dict], **params) -> dict:
    return {**settings.extra_body, "model": settings.model, "messages": messages, **params}


def _content(r: httpx.Response) -> str:
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"] or "").strip()


def chat(client: httpx.Client, settings: LLMSettings, messages: list[dict], **params) -> str:
    r = client.post(settings.chat_url, headers=settings.headers,
                    json=_payload(settings, messages, **params))
    return _content(r)


async def achat(client: httpx.AsyncClient, settings: LLMSettings, messages: list[dict],
                **params) -> str:
    r = await client.post(settings.chat_url, headers=settings.headers,
                          json=_payload(settings, messages, **params))
    return _content(r)


def parse_json_reply(text: str) -> dict:
    """Parse a JSON object from a model reply, tolerating a ``` fence around it."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rsplit("```", 1)[0]
    result = json.loads(t)
    if not isinstance(result, dict):
        raise ValueError(f"expected a JSON object, got {type(result).__name__}")
    return result
