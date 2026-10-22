"""Synchronous model calls through the genai client of `arena_simulation_setup.utils.llm`."""

from __future__ import annotations

import hashlib
import os
import typing

import attrs

from arena_simulation_setup.utils.llm import DEFAULT_MODEL

if typing.TYPE_CHECKING:
    from google import genai as genai_module


class Unavailable(RuntimeError):
    pass


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def prompt_sha256(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _require_llm() -> None:
    if not os.environ.get("LLM_API_ENDPOINT") and not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        raise Unavailable("no LLM configured: set GEMINI_API_KEY (or GOOGLE_API_KEY) for the hosted API, or LLM_API_ENDPOINT to point at a local one")


_CLIENT: genai_module.Client | None = None

TIMEOUT_MS = 120_000


@attrs.frozen
class Reply:
    text: str
    usage: dict[str, int | None] | None = None


def ask(system: str, user: str, *, model: str | None = None, temperature: float = 0.0) -> Reply:
    """One completion as raw text with its token usage, `Unavailable` when the model gives none."""
    global _CLIENT
    import httpx

    _require_llm()
    try:
        from google.genai import errors as genai_errors

        from arena_simulation_setup.utils.llm import genai
    except ImportError as exc:
        raise Unavailable(f"the genai client is not importable: {exc}") from exc
    response = None
    for attempt in range(2):
        if _CLIENT is None:
            _CLIENT = genai.Client(http_options=genai.types.HttpOptions(timeout=TIMEOUT_MS))
        try:
            response = _CLIENT.models.generate_content(
                model=model or DEFAULT_MODEL,
                contents=[user],
                config=genai.types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=temperature,
                ),
            )
            break
        except (genai_errors.APIError, genai_errors.UnknownApiResponseError, httpx.HTTPError) as exc:
            _CLIENT = None
            if attempt:
                raise Unavailable(f"the model call failed: {type(exc).__name__}: {exc}") from exc
    if response is None:
        raise Unavailable("the model call returned nothing")
    text = response.text
    if not text:
        raise Unavailable("the model returned an empty response")
    meta = response.usage_metadata
    usage = (
        {
            "input_tokens": meta.prompt_token_count,
            "output_tokens": meta.candidates_token_count,
            "thoughts_tokens": meta.thoughts_token_count,
            "cached_tokens": meta.cached_content_token_count,
            "total_tokens": meta.total_token_count,
        }
        if meta is not None
        else None
    )
    return Reply(text=text.strip(), usage=usage)


def strip_fence(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    return text[:-3].strip() if text.endswith("```") else text.strip()
