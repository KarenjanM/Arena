"""One model call per step at temperature 0, cached as one JSON file per `prompt_sha256(model, system, user)`."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import attrs

from ..propose.client import DEFAULT_MODEL, Unavailable, ask, prompt_sha256, strip_fence, text_sha256

DEFAULT_CACHE = Path(os.environ.get("ARENA_DATA_DIR") or Path(os.environ.get("ARENA_WS_DIR", "/opt/arena_ws")) / "data") / "promptgen" / "llm_cache"


class AnswerError(ValueError):
    pass


@attrs.frozen
class Answer:
    text: str
    sha256: str
    model: str
    cached: bool

    def json(self) -> dict[str, Any]:
        text = strip_fence(self.text)
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise AnswerError(f"no JSON object in the answer: {text[:200]!r}")
        try:
            doc = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise AnswerError(f"answer is not valid JSON ({exc}): {text[:200]!r}") from None
        if not isinstance(doc, dict):
            raise AnswerError("answer must be a JSON object")
        return doc


class Oracle:
    """Stand-in for the model: a fixed answer or a `(system, user)` callable, never cached."""

    def __init__(self, answer: object) -> None:
        self._answer = answer

    def __call__(self, system: str, user: str) -> str:
        value = self._answer(system, user) if callable(self._answer) else self._answer
        return value if isinstance(value, str) else json.dumps(value)


def _read_cached(path: Path, name: str) -> Answer | None:
    """The cached answer at `path`, or None when it is absent or unreadable."""
    try:
        doc = json.loads(path.read_text())
        text = str(doc["answer"])
        model = str(doc.get("model", name))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return None
    return Answer(text=text, sha256=text_sha256(text), model=model, cached=True)


def _write_cached(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    partial.write_text(json.dumps(doc, indent=1))
    os.replace(partial, path)


def call(system: str, user: str, *, cache_dir: Path | None = None, model: str | None = None, oracle: Oracle | None = None) -> Answer:
    """One completion, cached. Raises `Unavailable` when no model is configured and no oracle given."""
    name = model or DEFAULT_MODEL
    key = prompt_sha256(name, system, user)
    root = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    path = root / f"{key}.json"
    if oracle is None and (cached := _read_cached(path, name)) is not None:
        return cached
    if oracle is not None:
        text = oracle(system, user)
        return Answer(text=text, sha256=text_sha256(text), model="oracle", cached=False)
    reply = ask(system, user, model=name, temperature=0.0)
    _write_cached(path, {"model": name, "system_sha256": prompt_sha256(system), "user": user, "answer": reply.text, "usage": reply.usage})
    return Answer(text=reply.text, sha256=text_sha256(reply.text), model=name, cached=False)


__all__ = ["DEFAULT_CACHE", "Answer", "AnswerError", "Oracle", "Unavailable", "call"]
