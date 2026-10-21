"""One step's model question: its system prompt, at most two answers, the second told why the first was rejected."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from ..propose.client import prompt_sha256
from .llm import AnswerError, Oracle, call
from .spec import ConflictError, FailureKind, Record, SchemaError
from .step_a import StepError

T = TypeVar("T")


def system_prompt(step: str) -> str:
    """The built `system_<step>.md` beside this module."""
    return Path(__file__).with_name(f"system_{step}.md").read_text()


def ask(step: str, system: str, user: str, record: Record, parse: Callable[[dict[str, Any]], T], *, cache_dir: Path | None, model: str | None, oracle: Oracle | None = None) -> T:
    """`parse` of the answer to `user`, asked once more with the rejection, the hashes logged in `record.provenance[step]`."""
    complaint = ""
    kind: FailureKind = "schema"
    for attempt in range(2):
        asked = user + (f"\n\nYour previous answer was rejected: {complaint}. Answer again." if complaint else "")
        answer = call(system, asked, cache_dir=cache_dir, model=model, oracle=oracle)
        record.provenance.setdefault(step, []).append({"model": answer.model, "prompt_sha256": prompt_sha256(system, asked), "response_sha256": answer.sha256, "cached": answer.cached, "attempt": attempt})
        try:
            return parse(answer.json())
        except (AnswerError, SchemaError) as exc:
            complaint = str(exc)
            kind = "conflict" if isinstance(exc, ConflictError) else "schema"
    raise StepError(kind, complaint)
