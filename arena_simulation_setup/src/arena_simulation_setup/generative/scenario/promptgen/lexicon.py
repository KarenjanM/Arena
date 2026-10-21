"""Step 1 - Type A by lexicon. Deterministic, offline, and round-trip tested by the prompt sets."""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from .spec import A_CASES, PHRASES, ADecision, When

_LEXICON = Path(__file__).with_name("lexicon.yaml")

#: The score a case must reach to be applied.
THRESHOLD = 1.0

ONSETS: dict[str, When] = {
    "blackout": When(at=5.0),
    "evacuation": When(at=4.0),
    "stampede": When(at=2.0),
    "busy": When(at=0.0),
}


@cache
def load(path: str | None = None) -> dict[str, Any]:
    doc = yaml.safe_load(Path(path or _LEXICON).read_text()) or {}
    unknown = sorted(set(doc) - set(A_CASES))
    if unknown:
        raise ValueError(f"lexicon: unknown case(s) {unknown}")
    return doc


@cache
def _pattern(phrase: str) -> re.Pattern[str]:
    """Whole words of `phrase`, a trailing `*` lets the last word run on."""
    stem = phrase.lower().removesuffix("*")
    tail = r"\w*" if phrase.endswith("*") else r"(?!\w)"
    return re.compile(rf"(?<!\w){re.escape(stem)}{tail}")


def _find(phrase: object, text: str) -> int:
    hit = _pattern(str(phrase)).search(text)
    return hit.start() if hit else -1


def _has(phrases: Sequence[str], text: str) -> bool:
    return any(_find(p, text) >= 0 for p in phrases)


def _time_phrase(sentence: str) -> When | None:
    for phrase, when in sorted(PHRASES, key=lambda pw: -len(pw[0])):
        if _find(phrase, sentence) >= 0:
            return when
    return None


def _without_time_phrases(text: str) -> str:
    for phrase, _when in sorted(PHRASES, key=lambda pw: -len(pw[0])):
        text = _pattern(phrase).sub(" ", text)
    return text


def score(prompt: str, lexicon: dict[str, Any] | None = None) -> dict[str, tuple[float, int]]:
    """Case -> (total score, position of its earliest hit)."""
    text = " ".join(prompt.lower().split())
    lex = lexicon or load()
    out: dict[str, tuple[float, int]] = {}
    for case, entry in lex.items():
        total, first = 0.0, len(text) + 1
        for phrase, weight in (entry.get("phrases") or {}).items():
            pos = _find(phrase, text)
            if pos >= 0:
                total += float(weight)
                first = min(first, pos)
        if total > 0:
            out[case] = (total, first)
    return out


def _onset(prompt: str, default: When, case: str, lexicon: dict[str, Any]) -> When:
    """An onset phrase counts only in the sentence that names the case: "halfway" in a later
    sentence about a crossing pedestrian is that pedestrian's, not the evacuation's."""
    phrases = [str(p) for p in (lexicon.get(case, {}).get("phrases") or {})]
    for sentence in re.split(r"(?<=[.!?])\s+", prompt.lower()):
        if not _has(phrases, sentence):
            continue
        when = _time_phrase(sentence)
        if when is not None:
            return when
    return default


def when_in(text: str, *, exclude_cases: Sequence[str] = (), lexicon: dict[str, Any] | None = None) -> When | None:
    """The first time phrase in a sentence of `text` that does not name one of `exclude_cases`
    (their sentences own their phrases). Longest phrase first. None when there is none."""
    lex = lexicon or load()
    owned = [str(p) for case in exclude_cases for p in (lex.get(case, {}).get("phrases") or {})]
    for sentence in re.split(r"(?<=[.!?])\s+", text.lower()):
        if _has(owned, sentence):
            continue
        when = _time_phrase(sentence)
        if when is not None:
            return when
    return None


def match(prompt: str, lexicon: dict[str, Any] | None = None) -> ADecision | None:
    """The Type A case a prompt asks for, or None."""
    lex = lexicon or load()
    scores = score(prompt, lex)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1][0], kv[1][1], kv[0]))
    if not ranked or ranked[0][1][0] < THRESHOLD:
        return None
    case = ranked[0][0]
    k = 2.0
    if case == "busy":
        text = _without_time_phrases(" ".join(prompt.lower().split()))
        for word, value in (lex["busy"].get("k") or {}).items():
            if _find(word, text) >= 0:
                k = float(value)
                break
    return ADecision(
        case=case,
        k=k,
        when=_onset(prompt, ONSETS[case], case, lex),
        dropped=tuple(c for c, _ in ranked[1:] if scores[c][0] >= THRESHOLD),
        scores={c: s for c, (s, _pos) in scores.items()},
    )
