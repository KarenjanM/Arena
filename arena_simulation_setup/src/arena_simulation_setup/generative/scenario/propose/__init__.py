"""Model-proposed roles for the zone names the `zones` rules leave unknown, written as facts into the world's `kg.yaml`."""

from __future__ import annotations

from .client import DEFAULT_MODEL, Unavailable, available, prompt_sha256
from .zone_roles import (
    Proposal,
    build_prompt,
    classify_tail,
    merge,
    parse,
    unknown_names,
)

__all__ = [
    "DEFAULT_MODEL",
    "Proposal",
    "Unavailable",
    "available",
    "build_prompt",
    "classify_tail",
    "merge",
    "parse",
    "prompt_sha256",
    "unknown_names",
]
