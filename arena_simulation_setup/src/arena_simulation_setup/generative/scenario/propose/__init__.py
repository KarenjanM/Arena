"""Model-proposed roles for the zone names the `zones` rules leave unknown, written as facts into the world's `kg.yaml`."""

from __future__ import annotations

from .client import DEFAULT_MODEL, Unavailable
from .zone_roles import build_prompt, classify_tail, merge, unknown_names

__all__ = [
    "DEFAULT_MODEL",
    "Unavailable",
    "build_prompt",
    "classify_tail",
    "merge",
    "unknown_names",
]
