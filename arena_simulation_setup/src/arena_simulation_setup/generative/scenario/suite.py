"""Scenario directories as a benchmark suite: one `scenario` stage per (world, scenario)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from .materialize import is_generated


def scenarios(world_root: Path, match: Sequence[str] = ()) -> list[str]:
    """Scenarios matching any glob in `match`; without one, every scenario the generator wrote."""
    paths = sorted((world_root / "scenarios").glob("*/scenario.yaml"))
    if match:
        return [p.parent.name for p in paths if any(fnmatch(p.parent.name, m) for m in match)]
    return [p.parent.name for p in paths if is_generated(p)]


def build(worlds: Mapping[str, Sequence[str]], *, robot: str, episodes: int, seed: int, timeout: str) -> dict[str, Any]:
    return {
        "launch": {"human": "arena"},
        "stages": [
            {
                "name": f"{world}.{scenario}",
                "map": world,
                "robot": robot,
                "episodes": episodes,
                "seed": seed,
                "timeout": timeout,
                "tm_robots": "scenario",
                "tm_obstacles": "scenario",
                "config": {"scenario": {"file": scenario}},
            }
            for world, names in worlds.items()
            for scenario in names
        ],
    }
