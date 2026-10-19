"""Per-episode case records, appended as JSON Lines so a run that dies halfway still leaves usable records."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import attrs


@attrs.frozen
class CaseRecord:
    run_seed: str
    episode_id: int
    world: str
    seed: int

    scenario: str = ""
    case_id: str = ""
    base: str = ""
    prompt_id: str = ""
    steps: tuple[str, ...] = ()

    effects: tuple[Mapping[str, Any], ...] = ()
    plans: tuple[Mapping[str, Any], ...] = ()

    injected: int = 0
    target_agent: str = ""
    inject_type: str = ""
    inject_model: str = ""
    waypoint_mode: str = ""
    place: str = ""
    after: str = ""
    ped_speed: float | None = None

    approach_angle_requested: float | None = None
    approach_angle_achieved: float | None = None
    encounter_at_requested: float | None = None
    encounter_at_achieved: float | None = None
    encounter_point: tuple[float, float] | None = None
    t_encounter: float | None = None
    designed_ttc: float | None = None
    designed_pet: float | None = None
    co_arrival_offset: float | None = None
    lead_out_achieved: float | None = None
    spawn_pose: tuple[float, float] | None = None
    ped_goal: tuple[float, float] | None = None
    robot_leg: tuple[tuple[float, float], tuple[float, float]] | None = None
    robot_route: tuple[tuple[float, float], ...] | None = None
    route_length: float | None = None
    route_planned: bool | None = None
    geometry_retries: int | None = None
    robot_hold: float = 0.0
    # seconds before the robot is due at the encounter point that the walker leaves its spawn
    release_eta: float | None = None
    traversals_seen: int | None = None

    invalid_spawns: tuple[str, ...] = ()

    object_timeline: str = ""
    object_events: tuple[Mapping[str, Any], ...] = ()

    status: str = "ok"
    reason: str = ""

    created_at: float = attrs.Factory(time.time)

    def to_json(self) -> str:
        return json.dumps(attrs.asdict(self), separators=(",", ":"), sort_keys=True, default=str)


def default_record_dir() -> Path:
    data_dir = os.environ.get("ARENA_DATA_DIR")
    if data_dir:
        return Path(data_dir) / "edge_case"
    return Path(tempfile.gettempdir()) / "arena_edge_case"


class CaseLog:
    def __init__(self, record_dir: Path | str | None = None) -> None:
        self._dir = Path(record_dir) if record_dir else default_record_dir()
        self._path = self._dir / "cases.jsonl"

    @property
    def path(self) -> Path:
        return self._path

    def write(self, record: CaseRecord) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        with self._path.open("a") as fh:
            fh.write(record.to_json() + "\n")
        return self._path
