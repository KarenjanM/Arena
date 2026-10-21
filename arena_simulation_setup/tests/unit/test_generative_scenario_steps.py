from __future__ import annotations

from pathlib import Path

import yaml
from arena_simulation_setup.generative.scenario.formats import read_scenario, write_scenario
from arena_simulation_setup.tree.World import Scenario as scenario_format

_LOBBY = [[0.0, 0.0], [6.0, 0.0], [6.0, 4.0], [0.0, 4.0]]
_WARD = [[6.0, 0.0], [10.0, 0.0], [10.0, 4.0], [6.0, 4.0]]
_HUNAV = {"behavior": 2, "goal_radius": 0.4}


def _world(tmp_path: Path) -> Path:
    root = tmp_path / "world"
    root.mkdir()
    zones = [{"name": "lobby", "polygon": _LOBBY, "floor": False}, {"name": "ward", "polygon": _WARD, "floor": False}]
    (root / "world.yaml").write_text(yaml.safe_dump({"format": "arena-world/2", "levels": [{"id": "0", "zones": zones}]}))
    return root


def _human(**extra: object) -> dict:
    return {"name": "walker_1", "model": "female_adult_medical_01", "pose": "lobby", "agent": {"agent_type": "adult"}, "waypoints": ["lobby", "ward"], **extra}


def test_format_2_hunav_round_trips_without_nesting(tmp_path: Path) -> None:
    world = _world(tmp_path)
    base = world / "scenarios" / "base"
    base.mkdir(parents=True)
    (base / "scenario.yaml").write_text(yaml.safe_dump({"format": scenario_format.SCENARIO_FORMAT, "humans": [_human(hunav=_HUNAV)]}))

    path = write_scenario(world / "scenarios" / "once", read_scenario(base / "scenario.yaml"), world_root=world)
    again = write_scenario(world / "scenarios" / "twice", read_scenario(path), world_root=world)

    assert read_scenario(again)["humans"][0]["hunav"] == _HUNAV


def test_format_1_extra_human_keys_nest_once_across_rewrites(tmp_path: Path) -> None:
    world = _world(tmp_path)
    base = world / "scenarios" / "base"
    base.mkdir(parents=True)
    (base / "scenario.yaml").write_text(yaml.safe_dump({"robots": [{"start": [1.0, 1.0, 0.0], "goal": [8.0, 2.0, 0.0]}], "dynamic": [_human(**_HUNAV)]}))

    path = write_scenario(world / "scenarios" / "once", read_scenario(base / "scenario.yaml"), world_root=world)
    again = write_scenario(world / "scenarios" / "twice", read_scenario(path), world_root=world)

    assert read_scenario(again)["humans"][0]["hunav"] == _HUNAV
