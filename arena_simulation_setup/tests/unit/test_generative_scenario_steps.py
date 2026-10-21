from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from arena_simulation_setup.generative.scenario.formats import read_scenario, write_scenario
from arena_simulation_setup.generative.scenario.promptgen import prompts as P
from arena_simulation_setup.generative.scenario.promptgen.pipeline import Result, generate
from arena_simulation_setup.generative.scenario.promptgen.world import WorldContext
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


_ROOMS = {"west": (0.0, 0.0, 8.0, 8.0), "hall": (8.0, 3.0, 24.0, 5.0), "east": (24.0, 0.0, 32.0, 8.0)}
_START = [2.0, 4.0, 0.0]
_GOAL = [30.0, 4.0, 0.0]


def _walker(name: str, home: str, away: str, agent_type: str = "adult") -> dict:
    return {"name": name, "model": "male_adult_medical_01", "pose": home, "agent": {"agent_type": agent_type, "desired_velocity": 1.1}, "waypoints": [home, away], "waypoint_mode": "reverse"}


def _corridor_world(tmp_path: Path, names: dict[str, str], robot: dict, humans: list[dict]) -> tuple[WorldContext, dict, Path]:
    """Two rooms joined by a 2 m corridor, each zone named by `names` (a missing key leaves that area unzoned)."""
    np = pytest.importorskip("numpy")
    from PIL import Image

    root = tmp_path / "worlds" / "corridor"
    (root / "map").mkdir(parents=True)
    zones = [{"name": names[key], "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "floor": False} for key, (x0, y0, x1, y1) in _ROOMS.items() if key in names]
    (root / "world.yaml").write_text(yaml.safe_dump({"format": "arena-world/2", "levels": [{"id": "0", "zones": zones}]}))
    res = 0.1
    px = np.zeros((int(8 / res), int(32 / res)), dtype=np.uint8)
    for x0, y0, x1, y1 in ((0.3, 0.3, 7.7, 7.7), (7.7, 3.0, 24.3, 5.0), (24.3, 0.3, 31.7, 7.7)):
        px[px.shape[0] - int(y1 / res) : px.shape[0] - int(y0 / res), int(x0 / res) : int(x1 / res)] = 255
    Image.fromarray(px, mode="L").save(root / "map" / "0.png")
    (root / "map" / "0.yaml").write_text(yaml.safe_dump({"image": "0.png", "resolution": res, "origin": [0.0, 0.0, 0.0], "occupied_thresh": 0.65, "negate": 0}))
    base = root / "scenarios" / "base"
    base.mkdir(parents=True)
    doc = {"robots": [robot], "humans": humans}
    (base / "scenario.yaml").write_text(yaml.safe_dump({"format": scenario_format.SCENARIO_FORMAT, **doc}, sort_keys=False))
    return WorldContext.load(root, doc), doc, base


def _run(world: WorldContext, doc: dict, base: Path, text: str, targets: list[str], **oracle: object) -> Result:
    prompt = P.Prompt(id="t_001", text=text, targets=tuple(targets), oracle=dict(oracle))
    return generate(prompt, world=world, base_id="base", base_doc=doc, base_dir=base, oracle=True)


def test_a_failed_stampede_leaves_no_injected_pedestrians(tmp_path: Path) -> None:
    crowd = [_walker("p_1", "reception", "open_work_area"), _walker("p_2", "open_work_area", "reception"), {**_walker("idle", "reception", "reception"), "waypoints": []}]
    world, doc, base = _corridor_world(tmp_path, {"west": "reception", "hall": "central_hallway", "east": "open_work_area"}, {"start": _START, "phases": [{"goto": _GOAL}]}, crowd)

    result = _run(world, doc, base, "The break is over and the whole crowd rushes back to work all at once.", ["A:stampede"])

    assert any(f.step == "A" and "without waypoints" in f.reason for f in result.record.failures), result.record.failures
    assert result.doc["humans"] == doc["humans"] and not result.files.docs


def test_a_failed_formation_returns_the_agents_it_reused(tmp_path: Path) -> None:
    crowd = [_walker("p_1", "reception", "open_work_area"), _walker("p_2", "open_work_area", "reception")]
    world, doc, base = _corridor_world(tmp_path, {"west": "reception", "hall": "bedroom", "east": "open_work_area"}, {"start": "reception", "phases": [{"goto": "open_work_area"}]}, crowd)
    assert not world.other_zones()

    decision = {"case": "interaction_unbroken", "variant": "meet", "members": ["nurse", "nurse"], "place": {"zone": "reception"}, "reuse": ["p_1"]}
    result = _run(world, doc, base, "Two nurses meet and chat in reception.", ["D:interaction_unbroken"], D=decision)

    assert any(f.step == "D" and "walk in from" in f.reason for f in result.record.failures), result.record.failures
    assert [h["name"] for h in result.doc["humans"]] == ["p_1", "p_2"]


def test_a_failed_intrinsic_group_keeps_earlier_groups_types(tmp_path: Path) -> None:
    crowd = [_walker("plain", "west", "east"), _walker("tree", "west", "east", agent_type="./needless.yaml")]
    world, doc, base = _corridor_world(tmp_path, {"west": "west", "hall": "hall", "east": "east"}, {"start": _START, "phases": [{"goto": _GOAL}]}, crowd)
    walk = {"kind": "go_to", "target_pose": {"x": 4.0, "y": 6.0, "theta": 0.0}, "patience": {"mean": 30.0}}
    (base / "needless.yaml").write_text(yaml.safe_dump({"name": "needless", "mode": "behavior_tree", "extends": "adult", "initial_sequence": "walk", "sequences": {"walk": {"steps": {"to_1": walk}, "then": "walk"}}}))

    decision = {"case": "intrinsic", "select": {"amount": "all"}, "params": {"radius": "high", "need_decay": "high"}}
    result = _run(world, doc, base, "Everyone here is wide and restless.", ["B:intrinsic"], B=decision)

    assert any(f.step == "B" and "decay_rate" in f.reason for f in result.record.failures), result.record.failures
    assert [h["agent"]["agent_type"] for h in result.doc["humans"]] == ["adult", "./needless.yaml"]
    assert not [ref for ref in result.files.docs if "radius" in ref]
