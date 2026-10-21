from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from arena_simulation_setup.generative.scenario.cli import _override_robot, main

SHIPPED = Path(__file__).resolve().parents[2] / "worlds"


def _worlds(tmp_path: Path) -> Path:
    root = tmp_path / "worlds"
    (root / "sbend").mkdir(parents=True)
    shutil.copy(SHIPPED / "sbend_corridor" / "world.yaml", root / "sbend" / "world.yaml")
    return root


def _populate(tmp_path: Path, capsys: pytest.CaptureFixture[str], *extra: str) -> tuple[int, str, str]:
    rc = main(["populate", "sbend", "--worlds", str(_worlds(tmp_path)), *extra])
    out = capsys.readouterr()
    return rc, out.out, out.err


def test_a_coordinate_robot_start_is_parsed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc, out, _ = _populate(tmp_path, capsys, "--robot-start", "1,1.5,0")
    assert rc == 0
    assert "start=[1.0, 1.5, 0.0]" in out


def test_a_robot_goal_applies_without_a_robot_start(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc, out, _ = _populate(tmp_path, capsys, "--robot-goal", "8,11.5")
    assert rc == 0
    assert "goal=[8.0, 11.5]" in out


def test_a_zone_robot_goal_is_kept_as_a_zone(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc, out, _ = _populate(tmp_path, capsys, "--robot-start", "corridor_south", "--robot-goal", "corridor_north")
    assert rc == 0
    assert "start='corridor_south' goal='corridor_north'" in out


@pytest.mark.parametrize("value", ["nowhere", "1,2,x", "1", "1,2,3,4"])
def test_a_bad_robot_endpoint_names_the_valid_forms(tmp_path: Path, capsys: pytest.CaptureFixture[str], value: str) -> None:
    rc, _, err = _populate(tmp_path, capsys, "--robot-start", value)
    assert rc == 1
    assert "x,y[,yaw]" in err
    assert "corridor_north" in err


def test_a_new_start_keeps_the_route_and_a_new_goal_replaces_it() -> None:
    zones = [{"name": "a"}, {"name": "b"}]
    robot = {"start": "a", "phases": [{"goto": "b"}, {"goto": "a"}], "level": "1", "goal_level": "2"}
    moved = _override_robot(robot, zones, start="0,0", goal=None)
    assert moved == {**robot, "start": [0.0, 0.0]}
    retargeted = _override_robot(robot, zones, start=None, goal="b")
    assert retargeted == {"start": "a", "goal": "b", "level": "1"}


def _promptgen(worlds: Path, capsys: pytest.CaptureFixture[str], *extra: str) -> tuple[int, str, str]:
    rc = main(["promptgen", "sbend", "--worlds", str(worlds), "--oracle", *extra])
    out = capsys.readouterr()
    return rc, out.out, out.err


def test_promptgen_without_any_base_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc, _, err = _promptgen(_worlds(tmp_path), capsys)
    assert rc == 1
    assert "nothing generated: none of the bases normal, break, night, empty could be loaded" in err


def test_promptgen_fails_when_a_prompt_crashes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    worlds = _worlds(tmp_path)
    assert main(["populate", "sbend", "--worlds", str(worlds), "--write"]) == 0
    scenarios = worlds / "sbend" / "scenarios"
    doc = yaml.safe_load((scenarios / "normal" / "scenario.yaml").read_text())
    (scenarios / "broken").mkdir()
    (scenarios / "broken" / "scenario.yaml").write_text(yaml.safe_dump({**doc, "humans": [{"name": "x", "pose": ["a", "b"]}]}))
    capsys.readouterr()
    rc, out, err = _promptgen(worlds, capsys, "--bases", "broken,normal", "--ids", "u_002")
    assert "crashed=1" in out and "ok=1" in out, out
    assert rc == 1
    assert "1 prompt(s) crashed" in err


def test_promptgen_succeeds_when_a_case_is_realized(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    worlds = _worlds(tmp_path)
    assert main(["populate", "sbend", "--worlds", str(worlds), "--write"]) == 0
    capsys.readouterr()
    rc, out, err = _promptgen(worlds, capsys, "--bases", "normal", "--ids", "u_002")
    assert rc == 0, out + err
    assert "ok=1" in out or "partial=1" in out
