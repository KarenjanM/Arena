from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from arena_simulation_setup.generative.scenario.cli import _override_robot, _parser, main

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


def test_every_verb_and_option_has_help() -> None:
    verbs = next(a for a in _parser()._actions if isinstance(a, argparse._SubParsersAction))
    missing = [f"{name} {action.dest}" for name, sub in verbs.choices.items() for action in sub._actions if not action.help]
    assert missing == []


@pytest.mark.parametrize("verb", ["zones", "populate", "promptgen", "suite"])
def test_a_bare_verb_prints_its_help_and_the_worlds(tmp_path: Path, capsys: pytest.CaptureFixture[str], verb: str) -> None:
    assert main([verb, "--worlds", str(_worlds(tmp_path))]) == 2
    out = capsys.readouterr().out
    assert f"usage: generate_scenario {verb}" in out
    assert "sbend" in out


def test_an_unknown_world_names_the_worlds(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["zones", "nope", "--worlds", str(_worlds(tmp_path))]) == 1
    err = capsys.readouterr().err
    assert "'nope'" in err and "sbend" in err


def test_prune_without_write_says_to_add_write(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc, _, err = _populate(tmp_path, capsys, "--prune")
    assert rc == 1
    assert "add --write" in err


def test_unknown_bases_and_ids_name_the_valid_ones(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    worlds = _worlds(tmp_path)
    assert main(["populate", "sbend", "--worlds", str(worlds), "--write"]) == 0
    capsys.readouterr()
    rc, _, err = _promptgen(worlds, capsys, "--bases", "normal,nope")
    assert rc == 1
    assert "nope" in err and "break, empty, night, normal" in err
    rc, _, err = _promptgen(worlds, capsys, "--ids", "u_002,nope")
    assert rc == 1
    assert "nope" in err and "u_001" in err


def _run(*argv: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "arena_simulation_setup.generative.scenario.cli", *argv], env={**os.environ, "ARENA_WORLD_PATH": "", **env}, capture_output=True, text=True, timeout=300, check=False)


def _checkout(tmp_path: Path) -> Path:
    arena = tmp_path / "arena"
    shutil.copytree(_worlds(tmp_path), arena / "arena_simulation_setup" / "worlds")
    (arena / "arena_evaluation" / "arena_evaluation" / "configs" / "benchmark" / "suites").mkdir(parents=True)
    return arena


def test_listing_honors_arena_world_path(tmp_path: Path) -> None:
    arena = _checkout(tmp_path)
    extra = tmp_path / "extra"
    shutil.copytree(arena / "arena_simulation_setup" / "worlds" / "sbend", extra / "mine")
    done = _run("zones", ARENA_DIR=str(arena), ARENA_WORLD_PATH=str(extra))
    assert done.returncode == 2
    assert "mine, sbend" in done.stdout


def test_writes_default_to_the_source_tree(tmp_path: Path) -> None:
    arena = _checkout(tmp_path)
    done = _run("populate", "sbend", "--write", ARENA_DIR=str(arena))
    assert done.returncode == 0, done.stdout + done.stderr
    assert (arena / "arena_simulation_setup" / "worlds" / "sbend" / "scenarios" / "normal" / "scenario.yaml").is_file()
    done = _run("suite", "sbend", ARENA_DIR=str(arena))
    assert done.returncode == 0, done.stdout + done.stderr
    assert (arena / "arena_evaluation" / "arena_evaluation" / "configs" / "benchmark" / "suites" / "edge_case.yaml").is_file()


def test_without_a_root_the_verb_asks_for_one(tmp_path: Path) -> None:
    env = {k: v for k, v in os.environ.items() if k != "ARENA_DIR"}
    done = subprocess.run([sys.executable, "-m", "arena_simulation_setup.generative.scenario.cli", "populate", "sbend"], env={**env, "ARENA_WORLD_PATH": ""}, capture_output=True, text=True, timeout=300, check=False)
    assert done.returncode == 1
    assert "--worlds" in done.stderr and "ARENA_DIR" in done.stderr


def test_writes_into_an_install_prefix_or_the_asset_cache_are_refused(tmp_path: Path) -> None:
    src = _worlds(tmp_path)
    prefix = tmp_path / "install" / "pkg"
    shutil.copytree(src, prefix / "share" / "worlds")
    done = _run("populate", "sbend", "--write", "--worlds", str(prefix / "share" / "worlds"), AMENT_PREFIX_PATH=f"{os.environ.get('AMENT_PREFIX_PATH', '')}:{prefix}")
    assert done.returncode == 1
    assert "install prefix" in done.stderr
    assert not (prefix / "share" / "worlds" / "sbend" / "scenarios").exists()
    assets = tmp_path / "assets"
    shutil.copytree(src, assets / "worlds")
    done = _run("suite", "sbend", "--worlds", str(assets / "worlds"), "--out", str(assets / "s.yaml"), ARENA_ASSETS_DIR=str(assets))
    assert done.returncode == 1
    assert "asset cache" in done.stderr


@pytest.mark.parametrize("level", [0, "0", "2"])
def test_the_base_route_keeps_the_robots_levels(level: int | str) -> None:
    from arena_simulation_setup.generative.scenario.route import robot_for

    spec = robot_for({"start": [0.0, 0.0, 0.0], "goal": [20.0, 0.0], "level": level, "goal_level": level}, [])
    assert spec["level"] == level and spec["goal_level"] == level


def test_a_world_loads_only_its_own_prompt_set(tmp_path: Path) -> None:
    from arena_simulation_setup.generative.scenario.promptgen.prompts import load_sets

    for name in ("universal", "hospital_1", "hospital_1_bare"):
        (tmp_path / f"{name}.yaml").write_text(f"- {{id: {name}, text: t}}\n")
    assert [p.id for p in load_sets("hospital_1", directory=tmp_path)] == ["universal", "hospital_1"]
    assert [p.id for p in load_sets("hospital_1_bare", directory=tmp_path)] == ["universal", "hospital_1_bare"]


_NO_MODEL = {k: v for k, v in os.environ.items() if k not in ("LLM_API_ENDPOINT", "GOOGLE_API_KEY", "GEMINI_API_KEY", "ARENA_DATA_DIR")}


def _python(code: str, cwd: Path, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", code], cwd=cwd, env={**_NO_MODEL, **env}, capture_output=True, text=True, timeout=120, check=False)


def test_the_default_llm_cache_is_absolute_without_arena_data_dir(tmp_path: Path) -> None:
    done = _python("from arena_simulation_setup.generative.scenario.promptgen.llm import DEFAULT_CACHE; print(DEFAULT_CACHE)", tmp_path, ARENA_WS_DIR=str(tmp_path / "ws"))
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == str(tmp_path / "ws" / "data" / "promptgen" / "llm_cache")


def test_a_truncated_cache_entry_is_a_miss(tmp_path: Path) -> None:
    code = """
from pathlib import Path
from arena_simulation_setup.generative.scenario.promptgen.llm import Unavailable, call
from arena_simulation_setup.generative.scenario.propose.client import DEFAULT_MODEL, prompt_sha256
cache = Path('cache')
cache.mkdir()
(cache / f"{prompt_sha256(DEFAULT_MODEL, 's', 'u')}.json").write_text('{"model": "m", "answ')
try:
    call('s', 'u', cache_dir=cache)
except Unavailable:
    print('miss')
"""
    done = _python(code, tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "miss"


def test_a_cache_entry_is_written_whole_or_not_at_all(tmp_path: Path) -> None:
    from arena_simulation_setup.generative.scenario.promptgen.llm import _read_cached, _write_cached

    path = tmp_path / "k.json"
    _write_cached(path, {"model": "m", "answer": "{}"})
    assert [p.name for p in tmp_path.iterdir()] == ["k.json"]
    answer = _read_cached(path, "m")
    assert answer is not None and answer.text == "{}" and answer.cached
