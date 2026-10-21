from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml
from arena_humansim.core.agents import loader
from arena_simulation_setup.generative.scenario.cli import main
from arena_simulation_setup.generative.scenario.formats import read_scenario
from arena_simulation_setup.generative.scenario.promptgen import prompts as P
from arena_simulation_setup.generative.scenario.promptgen.pipeline import generate, write_result
from arena_simulation_setup.generative.scenario.promptgen.world import WorldContext
from arena_simulation_setup.tree.World.Scenario import parse_scenario
from arena_simulation_setup.tree.World.view import WorldView

SHIPPED = Path(__file__).resolve().parents[2] / "worlds"
_POPULATED: dict[str, Path] = {}


def _populated(name: str, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The shipped world `name` copied without its scenarios and populated by `populate --write`, once per module."""
    if name not in _POPULATED:
        worlds = tmp_path_factory.mktemp("worlds")
        shutil.copytree(SHIPPED / name, worlds / name, ignore=shutil.ignore_patterns("scenarios", "map"))
        assert main(["populate", name, "--worlds", str(worlds), "--write"]) == 0
        _POPULATED[name] = worlds / name
    return _POPULATED[name]


@pytest.mark.parametrize(
    ("name", "text", "targets", "oracle"),
    [
        ("three_storied_residential", "The lights go out.", ("A:blackout",), {}),
        ("sbend_corridor", "A visitor walks up from the robot's right and stops in front of it.", ("C:approach",), {"C": {"case": "approach", "profile": "adult", "angle": 90, "route_fraction": 0.5, "after": "stop_facing", "duration": 6}}),
        ("sbend_corridor", "An old man walks at the robot and will not give way. People wander about.", ("D:stubborn", "B:erratic_route"), {"D": {"case": "stubborn", "profile": "elder", "route_fraction": 0.4}, "B": {"case": "erratic_route", "select": {"amount": "all"}}}),
    ],
)
def test_populate_output_of_a_shipped_world_runs_through_promptgen_into_a_scenario_the_model_reads(tmp_path_factory: pytest.TempPathFactory, name: str, text: str, targets: tuple[str, ...], oracle: dict) -> None:
    populated = _populated(name, tmp_path_factory)
    base = populated / "scenarios" / "normal"
    doc = read_scenario(base / "scenario.yaml")
    assert doc["humans"]
    world = WorldContext.load(populated, doc)
    prompt = P.Prompt(id=f"t_{targets[0][0].lower()}", text=text, targets=targets, oracle=oracle)

    result = generate(prompt, world=world, base_id="normal", base_doc=doc, base_dir=base, oracle=True)
    assert result.record.status == "ok", result.record.failures
    out = write_result(result, populated)

    raw = yaml.safe_load((out / "scenario.yaml").read_text())
    view = WorldView(populated.resolve())
    model = view.load()
    origins = view.level_origins() or dict.fromkeys(model.levels, (0.0, 0.0))
    loaded = parse_scenario(raw, model.compact_world(origins).zone_converter(np.random.default_rng(0)))
    assert {h.name for h in loaded.humans} >= {h["name"] for h in doc["humans"]} and len(loaded.timeline) >= 1
    assert raw["robots"][0]["level"] == raw["robots"][0]["goal_level"] == world.level
    for agent in raw["humans"]:
        ref = agent["agent"]["agent_type"]
        if ref.startswith("./"):
            assert loader.load_agent_type_from_file(out / ref).mode == "behavior_tree"
    if "C:approach" in targets:
        (edge,) = [h for h in raw["humans"] if h["name"] == "edge_0"]
        ox, oy = origins[world.level]
        assert world.zone_of((edge["pose"][0] - ox, edge["pose"][1] - oy)) is not None
