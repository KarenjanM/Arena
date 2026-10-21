"""Scenario timeline stimuli checked against the agent types of the arena_humansim backend."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytest.importorskip("rclpy")
pytest.importorskip("arena_humansim_msgs")

from arena_simulation_setup.tree.World.Scenario import SCENARIO_FORMAT, Scenario, ScenarioView

from task_generator.simulators.human.arena_humansim.agent import agent_type_def, agent_type_stimuli
from task_generator.simulators.human.arena_humansim.arena_humansim import ArenaHumanSimulator

EDGE_TYPE = {
    "name": "edge",
    "mode": "behavior_tree",
    "extends": "adult",
    "needs": {"go": {"initial": 0.0, "decay_rate": 0.0}},
    "initial_sequence": "wait",
    "sequences": {
        "wait": {
            "steps": {"stand": {"autonomous": True, "until": "form"}},
            "transitions": [{"when": {"go": {"above": 50.0}}, "goto": "walk"}],
        },
        "walk": {"steps": {"stroll": {"duration": 5.0}}},
    },
}


def _scenario(directory: Path, timeline: list[dict]) -> Scenario:
    directory.mkdir()
    (directory / "edge.yaml").write_text(yaml.safe_dump(EDGE_TYPE))
    humans = [
        {"name": "edge_0", "model": "arenian", "pose": [1.0, 2.0], "agent": {"agent_type": "./edge.yaml"}},
        {"name": "bystander", "model": "arenian", "pose": [3.0, 2.0]},
    ]
    (directory / "scenario.yaml").write_text(yaml.safe_dump({"format": SCENARIO_FORMAT, "humans": humans, "timeline": timeline}))
    return ScenarioView(directory).load()


def _adapter() -> ArenaHumanSimulator:
    return object.__new__(ArenaHumanSimulator)


def test_agent_type_stimuli_are_its_needs_and_its_until_events(tmp_path: Path) -> None:
    path = tmp_path / "edge.yaml"
    path.write_text(yaml.safe_dump(EDGE_TYPE))
    definition = agent_type_def(str(path))
    assert definition is not None
    assert agent_type_stimuli(definition) == {"go", "form"}


def test_check_stimuli_accepts_a_declared_need_and_an_until_event(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path / "sc", [{"at": 1.0, "stimulus": [{"agents": ["edge_0"], "need": "go"}, {"agents": ["edge_0"], "need": "form"}]}])
    _adapter().check_stimuli("sc", scenario.humans, scenario.timeline)


def test_check_stimuli_rejects_a_need_the_agent_type_does_not_act_on(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path / "sc", [{"at": 1.0, "stimulus": [{"agents": ["edge_0"], "need": "run"}]}])
    with pytest.raises(ValueError, match=r"scenario 'sc': stimulus 'run' for 'edge_0' is neither a need nor a step `until` event of its agent type '.*edge.yaml', which acts on \['form', 'go'\]"):
        _adapter().check_stimuli("sc", scenario.humans, scenario.timeline)


def test_check_stimuli_skips_an_agent_without_an_agent_type(tmp_path: Path) -> None:
    scenario = _scenario(tmp_path / "sc", [{"at": 1.0, "stimulus": [{"agents": ["bystander"], "need": "run"}]}])
    assert _adapter().stimuli_taken(scenario.humans[1]) is None
    _adapter().check_stimuli("sc", scenario.humans, scenario.timeline)
