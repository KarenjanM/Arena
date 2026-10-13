from __future__ import annotations

import math
from pathlib import Path

import pytest
import shapely

from arena_simulation_setup.kg.scene import Scene
from arena_simulation_setup.shared.route import instruct, portals, zone_shapes
from arena_simulation_setup.shared.task import GoToPhase
from arena_simulation_setup.tree.World import Level, WorldView
from arena_simulation_setup.tree.World.Scenario import ScenarioView

WORLDS = Path(__file__).parents[2] / "worlds"
SCENARIOS = sorted(str(path.relative_to(WORLDS)) for path in WORLDS.glob("*/scenarios/landmark_*/scenario.yaml"))


def _level(world: str) -> tuple[Level, Scene]:
    loaded = WorldView(WORLDS / world).load()
    return loaded.all_levels[0], loaded.kg


def _robots(scenario: str) -> list[tuple[tuple[float, float, float], list[GoToPhase]]]:
    robots = ScenarioView((WORLDS / scenario).parent).load().robots
    return [(robot.start.to_2d(), robot.phase_list()) for robot in robots]


def test_landmark_scenarios_ship_for_both_furnished_worlds():
    assert {scenario.split("/")[0] for scenario in SCENARIOS} == {"hospital_1", "office_1"}


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_start_lies_in_a_zone(scenario):
    shapes = zone_shapes(_level(scenario.split("/")[0])[0])
    for start, _ in _robots(scenario):
        assert any(shape.covers(shapely.Point(start[:2])) for shape in shapes.values())


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_leg_names_a_zone_holds_a_pose_deeper_inside_than_its_tolerance_and_gets_walking_directions(scenario):
    level, semantics = _level(scenario.split("/")[0])
    shapes = zone_shapes(level)
    found = portals(level, semantics)
    for start, phases in _robots(scenario):
        here = start
        for phase in phases:
            assert phase.target in shapes
            assert shapes[phase.target].covers(shapely.Point(phase.pose.position.x, phase.pose.position.y))
            assert not shapes[phase.target].covers(shapely.Point(here[:2]))
            goal = (phase.pose.position.x, phase.pose.position.y)
            assert all(math.dist(goal, portal.mid) > phase.tolerance_radius for portal in found if phase.target in portal.zones)
            assert instruct(phase, level, semantics, here, "route", found=found)["source"] == "route"
            here = phase.pose.to_2d()
