"""Agent type lookup shared by the scripted and the flow spawn paths of the arena_humansim bridge."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from task_generator.simulators.human.arena_humansim import ArenaHumanDynamicObstacle, agent_type_def


def test_agent_type_def_resolves_builtin_names_and_yaml_paths(tmp_path: Path) -> None:
    chair = agent_type_def("wheelchair_manual")
    assert chair is not None and chair.assets == ("human::mobility::wheelchair",)
    adult = agent_type_def("adult")
    assert adult is not None and adult.assets == ()
    local = tmp_path / "scooter.yaml"
    local.write_text("name: scooter\nextends: adult\nassets: [human::mobility::scooter]\n")
    scooter = agent_type_def(str(local))
    assert scooter is not None and scooter.assets == ("human::mobility::scooter",)
    assert scooter.perception == adult.perception
    assert agent_type_def("ghost") is None
    assert agent_type_def(str(tmp_path / "missing.yaml")) is None


def test_sample_params_overrides_speed_and_radius_of_the_resolved_type() -> None:
    obstacle = ArenaHumanDynamicObstacle(agent_type="wheelchair_manual", desired_velocity_min=0.9, desired_velocity_max=0.9, agent_radius=0.325)
    params = obstacle.sample_params(np.random.default_rng(3))
    assert params is not None
    assert (params.desired_velocity, params.agent_radius) == (0.9, 0.325)
    assert params.locomotion.active and params.interaction_class == "wheelchair"
    assert ArenaHumanDynamicObstacle(agent_type="ghost").sample_params(np.random.default_rng(3)) is None


def test_scenario_human_agent_block_and_waypoint_mode_drive_the_spawn(tmp_path: Path) -> None:
    from arena_humansim_msgs.msg import Waypoints as WaypointsMsg

    from task_generator.shared import DynamicObstacle

    human = DynamicObstacle.parse(
        {
            "name": "nurse_1",
            "model": "arenian",
            "pose": [1.0, 2.0, 0.0],
            "waypoints": [[3.0, 4.0]],
            "waypoint_mode": "once",
            "agent": {"agent_type": "./doctor.yaml", "desired_velocity": {"min": 0.6, "max": 0.9}, "radius": 0.3},
        }
    )
    human.included_from = tmp_path
    parsed = ArenaHumanDynamicObstacle.from_dynamic_obstacle(human)
    assert parsed is not None
    assert parsed.agent_type == str((tmp_path / "doctor.yaml").resolve())
    assert (parsed.desired_velocity_min, parsed.desired_velocity_max, parsed.agent_radius) == (0.6, 0.9, 0.3)
    assert parsed.waypoint_mode == WaypointsMsg.MODE_ONCE
    assert [(waypoint.x, waypoint.y) for waypoint in parsed._waypoints] == [(3.0, 4.0)]


def test_scenario_human_without_agent_block_is_not_an_arena_humansim_agent() -> None:
    from task_generator.shared import DynamicObstacle

    human = DynamicObstacle.parse({"name": "ghost", "model": "arenian", "pose": [0.0, 0.0, 0.0]})
    assert ArenaHumanDynamicObstacle.from_dynamic_obstacle(human) is None
