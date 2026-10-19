from __future__ import annotations

import math

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")
    pytest.importorskip("arena_humansim.core.agents.loader")


ROUTE = ((0.0, 0.0), (20.0, 0.0))


def _block(**raw):
    from task_generator.tasks.obstacles.edge_case.scenario_block import parse

    return parse({"id": "case", **raw})


def _scene(**kwargs):
    from task_generator.tasks.obstacles.edge_case.build import Scene

    return Scene(**{"route": ROUTE, "robot_speed": 1.0, **kwargs})


def obstacle(name, start, waypoints, agent_type="adult", speed=1.0, model="arenian"):
    from task_generator.shared import DynamicObstacle, Pose, Position

    return DynamicObstacle(name=name, model=model, pose=Pose(Position(*start)), waypoints=[Position(*w) for w in waypoints], agent={"agent_type": agent_type, "desired_velocity": speed})


def _population():
    return [obstacle("walker_a", (5.0, 5.0), [(15.0, 5.0)]), obstacle("walker_b", (15.0, 6.0), [(5.0, 6.0)], speed=1.3)]


def _build(block, population=None, scene=None):
    from task_generator.tasks.obstacles.edge_case.build import build

    return build(block, _population() if population is None else population, scene or _scene())


def _aborts(match, block, population=None, scene=None):
    from task_generator.tasks.obstacles.edge_case.build import CaseAborted

    with pytest.raises(CaseAborted, match=match):
        _build(block, population, scene)


def test_empty_block_adds_nothing() -> None:
    case = _build(_block())
    assert case.injected == [] and case.plans == []
    assert case.geometry["robot_route"] == ROUTE and case.geometry["route_length"] == 20.0


def test_intercept_adds_one_agent_on_the_route() -> None:
    case = _build(_block(effects=[{"type": "intercept"}]))
    (edge,) = case.injected
    assert edge.name == "edge_0"
    assert edge.pose.position.x > 10.0 and edge.pose.position.y == pytest.approx(0.0)
    g = case.geometry
    assert g["place"] == "intercept" and g["encounter_point"] == pytest.approx((10.0, 0.0))
    assert g["target_agent"] == "edge_0" and g["inject_type"] == "adult" and g["inject_model"] == "arenian"


def test_intercept_speed_is_the_speed_the_geometry_used() -> None:
    from task_generator.tasks.obstacles.edge_case.build import LEAD_IN_M

    case = _build(_block(effects=[{"type": "intercept", "speed": 0.8}]))
    (edge,) = case.injected
    g = case.geometry
    assert edge.agent["desired_velocity"] == pytest.approx(0.8) and edge.agent["radius"] == pytest.approx(0.25)
    # it stands at most a short leg from the encounter and is released when the robot is as far from it as it drives meanwhile
    lead = math.dist((edge.pose.position.x, edge.pose.position.y), g["encounter_point"])
    assert lead == pytest.approx(min(LEAD_IN_M, 0.8 * g["t_encounter"]))
    ((plan, trigger, reference),) = case.plans
    assert plan.mode == "rally" and plan.targets["edge_0"] == pytest.approx(tuple(g["ped_goal"]))
    assert trigger.robot_eta == pytest.approx(round(lead / 0.8, 2)) and reference == pytest.approx(tuple(g["encounter_point"]))
    assert [(w.x, w.y) for w in edge.waypoints] == pytest.approx([(edge.pose.position.x, edge.pose.position.y)])


def test_stand_places_the_agent_on_the_route() -> None:
    from task_generator.tasks.obstacles.edge_case.build import STAND_TRIGGER_M

    case = _build(_block(effects=[{"type": "intercept", "after": "stand", "route_fraction": 0.3, "duration": 6}]))
    (edge,) = case.injected
    assert (edge.pose.position.x, edge.pose.position.y) == pytest.approx((6.0, 0.0)) and edge.waypoint_mode == "once"
    assert case.geometry["place"] == "stand"
    ((plan, trigger, ref),) = case.plans
    # the stand's duration starts when the robot reaches it
    assert plan.mode == "hold" and plan.duration == 6.0 and trigger.robot_within == STAND_TRIGGER_M and ref == pytest.approx((6.0, 0.0))
    assert plan.resume["edge_0"][0] > 6.0  # released further along the route


def test_stop_facing_holds_at_the_encounter_then_resumes() -> None:
    case = _build(_block(effects=[{"type": "intercept", "after": "stop_facing", "duration": 4}]))
    g = case.geometry
    ((plan, trigger, _r),) = case.plans
    walk = math.dist(g["spawn_pose"], g["encounter_point"]) / g["ped_speed"]
    assert plan.mode == "hold" and plan.at == 0.0 and trigger.robot_eta == g["release_eta"]
    assert plan.duration == pytest.approx(walk + 4.0)
    assert plan.points["edge_0"] == pytest.approx(tuple(g["encounter_point"])) and plan.resume["edge_0"] == pytest.approx(tuple(g["ped_goal"]))


def test_intercept_timing_does_not_shift_with_the_robots_hold() -> None:
    # The walker is released on the robot's approach, so a hold at the start (or nav2's first-plan delay) shifts nothing.
    case = _build(_block(effects=[{"type": "intercept", "after": "stop_facing", "duration": 4}]), scene=_scene(robot_hold=20.0))
    ((stop, trigger, _r),) = case.plans
    assert stop.mode == "hold" and stop.at == 0.0 and trigger.robot_eta == case.geometry["release_eta"] and case.geometry["robot_hold"] == 20.0


def test_follow_tracks_the_robot_after_the_encounter() -> None:
    case = _build(_block(effects=[{"type": "intercept", "after": "follow", "duration": 10}]))
    g = case.geometry
    ((plan, trigger, _r),) = case.plans
    walk = math.dist(g["spawn_pose"], g["encounter_point"]) / g["ped_speed"]
    assert plan.mode == "track_robot" and plan.duration == pytest.approx(walk + 10.0) and set(plan.homes) == {"edge_0"} and trigger is not None


def test_veer_walks_through_the_encounter_to_a_zone() -> None:
    case = _build(_block(effects=[{"type": "intercept", "after": "veer", "veer_to": "lobby"}]), scene=_scene(zones={"lobby": ((0, 10), (4, 10), (4, 14), (0, 14))}))
    (edge,) = case.injected
    ((plan, trigger, _r),) = case.plans
    assert plan.mode == "hold" and plan.points["edge_0"] == pytest.approx(tuple(case.geometry["encounter_point"])) and plan.resume["edge_0"] == pytest.approx((2.0, 12.0))
    assert edge.waypoint_mode == "once" and trigger is not None


def test_rally_over_everyone_includes_the_injected_agent() -> None:
    case = _build(_block(effects=[{"type": "intercept"}, {"type": "rally", "scope": "all", "target": [30.0, 0.0], "when": {"at": 8}}]))
    plans = [p for p, _t, _r in case.plans]
    (release,) = [p for p in plans if p.label == "edge_0:release"]
    (rally,) = [p for p in plans if p.mode == "rally" and p is not release]
    assert set(rally.homes) == {"walker_a", "walker_b", "edge_0"} and rally.at == 8.0
    assert set(rally.targets.values()) == {(30.0, 0.0)}


def test_rally_targets_are_dealt_round_robin() -> None:
    case = _build(_block(effects=[{"type": "rally", "scope": ["walker_a", "walker_b"], "targets": [[1.0, 1.0], [2.0, 2.0]]}]))
    ((plan, _t, _r),) = case.plans
    assert plan.targets == {"walker_a": (1.0, 1.0), "walker_b": (2.0, 2.0)}


def test_scope_naming_an_absent_agent_aborts() -> None:
    _aborts("nobody", _block(effects=[{"type": "scatter", "scope": ["nobody"]}]))


def test_injected_scope_with_nothing_injected_aborts() -> None:
    _aborts("matched none", _block(effects=[{"type": "hold", "scope": "injected"}]))


def test_route_fraction_gate_resolves_to_a_point_on_the_route() -> None:
    case = _build(_block(effects=[{"type": "scatter", "scope": ["walker_a"], "when": {"route_fraction": 0.25}}]))
    ((plan, trigger, reference),) = case.plans
    assert plan.mode == "shuffle" and plan.once
    assert trigger is not None and trigger.robot_within == 1.5 and reference == pytest.approx((5.0, 0.0))


def test_proximity_gate_defaults_to_the_owned_centroid() -> None:
    case = _build(_block(effects=[{"type": "scatter", "scope": ["walker_a", "walker_b"], "when": {"robot_within": 3.0}}]))
    ((_plan, trigger, reference),) = case.plans
    assert trigger.robot_within == 3.0 and reference == pytest.approx((10.0, 5.5))


def test_unknown_zone_aborts_with_the_known_ones() -> None:
    _aborts("lobby", _block(effects=[{"type": "rally", "target": "loby"}]), scene=_scene(zones={"lobby": ((0, 0), (1, 0), (1, 1), (0, 1))}))


def test_hold_releases_to_the_agents_own_goal() -> None:
    case = _build(_block(effects=[{"type": "hold", "scope": ["walker_b"], "duration": 5}]))
    ((plan, _t, _r),) = case.plans
    assert plan.resume == {"walker_b": (5.0, 6.0)}


def test_no_robot_route_aborts_an_intercept_with_the_reason() -> None:
    _aborts("no robot is present", _block(effects=[{"type": "intercept"}]), scene=_scene(route=(), no_route="an effect needs the robot's route, but no robot is present"))


def test_a_non_positive_robot_speed_aborts_an_intercept() -> None:
    _aborts("must be positive", _block(effects=[{"type": "intercept"}]), scene=_scene(robot_speed=0.0))


def test_profile_adopts_from_the_population_by_sorted_name() -> None:
    population = [obstacle("w", (5.0, 5.0), [(6.0, 5.0)], agent_type="elder"), obstacle("v", (6.0, 5.0), [(7.0, 5.0)], agent_type="hurried")]
    case = _build(_block(effects=[{"type": "intercept"}]), population)
    assert case.injected[-1].agent["agent_type"] == "elder"


def test_unknown_profile_aborts() -> None:
    _aborts("nope.yaml", _block(effects=[{"type": "intercept", "profile": "./types/nope.yaml"}]))


def test_a_scenario_local_profile_is_found_beside_the_scenario(tmp_path) -> None:
    from task_generator.tasks.obstacles.edge_case.build import AgentTypes

    (tmp_path / "types").mkdir()
    (tmp_path / "types" / "slow.yaml").write_text("name: slow\nextends: adult\ndesired_velocity: {mean: 0.4, std: 0.0}\n")
    case = _build(_block(effects=[{"type": "intercept", "profile": "./types/slow.yaml"}]), scene=_scene(types=AgentTypes(scenario_dirs=[str(tmp_path)])))
    assert case.injected[0].agent["desired_velocity"] == pytest.approx(0.4)


# A need switches a behaviour tree's sequence
# -------------------------------------------


def _tree_type(tmp_path, need="alarm"):
    path = tmp_path / "evacuating.yaml"
    path.write_text(
        "name: evacuating\nmode: behavior_tree\nextends: adult\n"
        f"needs: {{{need}: {{initial: 0.0, decay_rate: 0.0}}}}\n"
        "initial_sequence: chat\n"
        "sequences:\n"
        "  chat: {steps: {talk: {interaction: GROUP_CONVERSATION, duration: 30.0}}, then: chat, transitions: [{when: {alarm: {above: 50.0}}, goto: leave}]}\n"
        "  leave: {steps: {go: {kind: go_to, target_pose: {x: 1.0, y: 2.0, theta: 0.0}}}}\n"
    )
    return str(path)


def test_stimulus_plans_the_need_on_behaviour_trees(tmp_path) -> None:
    talker = _tree_type(tmp_path)
    population = [obstacle("talker_1", (5.0, 5.0), [], agent_type=talker), obstacle("talker_2", (6.0, 5.0), [], agent_type=talker), *_population()]
    case = _build(_block(effects=[{"type": "stimulus", "scope": ["talker_1", "talker_2"], "need": "alarm", "label": "A:evacuation", "when": {"at": 8}}]), population)
    ((plan, _t, _r),) = case.plans
    assert plan.mode == "stimulus" and plan.need == "alarm" and plan.intensity == 1.0 and plan.at == 8.0 and sorted(plan.homes) == ["talker_1", "talker_2"]
    assert plan.as_dict()["need"] == "alarm"


def test_stimulus_on_an_agent_that_cannot_react_aborts(tmp_path) -> None:
    from task_generator.tasks.obstacles.edge_case.build import CaseAborted

    deaf = _tree_type(tmp_path, need="social")
    population = [obstacle("talker_1", (5.0, 5.0), [], agent_type=deaf), *_population()]
    with pytest.raises(CaseAborted, match="reaches no behaviour tree") as raised:
        _build(_block(effects=[{"type": "stimulus", "scope": ["talker_1", "walker_a"], "need": "alarm"}]), population)
    assert "talker_1" in str(raised.value) and "walker_a (adult)" in str(raised.value)


# Spawns outside free space
# -------------------------


def _walled_map():
    import numpy as np
    from arena_rclpy_mixins.Time import Time
    from task_generator.manager.world_manager.utils import WorldLayers, WorldMap, WorldOccupancy
    from task_generator.shared import Position

    # 30 x 20 m at 0.1 m, a wall across x = 12..13 m
    grid = np.full((200, 300), WorldOccupancy.EMPTY, dtype=np.uint8)
    grid[:, 120:130] = WorldOccupancy.FULL
    return WorldMap(occupancy=WorldLayers(walls=WorldOccupancy(grid)), origin=Position(x=0.0, y=-10.0), resolution=0.1, time=Time())


def test_a_bystander_in_a_wall_is_recorded_not_fatal() -> None:
    from task_generator.tasks.obstacles._validation import make_is_valid

    population = [obstacle("walker_a", (5.0, 5.0), [(6.0, 5.0)]), obstacle("stuck", (12.5, 5.0), [(6.0, 5.0)])]
    case = _build(_block(), population, scene=_scene(is_valid=make_is_valid(_walled_map(), 0.3)))
    assert case.invalid_spawns == ("stuck",)
