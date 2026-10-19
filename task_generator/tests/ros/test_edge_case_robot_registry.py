from __future__ import annotations


import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


def test_edge_case_is_in_the_robot_enum() -> None:
    from task_generator.constants import Constants

    assert Constants.TaskMode.TM_Robots("edge_case") is Constants.TaskMode.TM_Robots.EDGE_CASE


def test_registered_without_a_humansim_backend() -> None:
    from task_generator.constants import Constants
    from task_generator.tasks.registry import ROBOTS_MODES

    assert Constants.TaskMode.TM_Robots.EDGE_CASE in ROBOTS_MODES


def test_loader_returns_the_class() -> None:
    from task_generator.constants import Constants
    from task_generator.tasks.registry import ROBOTS_MODES
    from task_generator.tasks.robots.edge_case.impl import TM_EdgeCase

    assert ROBOTS_MODES.get(Constants.TaskMode.TM_Robots.EDGE_CASE) is TM_EdgeCase


def test_namespace_is_separate_from_the_obstacle_mode() -> None:
    from task_generator.constants import Constants
    from task_generator.tasks.registry import OBSTACLES_MODES, ROBOTS_MODES

    robots_ns = str(ROBOTS_MODES.meta(Constants.TaskMode.TM_Robots.EDGE_CASE).namespace)
    assert "edge_case_robot" in robots_ns

    try:
        obstacles_ns = str(OBSTACLES_MODES.meta(Constants.TaskMode.TM_Obstacles.EDGE_CASE).namespace)
    except KeyError:
        return
    assert robots_ns != obstacles_ns


def test_schema_declares_every_param() -> None:
    from arena_rclpy_mixins.shared import Namespace
    from task_generator.tasks.robots.edge_case import declare_schema

    declared: list[str] = []

    class _Recorder:
        def declare_forward(self, name, default, *args, **kwargs):
            declared.append(str(name))

    class _StubNode:
        rosparam = _Recorder()

    declare_schema(_StubNode(), Namespace("task")("edge_case_robot"))
    leaves = {n.rsplit(".", 1)[-1].rsplit("/", 1)[-1] for n in declared}
    assert {"traversals", "blocked_distance", "blocked_timeout", "on_blocked"} <= leaves


def test_reads_the_shared_scenario_param() -> None:
    import inspect

    from task_generator.tasks.robots.edge_case import impl

    src = inspect.getsource(impl.TM_EdgeCase.__init__)
    assert '_REGISTRY_NAMESPACE("scenario")' in src
    assert "TM_Robots.__init__" in src, "must skip TM_Scenario.__init__, which binds the wrong namespace"


def test_robot_spawn_clearance_is_robot_sized_not_pedestrian_sized() -> None:
    from task_generator.tasks.robots.scenario.impl import spawn_clearance

    assert spawn_clearance([0.52, 0.4], 0.25) == pytest.approx(0.52)
    assert spawn_clearance([], 0.25) == pytest.approx(0.55), "must not collapse to the pedestrian clearance"
