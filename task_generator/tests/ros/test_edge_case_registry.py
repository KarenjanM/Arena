from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


def _register() -> None:
    from task_generator.simulators.human.arena_humansim.arena_humansim import (
        ArenaHumanSimulator,
    )

    try:
        ArenaHumanSimulator._register_task_modes()
    except (AssertionError, KeyError, ValueError):
        pass


def test_edge_case_is_in_the_enum() -> None:
    from task_generator.constants import Constants

    assert Constants.TaskMode.TM_Obstacles("edge_case") is Constants.TaskMode.TM_Obstacles.EDGE_CASE


def test_edge_case_registers_with_the_humansim_backend() -> None:
    pytest.importorskip("arena_humansim_msgs.msg")
    from task_generator.constants import Constants
    from task_generator.tasks.registry import OBSTACLES_MODES

    _register()
    assert Constants.TaskMode.TM_Obstacles.EDGE_CASE in OBSTACLES_MODES


def test_edge_case_loader_returns_the_class() -> None:
    pytest.importorskip("arena_humansim_msgs.msg")
    from task_generator.constants import Constants
    from task_generator.tasks.obstacles.edge_case.impl import TM_EdgeCase
    from task_generator.tasks.registry import OBSTACLES_MODES

    _register()
    assert OBSTACLES_MODES.get(Constants.TaskMode.TM_Obstacles.EDGE_CASE) is TM_EdgeCase


def test_edge_case_namespace() -> None:
    pytest.importorskip("arena_humansim_msgs.msg")
    from task_generator.constants import Constants
    from task_generator.tasks.registry import OBSTACLES_MODES

    _register()
    meta = OBSTACLES_MODES.meta(Constants.TaskMode.TM_Obstacles.EDGE_CASE)
    assert "edge_case" in str(meta.namespace)
    assert meta.schema is not None


def test_schema_declares_every_param() -> None:
    from arena_rclpy_mixins.shared import Namespace
    from task_generator.tasks.obstacles.edge_case import declare_schema

    declared: list[str] = []

    class _Recorder:
        def declare_forward(self, name, default, *args, **kwargs):
            declared.append(str(name))

    class _StubNode:
        rosparam = _Recorder()

    declare_schema(_StubNode(), Namespace("task")("edge_case"))
    leaves = {name.rsplit(".", 1)[-1].rsplit("/", 1)[-1] for name in declared}
    assert {"read_scenario_block", "record_dir", "score", "score_rate_hz", "score_scope", "objects", "objects_rate_hz"} <= leaves
    assert not {"knob", "level", "mode", "inject", "place", "approach_angle", "select", "catalogue"} & leaves


def test_schema_and_impl_agree_on_every_default() -> None:
    import inspect

    from arena_rclpy_mixins.shared import Namespace
    from task_generator.tasks.obstacles.edge_case import declare_schema
    from task_generator.tasks.obstacles.edge_case.impl import TM_EdgeCase

    schema: dict[str, object] = {}

    class _Recorder:
        def declare_forward(self, name, default, *args, **kwargs):
            schema[str(name).rsplit(".", 1)[-1].rsplit("/", 1)[-1]] = default

    class _StubNode:
        rosparam = _Recorder()

    declare_schema(_StubNode(), Namespace("task")("edge_case"))

    src = inspect.getsource(TM_EdgeCase.__init__)
    impl: dict[str, str] = {}
    for line in src.splitlines():
        if 'self.namespace("' not in line or "value=" not in line:
            continue
        leaf = line.split('self.namespace("', 1)[1].split('"', 1)[0]
        impl[leaf] = line.split("value=", 1)[1].rstrip(" ),")

    consts = {
        "_MODE_INJECT": "inject",
        "_PLACE_INTERCEPT": "intercept",
        "_SELECT_NEAREST": "nearest_approach",
        "DEFAULT_CATALOGUE": "level_d",
        "SCOPE_AUTO": "auto",
        "DEFAULT_RATE_HZ": "10.0",
        "OBJECT_RATE_HZ": "5.0",
        "ROBOT_SPEED_MPS": "1.0",
    }
    mismatches = []
    for leaf, declared_default in schema.items():
        if leaf not in impl:
            continue
        raw = impl[leaf]
        resolved = consts.get(raw, raw.strip('"'))
        if str(resolved) != str(declared_default):
            mismatches.append(f"{leaf}: schema={declared_default!r} impl={resolved!r}")
    assert not mismatches, "schema and impl defaults disagree: " + "; ".join(mismatches)
