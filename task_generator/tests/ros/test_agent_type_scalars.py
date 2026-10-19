"""A speed and radius the scenario declares override the agent type's, and every builtin type resolves by name."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.ros

pytest.importorskip("arena_humansim_msgs", reason="needs a sourced environment")

from task_generator.shared import DynamicObstacle, Pose, Position  # noqa: E402
from task_generator.simulators.human.arena_humansim.agent import ArenaHumanDynamicObstacle  # noqa: E402


def _obstacle(agent: dict | str) -> DynamicObstacle:
    return DynamicObstacle(
        name="p",
        model="arenian",
        pose=Pose(Position(1.0, 1.0)),
        waypoints=[Position(2.0, 2.0)],
        agent=agent,
    )


def test_declared_scalars_are_kept() -> None:
    parsed = ArenaHumanDynamicObstacle.from_dynamic_obstacle(
        _obstacle({"agent_type": "adult", "desired_velocity": 1.2, "radius": 0.3}),
    )
    assert (parsed.desired_velocity_min, parsed.desired_velocity_max, parsed.agent_radius) == (1.2, 1.2, 0.3)


def test_a_declared_scalar_still_wins() -> None:
    rng = np.random.default_rng(0)
    parsed = ArenaHumanDynamicObstacle.from_dynamic_obstacle(
        _obstacle({"agent_type": "elder", "desired_velocity": 1.4, "radius": 0.5}),
    )
    params = parsed.sample_params(rng)
    assert params.desired_velocity == 1.4 and params.agent_radius == 0.5


@pytest.mark.parametrize("kind", ["child", "hurried", "distracted"])
def test_new_builtin_types_resolve_by_bare_name(kind: str) -> None:
    rng = np.random.default_rng(1)
    params = ArenaHumanDynamicObstacle.from_dynamic_obstacle(_obstacle({"agent_type": kind})).sample_params(rng)
    assert params is not None and params.name == kind
