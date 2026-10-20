"""Mid-episode obstacle removal by id: what resolves, what is forgotten, what is reported."""

from __future__ import annotations

import asyncio
import typing
from pathlib import Path

import pytest

if typing.TYPE_CHECKING:
    from task_generator.manager.environment_manager import EnvironmentManager
    from task_generator.simulators.human.noop import NoopHumanSimulator


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("rclpy")
    pytest.importorskip("arena_runtime.sim.dummy_simulator")


@pytest.fixture
def world_dir(tmp_path: Path) -> typing.Iterator[Path]:
    from arena_simulation_setup.tree import DynamicPaths

    crate = tmp_path / "assets" / "Common" / "Object" / "crate"
    crate.mkdir(parents=True)
    (crate / "annotation.yaml").write_text("bounding_box: [[-0.5, 0.5], [-0.5, 0.5], [0.0, 1.0]]\n")
    previous = DynamicPaths.WORLD.path
    DynamicPaths.WORLD.path = tmp_path
    try:
        yield tmp_path
    finally:
        DynamicPaths.WORLD.path = previous


def _on_env(body: typing.Callable[[EnvironmentManager, NoopHumanSimulator], typing.Awaitable[None]]) -> None:
    from arena_runtime.sim.dummy_simulator import DummySimulator
    from task_generator.constants import Constants
    from task_generator.manager.environment_manager import EnvironmentManager
    from task_generator.manager.realizer import Realizer
    from task_generator.node import TaskGenerator
    from task_generator.simulators.acoustics import AcousticsSimulatorRegistry
    from task_generator.simulators.human.noop import NoopHumanSimulator

    async def main() -> None:
        node = TaskGenerator()
        try:
            realizer = Realizer(Realizer._Configuration(x=0.0, y=0.0, prefix="env_0"))
            sim = DummySimulator(node=node, namespace=node._namespace, realizer=realizer)
            acoustics = await AcousticsSimulatorRegistry.get(Constants.AcousticsSimulator.NONE, node=node, namespace=node._namespace)
            human = NoopHumanSimulator(node=node, namespace=node._namespace, simulator=sim, realizer=realizer, acoustics=acoustics)
            await body(EnvironmentManager(node=node, simulator=sim, human_simulator=human, realizer=realizer), human)
        finally:
            node._heartbeat_stop.set()
            node.destroy_node()

    asyncio.run(main())


def test_removal_resolves_the_spawned_id_and_the_plain_name_and_skips_the_rest(world_dir: Path) -> None:
    from task_generator.shared import Obstacle, Pose, Position

    async def body(env: EnvironmentManager, human: NoopHumanSimulator) -> None:
        await env.spawn_obstacles([Obstacle(name=name, model="crate", pose=Pose(Position(float(i), 1.0))) for i, name in enumerate(("cart", "sign", "crate"))])
        assert await env.remove_obstacles([]) == []
        assert await env.remove_obstacles(["env_0/cart", "sign", "env_0/never_existed"]) == ["cart", "sign"]
        assert sorted(human._known_obstacles.keys()) == ["crate"]

    _on_env(body)


def test_a_backend_without_per_agent_removal_keeps_the_pedestrian() -> None:
    from task_generator.shared import DynamicObstacle, Pose, Position

    async def body(env: EnvironmentManager, human: NoopHumanSimulator) -> None:
        await env.spawn_dynamic_obstacles([DynamicObstacle(name="edge_0", model="arenian", pose=Pose(Position(3.0, 1.0)), waypoints=[])])
        assert await env.remove_obstacles(["env_0/edge_0"]) == []
        assert "edge_0" in human._known_obstacles.keys(), "a failed removal must not forget the entity"

    _on_env(body)
