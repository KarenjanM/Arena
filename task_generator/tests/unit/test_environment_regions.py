"""Flow regions reach the human simulator in the env's frame, so the engine frame puts them back on the scenario polygon."""

from __future__ import annotations

import asyncio

ORIGIN = (30.0, -12.0)
POLYGON = [(1.0, 1.0), (4.0, 1.0), (4.0, 3.0), (1.0, 3.0)]


def test_flow_regions_are_realized_into_the_env_frame():
    from arena_runtime.sim.dummy_simulator import DummySimulator

    from task_generator.constants import Constants
    from task_generator.manager.environment_manager import EnvironmentManager
    from task_generator.manager.realizer import Realizer
    from task_generator.node import TaskGenerator
    from task_generator.shared import Position, Region
    from task_generator.simulators.acoustics import AcousticsSimulatorRegistry
    from task_generator.simulators.human.noop import NoopHumanSimulator

    async def main() -> None:
        node = TaskGenerator()
        try:
            realizer = Realizer(Realizer._Configuration(x=ORIGIN[0], y=ORIGIN[1], prefix="env_1"))
            sim = DummySimulator(node=node, namespace=node._namespace, realizer=realizer)
            acoustics = await AcousticsSimulatorRegistry.get(Constants.AcousticsSimulator.NONE, node=node, namespace=node._namespace)
            human = NoopHumanSimulator(node=node, namespace=node._namespace, simulator=sim, realizer=realizer, acoustics=acoustics)
            env = EnvironmentManager(node=node, simulator=sim, human_simulator=human, realizer=realizer)
            region = Region(name="queue", type="source", polygon=[Position(x, y) for x, y in POLYGON])
            await env.setup_regions([region])
            config = realizer.get_config()
            engine = [(v.x - config.x, v.y - config.y) for v in human._known_regions["queue"].polygon]
            assert engine == POLYGON
            assert region.polygon == [Position(x, y) for x, y in POLYGON]
        finally:
            node._heartbeat_stop.set()
            node.destroy_node()

    asyncio.run(main())
