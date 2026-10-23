"""Elevator cabins, door states and teleports the arena_humansim bridge hands to the engine."""

from __future__ import annotations

import asyncio
import math
import threading
import typing
import uuid

import pytest

pytest.importorskip("rclpy")

from arena_humansim_msgs.srv import NotifyStimulus, SetElevatorDoors, SetElevators, TeleportAgents
from arena_simulation_setup.shared import Elevator
from arena_simulation_setup.tree.World.model import Level, World
from arena_simulation_setup.tree.World.view import _ring_free_cabins
from arena_simulation_setup.utils.geometry import Position
from task_generator.manager.realizer import Realizer
from task_generator.simulators.human.arena_humansim.arena_humansim import ArenaHumanSimulator

ENV_OFFSET = (100.0, 50.0)
LEVEL_ORIGINS = {"0": (0.0, 0.0), "1": (30.0, 0.0), "2": (60.0, 0.0)}


def _realizer() -> Realizer:
    return Realizer(Realizer._Configuration(x=ENV_OFFSET[0], y=ENV_OFFSET[1], prefix="env_0"), level_origins=dict(LEVEL_ORIGINS))


def _adapter(realizer: Realizer) -> ArenaHumanSimulator:
    adapter = object.__new__(ArenaHumanSimulator)
    adapter._realizer = realizer
    return adapter


@pytest.mark.parametrize(
    ("door_side", "doorway", "facing_in"),
    [
        ("+x", (1.45, 0.0), math.pi),
        ("-x", (-1.45, 0.0), 0.0),
        ("+y", (0.0, 1.95), -math.pi / 2),
        ("-y", (0.0, -1.95), math.pi / 2),
    ],
)
def test_cabin_doorway_waits_outside_its_door_side_facing_the_cabin(door_side: str, doorway: tuple[float, float], facing_in: float) -> None:
    realizer = _realizer()
    elevator = realizer.realize(Elevator(name="lift", position=Position(4.0, 2.0), size=[2.0, 3.0, 2.5], door_side=door_side), "1")
    msg = _adapter(realizer)._elevator_msg(elevator)

    assert (msg.doorway.x, msg.doorway.y) == pytest.approx((34.0 + doorway[0], 2.0 + doorway[1]))
    assert math.remainder(msg.doorway.theta - facing_in, math.tau) == pytest.approx(0.0)
    assert math.remainder(msg.boarding.theta - facing_in - math.pi, math.tau) == pytest.approx(0.0)


def test_cabin_goes_to_the_engine_in_the_laid_out_frame() -> None:
    realizer = _realizer()
    elevator = realizer.realize(Elevator(name="lift", position=Position(4.0, 2.0), size=[2.0, 3.0, 2.5], door_side="+y"), "1")
    msg = _adapter(realizer)._elevator_msg(elevator)

    assert msg.name == realizer.prefix("lift", "1")
    assert (msg.boarding.x, msg.boarding.y) == pytest.approx((34.0, 2.0))
    assert [(p.x, p.y) for p in msg.footprint] == pytest.approx([(33.0, 0.5), (35.0, 0.5), (35.0, 3.5), (33.0, 3.5)])
    assert msg.door_open is False


def test_linked_levels_ring_names_each_next_realized_cabin_as_destination() -> None:
    world = World(levels={level_id: Level(id=level_id, source="floor", elevators=[Elevator(name=f"lift_{level_id}", position=Position(1.0, 1.0), door_side="+x")]) for level_id in LEVEL_ORIGINS})
    _ring_free_cabins(world)
    realizer = _realizer()
    realizer.reset_elevator_levels(world.elevator_levels())
    adapter = _adapter(realizer)
    msgs = [adapter._elevator_msg(realizer.realize(elevator, level_id)) for level_id, level in world.levels.items() for elevator in level.elevators]

    assert [msg.destination for msg in msgs] == [msgs[1].name, msgs[2].name, msgs[0].name]
    assert [msg.boarding.x for msg in msgs] == pytest.approx([1.0, 31.0, 61.0])


class _Bridge(ArenaHumanSimulator):
    """The bridge without its process-global task mode registration."""

    @classmethod
    def _register_task_modes(cls) -> None:
        pass


class _Engine:
    """In-process stand-in for the engine's elevator, teleport and stimulus services, recording every request."""

    def __init__(self, node: typing.Any) -> None:
        self.elevators: list[list[str]] = []
        self.doors: list[dict[str, bool]] = []
        self.teleports: list[TeleportAgents.Request] = []
        self.stimuli: list[tuple[int, str, float]] = []
        node.create_service(SetElevators, str(node.service_namespace(ArenaHumanSimulator.SERVICE_SET_ELEVATORS)), self._set_elevators)
        node.create_service(SetElevatorDoors, str(node.service_namespace(ArenaHumanSimulator.SERVICE_SET_ELEVATOR_DOORS)), self._set_doors)
        node.create_service(TeleportAgents, str(node.service_namespace(ArenaHumanSimulator.SERVICE_TELEPORT_AGENTS)), self._teleport)
        node.create_service(NotifyStimulus, str(node.service_namespace(ArenaHumanSimulator.SERVICE_NOTIFY_STIMULUS)), self._notify)

    def _set_elevators(self, request: SetElevators.Request, response: SetElevators.Response) -> SetElevators.Response:
        self.elevators.append([elevator.name for elevator in request.elevators])
        response.success = True
        return response

    def _set_doors(self, request: SetElevatorDoors.Request, response: SetElevatorDoors.Response) -> SetElevatorDoors.Response:
        self.doors.append(dict(zip(request.name, request.open, strict=True)))
        response.success = True
        return response

    def _teleport(self, request: TeleportAgents.Request, response: TeleportAgents.Response) -> TeleportAgents.Response:
        self.teleports.append(request)
        response.success = True
        return response

    def _notify(self, request: NotifyStimulus.Request, response: NotifyStimulus.Response) -> NotifyStimulus.Response:
        self.stimuli.append((request.agent_id, request.stimulus, request.intensity))
        response.success = True
        return response


def _on_bridge(body: typing.Callable[[ArenaHumanSimulator, _Engine], typing.Awaitable[None]]) -> None:
    import rclpy.executors
    from arena_rclpy_mixins import ArenaMixinNode
    from arena_rclpy_mixins.shared import Namespace
    from arena_runtime.sim.dummy_simulator import DummySimulator
    from task_generator.constants import Constants
    from task_generator.simulators.acoustics import AcousticsSimulatorRegistry

    async def main() -> None:
        suffix = uuid.uuid4().hex[:8]
        namespace = Namespace(f"/humansim_elevators_{suffix}")
        node = ArenaMixinNode(f"humansim_elevators_{suffix}", namespace=str(namespace))
        realizer = _realizer()
        sim = DummySimulator(node=node, namespace=namespace, realizer=realizer)
        acoustics = await AcousticsSimulatorRegistry.get(Constants.AcousticsSimulator.NONE, node=node, namespace=namespace)
        bridge = _Bridge(node=node, namespace=namespace, simulator=sim, realizer=realizer, acoustics=acoustics)
        engine = _Engine(node)
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()
        try:
            for client in (bridge._set_elevators_client, bridge._set_elevator_doors_client, bridge._teleport_agents_client, bridge._notify_stimulus_client):
                assert await client.ensure(timeout_sec=5.0)
            await body(bridge, engine)
        finally:
            await sim.stop_mechanisms()
            executor.shutdown()
            thread.join(timeout=2.0)
            node.destroy_node()

    asyncio.run(main())


def test_teleport_reaches_the_engine_by_agent_id_in_the_engine_frame() -> None:
    from arena_people_msgs.msg import Pedestrian, Pedestrians

    async def body(bridge: ArenaHumanSimulator, engine: _Engine) -> None:
        bridge._agent_names = {7: "/env_0/walker"}
        walker = Pedestrian(name="/env_0/walker")
        walker.pose.orientation.z = math.sin(math.pi / 4)
        walker.pose.orientation.w = math.cos(math.pi / 4)
        bridge._on_arena_peds(Pedestrians(pedestrians=[walker]))

        assert await bridge.pedestrian_teleport({"/env_0/walker": (131.0, 52.0), "12": (100.5, 49.0)})

        (request,) = engine.teleports
        assert list(request.agent_id) == [7, 12]
        assert [(pose.x, pose.y) for pose in request.pose] == pytest.approx([(31.0, 2.0), (0.5, -1.0)])
        assert [pose.theta for pose in request.pose] == pytest.approx([math.pi / 2, 0.0])
        assert set(bridge._teleport_stamps) == {7, 12}
        assert engine.stimuli == [(7, "teleported", 1.0), (12, "teleported", 1.0)]

    _on_bridge(body)


def test_teleport_of_an_unknown_pedestrian_fails_but_moves_the_known_ones() -> None:
    async def body(bridge: ArenaHumanSimulator, engine: _Engine) -> None:
        bridge._agent_names = {7: "/env_0/walker"}

        assert not await bridge.pedestrian_teleport({"/env_0/walker": (131.0, 52.0), "/env_0/ghost": (100.0, 50.0)})
        assert [list(request.agent_id) for request in engine.teleports] == [[7]]

    _on_bridge(body)


def test_world_cabins_reach_the_engine_and_leave_with_the_world() -> None:
    from task_generator.simulators.human.utils import ObstacleLayer

    async def body(bridge: ArenaHumanSimulator, engine: _Engine) -> None:
        realizer = bridge._realizer
        cabins = [realizer.realize(Elevator(name=f"lift_{level_id}", position=Position(1.0, 1.0), door_side="+x"), level_id) for level_id in ("0", "1")]

        assert await bridge.spawn_elevators(cabins)
        await bridge.remove_obstacles(purge=ObstacleLayer.WORLD)
        await bridge.remove_obstacles(purge=ObstacleLayer.WORLD)
        assert await bridge.spawn_elevators([])

        assert engine.elevators == [[cabin.name for cabin in cabins], []]

    _on_bridge(body)


def test_door_changes_reach_the_engine_by_cabin_name() -> None:
    async def body(bridge: ArenaHumanSimulator, engine: _Engine) -> None:
        await bridge.elevator_doors_changed({"env_0/0/lift_0": True, "env_0/1/lift_1": False})

        assert engine.doors == [{"env_0/0/lift_0": True, "env_0/1/lift_1": False}]

    _on_bridge(body)
