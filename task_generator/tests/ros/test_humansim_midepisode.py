from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("rclpy")
    pytest.importorskip("arena_humansim_msgs.srv")


def _bridge():
    from task_generator.simulators.human.arena_humansim.arena_humansim import ArenaHumanSimulator

    class _Bridge(ArenaHumanSimulator):
        _logger = SimpleNamespace(warning=lambda msg: None, error=lambda msg: None, info=lambda msg: None)

    return object.__new__(_Bridge)


def test_stimulate_resolves_names_and_counts_who_was_reached() -> None:
    sim = _bridge()
    sim._known_obstacles = {"talker_1": SimpleNamespace(obstacle=SimpleNamespace(sim_path="env_0/talker_1"))}
    sim._agent_names = {3: "env_0/talker_1", 4: "env_0/talker_2"}
    calls: list[tuple[int, str, float]] = []

    async def notify(agent_id: int, stimulus: str, intensity: float) -> bool:
        calls.append((agent_id, stimulus, intensity))
        return agent_id == 3

    sim.notify_stimulus = notify
    # a plain name and a sim_path both resolve; the refused one and the unknown one are not counted
    assert asyncio.run(sim.stimulate_agents(["talker_1", "env_0/talker_2", "ghost"], "alarm", 1.0)) == 1
    assert calls == [(3, "alarm", 1.0), (4, "alarm", 1.0)]


def test_reroute_points_leave_the_env_offset_like_spawns() -> None:
    from task_generator.manager.realizer import Realizer

    sim = _bridge()
    sim._realizer = Realizer(Realizer._Configuration(x=5.0, y=5.0))
    sim._agent_names = {7: "env_0/walker"}
    sent = []

    async def call(request):
        sent.append(request)
        return SimpleNamespace(success=True, message="")

    sim._set_waypoints_client = SimpleNamespace(call_timeout=call)
    # a rally target realized into map (centroid 10.0, 17.1 + the env anchor) reaches the engine in the world frame
    assert asyncio.run(sim._set_waypoints_impl({"env_0/walker": [(15.0, 22.1)]}))
    (request,) = sent
    assert (request.waypoints.points[0].pose.x, request.waypoints.points[0].pose.y) == pytest.approx((10.0, 17.1))
