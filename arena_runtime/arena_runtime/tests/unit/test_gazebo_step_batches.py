from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("arena_runtime.sim.gazebo_simulator.gazebo_simulator")
world_stats_pb2 = pytest.importorskip("gz.msgs10.world_stats_pb2")

from arena_rclpy_mixins.Time import Time  # noqa: E402

from arena_runtime.sim.gazebo_simulator.gazebo_simulator import GazeboHost  # noqa: E402


class _Logger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, msg: str) -> None:
        self.warnings.append(msg)


class _Host:
    def __init__(self, stats: world_stats_pb2.WorldStatistics) -> None:
        self._stats = stats
        self._logger = _Logger()


def _stats(*, paused: bool, sim: float) -> world_stats_pb2.WorldStatistics:
    msg = world_stats_pb2.WorldStatistics(paused=paused)
    msg.sim_time.sec = int(sim)
    msg.sim_time.nsec = round((sim - int(sim)) * 1e9)
    return msg


async def _finish_batch(host: _Host, at: float) -> None:
    await asyncio.sleep(0.3)
    host._stats = _stats(paused=False, sim=at - 0.01)
    await asyncio.sleep(0.3)
    host._stats = _stats(paused=True, sim=at)


def test_waits_until_gz_paused_past_the_target_after_its_batch():
    host = _Host(_stats(paused=False, sim=1.0))

    async def run() -> float:
        finisher = asyncio.ensure_future(_finish_batch(host, 2.0))
        start = asyncio.get_running_loop().time()
        await GazeboHost._await_batches_done(host, Time.from_float(1.98))  # pyright: ignore[reportArgumentType]
        await finisher
        return asyncio.get_running_loop().time() - start

    assert asyncio.run(run()) >= 0.6
    assert host._logger.warnings == []


def test_a_paused_report_from_before_the_target_does_not_count():
    host = _Host(_stats(paused=True, sim=1.0))

    async def run() -> float:
        finisher = asyncio.ensure_future(_finish_batch(host, 2.0))
        start = asyncio.get_running_loop().time()
        await GazeboHost._await_batches_done(host, Time.from_float(1.98))  # pyright: ignore[reportArgumentType]
        await finisher
        return asyncio.get_running_loop().time() - start

    assert asyncio.run(run()) >= 0.6
