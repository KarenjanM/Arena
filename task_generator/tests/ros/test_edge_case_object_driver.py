"""When each object-timeline event fires, and what spawn and despawn failures mean."""

from __future__ import annotations

import asyncio

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


def _events(*specs):
    from task_generator.tasks.obstacles.edge_case.objects import ObjectEvent, ResolvedEvent

    out = []
    for action, entity, t in specs:
        event = ObjectEvent(action=action, entity=entity, model="m" if action == "spawn" else "", at_t=t)
        out.append(ResolvedEvent(event=event, t=t, pose=(1.0, 2.0) if action == "spawn" else None))
    return out


class _Harness:
    def __init__(self, *, spawn_result="env_0/cart", spawn_raises=None):
        import rclpy.logging
        from task_generator.tasks.obstacles.edge_case.object_driver import ObjectDriver

        self.spawned: list[str] = []
        self.despawned: list[str] = []
        self.submitted: list[object] = []
        self._spawn_result = spawn_result
        self._spawn_raises = spawn_raises
        self.now = 0.0

        async def spawn(event):
            if self._spawn_raises is not None:
                raise self._spawn_raises
            self.spawned.append(event.entity)
            return self._spawn_result

        async def despawn(event, entity_id):
            self.despawned.append(entity_id)
            return True

        def submit(coroutine):
            self.submitted.append(coroutine)

        self.driver = ObjectDriver(spawn=spawn, despawn=despawn, submit=submit, logger=rclpy.logging.get_logger("pytest_objects"))

    def tick(self, at=None):
        if at is not None:
            self.now = at
        self.driver.tick(self.now)

    def drain(self):
        for coroutine in self.submitted:
            asyncio.run(coroutine)
        self.submitted.clear()

    def close(self):
        for coroutine in self.submitted:
            coroutine.close()


@pytest.fixture()
def harness():
    h = _Harness()
    yield h
    h.close()


def test_the_clock_starts_on_the_first_tick_not_on_arm(harness) -> None:
    harness.driver.arm(_events(("spawn", "cart", 1.0)))
    harness.tick(at=1000.0)

    assert harness.submitted == []

    harness.tick(at=1001.5)
    harness.drain()
    assert harness.spawned == ["cart"]


def test_a_clock_reading_of_zero_is_not_a_start(harness) -> None:
    harness.driver.arm(_events(("spawn", "cart", 1.0)))
    harness.tick(at=0.0)
    harness.tick(at=0.0)

    assert harness.submitted == []
    assert harness.driver.pending() == 1


def test_an_event_fires_once(harness) -> None:
    harness.driver.arm(_events(("spawn", "cart", 1.0)))
    harness.tick(at=100.0)
    for t in (101.0, 102.0, 103.0):
        harness.tick(at=t)
    harness.drain()

    assert harness.spawned == ["cart"]


def test_events_before_their_time_do_not_fire(harness) -> None:
    harness.driver.arm(_events(("spawn", "late", 30.0)))
    harness.tick(at=10.0)
    harness.tick(at=25.0)

    assert harness.submitted == []
    assert harness.driver.pending() == 1


def test_a_pause_that_outlasts_several_events_fires_all_of_them(harness) -> None:
    harness.driver.arm(_events(("spawn", "a", 1.0), ("spawn", "b", 2.0), ("spawn", "c", 3.0)))
    harness.tick(at=100.0)
    harness.tick(at=110.0)
    harness.drain()

    assert harness.spawned == ["a", "b", "c"]
    assert harness.driver.pending() == 0


def test_pending_counts_what_never_fired(harness) -> None:
    harness.driver.arm(_events(("spawn", "a", 1.0), ("spawn", "b", 60.0)))
    harness.tick(at=100.0)
    harness.tick(at=102.0)
    harness.drain()

    assert harness.driver.pending() == 1, "an episode that ended before its timeline did"


def test_arming_a_new_timeline_forgets_the_old_one(harness) -> None:
    harness.driver.arm(_events(("spawn", "old", 1.0)))
    harness.tick(at=100.0)
    harness.driver.arm(_events(("spawn", "new", 1.0)))
    harness.tick(at=100.0)
    harness.tick(at=102.0)
    harness.drain()

    assert harness.spawned == ["new"]


def test_a_despawn_uses_the_id_its_spawn_returned(harness) -> None:
    harness.driver.arm(_events(("spawn", "cart", 1.0), ("despawn", "cart", 5.0)))
    harness.tick(at=100.0)
    harness.tick(at=102.0)
    harness.drain()
    harness.tick(at=106.0)
    harness.drain()

    assert harness.despawned == ["env_0/cart"]


def test_a_despawn_whose_spawn_failed_is_skipped_with_a_reason() -> None:
    harness = _Harness(spawn_raises=RuntimeError("no such model"))
    try:
        harness.driver.arm(_events(("spawn", "cart", 1.0), ("despawn", "cart", 5.0)))
        harness.tick(at=100.0)
        harness.tick(at=102.0)
        harness.drain()
        harness.tick(at=106.0)
        harness.drain()

        spawn, despawn = harness.driver.outcomes()
        assert spawn.status == "failed"
        assert "no such model" in spawn.reason
        assert despawn.status == "skipped"
        assert "nothing was spawned" in despawn.reason
        assert harness.despawned == []
    finally:
        harness.close()


def test_a_spawn_that_returns_no_id_is_a_failure() -> None:
    harness = _Harness(spawn_result="")
    try:
        harness.driver.arm(_events(("spawn", "cart", 1.0)))
        harness.tick(at=100.0)
        harness.tick(at=102.0)
        harness.drain()

        (outcome,) = harness.driver.outcomes()
        assert outcome.status == "failed"
    finally:
        harness.close()


def test_outcomes_record_designed_and_fired_times(harness) -> None:
    harness.driver.arm(_events(("spawn", "cart", 3.0)))
    harness.tick(at=100.0)
    harness.tick(at=107.5)
    harness.drain()

    (outcome,) = harness.driver.outcomes()
    assert outcome.t_designed == 3.0
    assert outcome.t_fired == pytest.approx(7.5)
    assert outcome.status == "ok"
    assert outcome.entity_id == "env_0/cart"


def test_outcomes_are_serialisable() -> None:
    import json

    harness = _Harness()
    try:
        harness.driver.arm(_events(("spawn", "cart", 1.0)))
        harness.tick(at=100.0)
        harness.tick(at=102.0)
        harness.drain()

        json.dumps([o.as_dict() for o in harness.driver.outcomes()])
    finally:
        harness.close()


def test_an_empty_timeline_is_inert(harness) -> None:
    harness.driver.arm([])
    harness.tick(at=100.0)
    harness.tick(at=1000.0)

    assert harness.driver.outcomes() == []
    assert harness.driver.pending() == 0
