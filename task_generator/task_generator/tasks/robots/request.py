"""Typed task requests submitted to robots, the shared types plus their action kinds."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from arena_robots.task_kinds import TaskKind
from arena_simulation_setup.shared.task import GoToPhase, LoopPhase, PlayGesturePhase, ReachPhase, TaskPhase, TaskRequest


def kind_of(phase: TaskPhase) -> TaskKind:
    """Action kind dispatched for a phase, loops have none."""
    return TaskKind(phase.kind)


def leaf_phases(phases: Iterable[TaskPhase]) -> Iterator[TaskPhase]:
    """The dispatchable phases in order, loop bodies descended into."""
    for phase in phases:
        if isinstance(phase, LoopPhase):
            yield from leaf_phases(phase.phases)
        else:
            yield phase


__all__ = [
    "TaskKind",
    "TaskPhase",
    "GoToPhase",
    "LoopPhase",
    "ReachPhase",
    "PlayGesturePhase",
    "TaskRequest",
    "kind_of",
    "leaf_phases",
]
