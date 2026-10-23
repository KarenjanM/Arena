"""Expand GoTo phases into elevator rides: explicit `via` elevators plus automatic cross-level routing."""

from __future__ import annotations

import math
import typing

import attrs

from task_generator.shared import Orientation, Pose, Position
from task_generator.tasks.robots.request import ElevatorLegPhase, ElevatorRidePhase, GoToPhase, LoopPhase

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

    from task_generator.manager.world_manager.world_manager import ElevatorHop, WorldManager
    from task_generator.tasks.robots.request import TaskPhase

VIA_DESTINATION_SEP = '>'


def parse_via(entry: str) -> tuple[str, str]:
    """`"<elevator>"` or `"<elevator>><destination>"` -> (elevator, destination or '')."""
    name, _, to = entry.partition(VIA_DESTINATION_SEP)
    return name.strip(), to.strip()


def ride_phase(hop: ElevatorHop, orientation: Orientation) -> ElevatorRidePhase:
    """Ride phase for one hop: arrive in the boarding cabin, complete once the robot is in the destination cabin."""
    return ElevatorRidePhase(
        pose=Pose(position=hop.board_target or hop.board_position, orientation=orientation),
        target=hop.board_name,
        tolerance_angle=math.pi,
        hold_time=0.0,
        until=None if hop.dest_position is None else f"robot in {hop.dest_name}",
        destination=hop.target,
    )


LEG_TOLERANCE = 1.6


def ride_phases(hop: ElevatorHop, orientation: Orientation) -> list[GoToPhase]:
    """Door-aligned approach leg, the ride, and a door-aligned exit leg (legs omitted when unknown)."""
    phases: list[GoToPhase] = []
    if hop.board_approach is not None:
        phases.append(ElevatorLegPhase(pose=Pose(position=hop.board_approach, orientation=orientation), tolerance_radius=LEG_TOLERANCE, tolerance_angle=math.pi, hold_time=0.0))
    phases.append(ride_phase(hop, orientation))
    if hop.dest_exit is not None:
        phases.append(ElevatorLegPhase(pose=Pose(position=hop.dest_exit, orientation=orientation), tolerance_radius=LEG_TOLERANCE, tolerance_angle=math.pi, hold_time=0.0))
    return phases


def _via_toward(world_manager: WorldManager, vias: Sequence[str], index: int, goal: Position) -> Position:
    """Where the robot heads after ride `index`: the next via boarding cabin, else the goal."""
    if index + 1 < len(vias):
        position = world_manager.elevator_position(parse_via(vias[index + 1])[0])
        if position is not None:
            return position
    return goal


def route_phases(phases: Sequence[TaskPhase], start: Position | None, world_manager: WorldManager) -> list[TaskPhase]:
    """Insert elevator rides before GoTo phases in the abstract frame: explicit `via` rides, then the BFS route to the goal level."""
    current_level = world_manager.level_of_point(start.x, start.y) if start is not None else ""

    def route(phases: Sequence[TaskPhase]) -> list[TaskPhase]:
        nonlocal current_level
        routed: list[TaskPhase] = []
        for phase in phases:
            if isinstance(phase, LoopPhase):
                routed.append(attrs.evolve(phase, phases=route(phase.phases)))
                continue
            if not isinstance(phase, GoToPhase) or isinstance(phase, (ElevatorRidePhase, ElevatorLegPhase)) or phase.pose is None:
                routed.append(phase)
                continue
            goal = phase.pose.position
            orientation = phase.pose.orientation
            for index, entry in enumerate(phase.via):
                name, to = parse_via(entry)
                hop = world_manager.elevator_hop(name, to=to, toward=_via_toward(world_manager, phase.via, index, goal))
                if current_level and hop.board_level != current_level:
                    routed.extend(p for h in world_manager.elevator_hops(current_level, hop.board_level) for p in ride_phases(h, orientation))
                routed.extend(ride_phases(hop, orientation))
                current_level = hop.dest_level
            goal_level = world_manager.level_of_point(goal.x, goal.y)
            if current_level and goal_level and goal_level != current_level:
                routed.extend(p for h in world_manager.elevator_hops(current_level, goal_level) for p in ride_phases(h, orientation))
            if goal_level:
                current_level = goal_level
            routed.append(attrs.evolve(phase, via=()) if phase.via else phase)
        return routed

    return route(phases)
