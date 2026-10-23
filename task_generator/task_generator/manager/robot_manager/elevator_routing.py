"""Expand GoTo phases into elevator rides: explicit `via` elevators plus automatic cross-level routing."""

from __future__ import annotations

import math
import typing

import attrs

from task_generator.shared import Orientation, Pose, Position
from task_generator.tasks.robots.request import ElevatorLegPhase, ElevatorRidePhase, GoToPhase, LoopPhase

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

    from task_generator.manager.world_manager.world_manager import ElevatorHop, ElevatorRide, WorldManager
    from task_generator.tasks.robots.request import TaskPhase

VIA_DESTINATION_SEP = '>'


def parse_via(entry: str) -> tuple[str, str]:
    """`"<elevator>"` or `"<elevator>><destination>"` -> (elevator, destination or '')."""
    name, _, to = entry.partition(VIA_DESTINATION_SEP)
    return name.strip(), to.strip()


MIN_RIDE_TOLERANCE = 0.05


def ride_tolerance(boarding_radius: float, goal_tolerance: float) -> float:
    """Ride arrival tolerance: the goal tolerance capped by the boarding radius, `MIN_RIDE_TOLERANCE` when the robot barely fits."""
    return min(goal_tolerance, boarding_radius) if boarding_radius > 0.0 else MIN_RIDE_TOLERANCE


def ride_phase(ride: ElevatorRide, orientation: Orientation, goal_tolerance: float) -> ElevatorRidePhase:
    """Ride phase for one hop: arrive in the boarding cabin, complete once the robot is in the destination cabin."""
    hop = ride.hop
    return ElevatorRidePhase(
        pose=Pose(position=ride.board_target, orientation=orientation),
        target=hop.board_name,
        tolerance_radius=ride_tolerance(ride.boarding_radius, goal_tolerance),
        tolerance_angle=math.pi,
        hold_time=0.0,
        until=None if hop.dest_position is None else f"robot in {hop.dest_name}",
        destination=hop.target,
        exit_cabin="" if hop.dest_position is None else hop.dest_name,
    )


LEG_TOLERANCE = 1.6
EXIT_CLEARANCE = 0.1


def exit_tolerance(door_standoff: float) -> float:
    """Exit leg tolerance, met only with the robot center `EXIT_CLEARANCE` past the door face."""
    return min(LEG_TOLERANCE, door_standoff - EXIT_CLEARANCE)


def ride_phases(ride: ElevatorRide, orientation: Orientation, goal_tolerance: float) -> list[GoToPhase]:
    """Door-aligned approach leg, the ride, and a door-aligned exit leg out of the cabin (exit omitted when unknown)."""
    phases: list[GoToPhase] = [
        ElevatorLegPhase(pose=Pose(position=ride.board_approach, orientation=orientation), tolerance_radius=LEG_TOLERANCE, tolerance_angle=math.pi, hold_time=0.0),
        ride_phase(ride, orientation, goal_tolerance),
    ]
    if ride.dest_exit is not None:
        phases.append(ElevatorLegPhase(pose=Pose(position=ride.dest_exit, orientation=orientation), tolerance_radius=exit_tolerance(ride.door_standoff), tolerance_angle=math.pi, hold_time=0.0))
    return phases


def _via_toward(world_manager: WorldManager, vias: Sequence[str], index: int, goal: Position) -> Position:
    """Where the robot heads after ride `index`: the next via boarding cabin, else the goal."""
    if index + 1 < len(vias):
        position = world_manager.elevator_position(parse_via(vias[index + 1])[0])
        if position is not None:
            return position
    return goal


def route_phases(phases: Sequence[TaskPhase], start: Position | None, world_manager: WorldManager, *, robot_radius: float, goal_tolerance: float) -> list[TaskPhase]:
    """Insert elevator rides for a robot disc of `robot_radius` before GoTo phases in the abstract frame: explicit `via` rides, then the BFS route to the goal level."""
    current_level = world_manager.level_of_point(start.x, start.y) if start is not None else ""

    def rides(hops: Sequence[ElevatorHop], orientation: Orientation) -> list[GoToPhase]:
        return [p for hop in hops for p in ride_phases(world_manager.elevator_ride(hop, robot_radius), orientation, goal_tolerance)]

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
                    routed.extend(rides(world_manager.elevator_hops(current_level, hop.board_level), orientation))
                routed.extend(rides([hop], orientation))
                current_level = hop.dest_level
            goal_level = world_manager.level_of_point(goal.x, goal.y)
            if current_level and goal_level and goal_level != current_level:
                routed.extend(rides(world_manager.elevator_hops(current_level, goal_level), orientation))
            if goal_level:
                current_level = goal_level
            routed.append(attrs.evolve(phase, via=()) if phase.via else phase)
        return routed

    return route(phases)
