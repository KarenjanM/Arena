"""Ego-relative spawn geometry: where and when an injected pedestrian must start to meet the robot on its planned route, solved offline."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import attrs

from arena_simulation_setup.shared.pathing import IsValid, Point, iter_points, point_at_arc, route_length

_SEGMENT_STEP_M = 0.25

_LEAD_OUT_SCALES = (1.0, 0.5, 0.25)

_ENCOUNTER_STEP = 0.05
_ENCOUNTER_MIN = 0.1
_ENCOUNTER_MAX = 0.9


def encounter_fractions(requested: float) -> Iterator[float]:
    yield requested
    seen = {round(requested, 6)}
    for k in range(1, int((_ENCOUNTER_MAX - _ENCOUNTER_MIN) / _ENCOUNTER_STEP) + 1):
        for candidate in (requested - k * _ENCOUNTER_STEP, requested + k * _ENCOUNTER_STEP):
            value = round(candidate, 6)
            if value in seen or not _ENCOUNTER_MIN <= value <= _ENCOUNTER_MAX:
                continue
            seen.add(value)
            yield value


_MIN_ROBOT_CLEARANCE_M = 1.0


class GeometryError(RuntimeError):
    pass


@attrs.frozen
class Encounter:
    spawn: Point
    goal: Point
    encounter: Point
    t_encounter: float
    angle_requested: float
    angle_achieved: float
    encounter_at_requested: float
    encounter_at_achieved: float
    ped_speed: float
    co_arrival_offset: float
    retries: int

    @property
    def designed_pet(self) -> float:
        return abs(self.co_arrival_offset)

    @property
    def deviated(self) -> bool:
        return abs(self.angle_achieved - self.angle_requested) > 1e-9 or abs(self.encounter_at_achieved - self.encounter_at_requested) > 1e-9


def rotate(v: Point, degrees: float) -> Point:
    rad = math.radians(degrees)
    cos_r, sin_r = math.cos(rad), math.sin(rad)
    return (v[0] * cos_r - v[1] * sin_r, v[0] * sin_r + v[1] * cos_r)


def _along(origin: Point, direction: Point, distance: float) -> Point:
    return (origin[0] + direction[0] * distance, origin[1] + direction[1] * distance)


def _sub_route(route: Sequence[Point], start_s: float, end_s: float) -> list[Point]:
    if end_s <= start_s:
        return [point_at_arc(route, start_s)[0], point_at_arc(route, end_s)[0]]

    out = [point_at_arc(route, start_s)[0]]
    traveled = 0.0
    for a, b in zip(route, route[1:], strict=False):
        seg = math.dist(a, b)
        if seg > 1e-12 and start_s < traveled + seg < end_s:
            out.append(b)
        traveled += seg
    out.append(point_at_arc(route, end_s)[0])
    return out


def _shortest_clear_lead_in(encounter: Point, direction: Point, goal: Point, travel: float, is_valid: IsValid | None, *, min_lead_in: float, step: float = 0.5, spawn_valid: IsValid | None = None) -> float | None:
    d = travel - step
    while d >= min_lead_in:
        spawn = _along(encounter, direction, -d)
        if (spawn_valid is None or spawn_valid(spawn)) and first_blocked(spawn, goal, is_valid) is None:
            return d
        d -= step
    return None


def first_blocked(a: Point, b: Point, is_valid: IsValid | None, *, step: float = _SEGMENT_STEP_M) -> Point | None:
    if is_valid is None:
        return None

    length = math.dist(a, b)
    steps = max(1, int(math.ceil(length / max(step, 1e-6))))
    for i in range(steps + 1):
        f = i / steps
        pt = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
        if not is_valid(pt):
            return pt
    return None


def angle_offsets(search: float, step: float) -> Iterator[float]:
    yield 0.0
    if search <= 0 or step <= 0:
        return
    k = 1
    while k * step <= search + 1e-9:
        yield k * step
        yield -k * step
        k += 1


def _candidates(
    angle: float,
    encounter_at: float,
    lead_out: float,
    *,
    angle_search: float,
    angle_step: float,
) -> Iterator[tuple[float, float, float]]:
    seen: set[tuple[float, float, float]] = set()
    for enc in encounter_fractions(encounter_at):
        for scale in _LEAD_OUT_SCALES:
            for offset in angle_offsets(angle_search, angle_step):
                candidate = (angle + offset, enc, lead_out * scale)
                if candidate in seen:
                    continue
                seen.add(candidate)
                yield candidate


def solve_encounter(
    route: Sequence[Point],
    *,
    robot_speed: float,
    ped_speed: float,
    approach_angle: float = 180.0,
    encounter_at: float = 0.5,
    co_arrival_offset: float = 0.0,
    lead_out: float = 3.0,
    angle_search: float = 45.0,
    angle_step: float = 15.0,
    is_valid: IsValid | None = None,
    min_robot_clearance: float = _MIN_ROBOT_CLEARANCE_M,
    min_lead_in: float = 3.0,
    spawn_valid: IsValid | None = None,
    max_lead_in: float | None = None,
) -> Encounter:
    """A pedestrian meeting the robot at `encounter_at` of the route from `approach_angle` (180 head-on, 90 from its right); raises GeometryError when none fits."""
    if robot_speed <= 0:
        raise GeometryError(f"robot_speed must be positive, got {robot_speed}")
    if ped_speed <= 0:
        raise GeometryError(f"ped_speed must be positive, got {ped_speed}")
    if not 0.0 < encounter_at < 1.0:
        raise GeometryError(f"encounter_at must be strictly between 0 and 1, got {encounter_at}")
    if lead_out <= 0:
        raise GeometryError(f"lead_out must be positive, got {lead_out}")
    if len(route) < 2:
        raise GeometryError(f"route needs at least two points, got {len(route)}")

    robot_start, robot_goal = route[0], route[-1]
    leg_length = route_length(route)
    if leg_length < 1e-9:
        raise GeometryError("robot start and goal coincide; there is no route to intercept")

    attempts = 0
    reasons: list[str] = []
    for angle, enc_at, lead in _candidates(approach_angle, encounter_at, lead_out, angle_search=angle_search, angle_step=angle_step):
        attempts += 1

        t_enc = enc_at * leg_length / robot_speed
        encounter, heading = point_at_arc(route, enc_at * leg_length)

        travel = ped_speed * (t_enc + co_arrival_offset)
        if max_lead_in is not None and travel > 1e-6:
            travel = min(travel, max_lead_in)
        if travel <= 1e-6:
            raise GeometryError(f"co_arrival_offset={co_arrival_offset:g}s cancels the pedestrian's approach (it would have to start at or past the encounter point, {travel:.2f} m away); the offset must exceed -{t_enc:.2f}s at encounter_at={enc_at:g}")

        direction = rotate(heading, angle)
        spawn = _along(encounter, direction, -travel)
        goal = _along(encounter, direction, lead)

        walk = f"({spawn[0]:.2f},{spawn[1]:.2f})->({goal[0]:.2f},{goal[1]:.2f})"

        if math.dist(spawn, robot_start) < min_robot_clearance:
            reasons.append(f"angle={angle:g}: spawn {walk[: walk.index(')') + 1]} is {math.dist(spawn, robot_start):.2f} m from the robot's start")
            continue

        blocked = first_blocked(spawn, goal, is_valid)
        reserved = spawn_valid is not None and not spawn_valid(spawn)
        if blocked is not None or reserved:
            shorter = _shortest_clear_lead_in(encounter, direction, goal, travel, is_valid, min_lead_in=min_lead_in, spawn_valid=spawn_valid)
            if shorter is None or math.dist(_along(encounter, direction, -shorter), robot_start) < min_robot_clearance:
                why = f"hits occupied space at ({blocked[0]:.2f},{blocked[1]:.2f})" if blocked is not None else "starts in space reserved for the robot"
                reasons.append(f"angle={angle:g}, encounter_at={enc_at:g}, lead_out={lead:g}: walk {walk} {why}")
                continue
            spawn = _along(encounter, direction, -shorter)

        return Encounter(
            spawn=spawn,
            goal=goal,
            encounter=encounter,
            t_encounter=t_enc,
            angle_requested=approach_angle,
            angle_achieved=angle,
            encounter_at_requested=encounter_at,
            encounter_at_achieved=enc_at,
            ped_speed=ped_speed,
            co_arrival_offset=co_arrival_offset,
            retries=attempts - 1,
        )

    detail = "; ".join(reasons[:3])
    leg = f"({robot_start[0]:.2f},{robot_start[1]:.2f})->({robot_goal[0]:.2f},{robot_goal[1]:.2f}) [{leg_length:.2f} m]"

    margin = min(min_robot_clearance, 0.4 * leg_length)
    blocked_leg = None
    if is_valid is not None:
        for probe in iter_points(_sub_route(route, margin, leg_length - margin), _SEGMENT_STEP_M):
            if not is_valid(probe):
                blocked_leg = probe
                break
    if blocked_leg is not None:
        planned = "planned route" if len(route) > 2 else "straight-line leg"
        raise GeometryError(
            f"the robot's {planned} {leg} is itself blocked at "
            f"({blocked_leg[0]:.2f},{blocked_leg[1]:.2f}), so there is no valid space on the route "
            "to design an encounter against. On a planned (bent) route this is not a wall - the "
            "planner already routed around those - so look for something occupying the route "
            "itself: base-population agents, their forbidden discs, or furniture inflated since "
            "the route was planned. On a two-point route it usually is a wall between two rooms. "
            "Widen angle_search, move encounter_at, or set place=scenario to drop the tail "
            f"without a designed encounter. ({attempts} candidate(s) tried at "
            f"ped_speed={ped_speed:g} m/s.)"
        )

    raise GeometryError(f"no navigable interception after {attempts} candidate(s) on robot leg {leg} at ped_speed={ped_speed:g} m/s (angle={approach_angle:g} +/-{angle_search:g} deg, encounter_at={encounter_at:g}, lead_out={lead_out:g} m): {detail or 'no candidates generated'}")
