"""Criticality measured from a sampled trajectory: outcomes, as opposed to the designed encounter. No ROS."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import attrs

Point = tuple[float, float]

# same as FREEZE_SPEED_THRESHOLD in the robot metrics
FREEZE_SPEED = 0.05

PERSONAL_SPACE = 0.5

# nav2 is still bringing costmaps up; counting this as freeze would score every episode frozen
SETTLE_S = 2.0


@attrs.frozen
class Sample:
    t: float
    robot: Point
    robot_vel: Point = (0.0, 0.0)
    robot_radius: float = 0.3
    peds: dict[str, tuple[Point, Point, float]] = attrs.Factory(dict)


# None means "not observed", never zero
@attrs.frozen
class Criticality:
    min_ttc_s: float | None = None
    min_ttc_with: str = ""
    min_ttc_at: float | None = None

    min_clearance_m: float | None = None
    min_clearance_with: str = ""
    min_clearance_at: float | None = None

    pet_s: float | None = None
    freeze_duration_s: float = 0.0
    intrusion_time_s: float = 0.0

    collided: bool = False
    collision_fault: str = ""
    collision_with: str = ""
    collision_at: float | None = None
    collision_robot_closing: float | None = None
    collision_ped_closing: float | None = None
    samples: int = 0
    duration_s: float = 0.0

    @property
    def observed(self) -> bool:
        return self.samples > 0 and self.min_clearance_m is not None


def time_to_collision(
    p_a: Point,
    v_a: Point,
    r_a: float,
    p_b: Point,
    v_b: Point,
    r_b: float,
) -> float | None:
    dpx, dpy = p_b[0] - p_a[0], p_b[1] - p_a[1]
    dvx, dvy = v_b[0] - v_a[0], v_b[1] - v_a[1]
    r = r_a + r_b

    a = dvx * dvx + dvy * dvy
    if a < 1e-12:
        return None
    b = 2.0 * (dpx * dvx + dpy * dvy)
    c = dpx * dpx + dpy * dpy - r * r

    if c <= 0.0:
        return 0.0

    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    sqrt_disc = math.sqrt(disc)
    for root in ((-b - sqrt_disc) / (2 * a), (-b + sqrt_disc) / (2 * a)):
        if root >= 0.0:
            return root
    return None


def _speed(v: Point) -> float:
    return math.hypot(v[0], v[1])


def post_encroachment_time(samples: Sequence[Sample], encounter: Point) -> float | None:
    if not samples:
        return None

    near = 1.5

    def crossing_time(track: Iterable[tuple[float, Point]]) -> float | None:
        best_t, best_d = None, math.inf
        for t, p in track:
            d = math.dist(p, encounter)
            if d < best_d:
                best_t, best_d = t, d
        return best_t if best_d <= near else None

    t_rob = crossing_time((s.t, s.robot) for s in samples)
    if t_rob is None:
        return None

    names = {n for s in samples for n in s.peds}
    best: float | None = None
    for name in names:
        t_ped = crossing_time((s.t, s.peds[name][0]) for s in samples if name in s.peds)
        if t_ped is None:
            continue
        pet = abs(t_ped - t_rob)
        if best is None or pet < best:
            best = pet
    return best


def score(
    samples: Sequence[Sample],
    *,
    encounter: Point | None = None,
    only: Sequence[str] | None = None,
    settle_s: float = SETTLE_S,
) -> Criticality:
    if not samples:
        return Criticality()

    t0 = samples[0].t
    wanted = set(only) if only else None

    min_ttc: float | None = None
    ttc_with, ttc_at = "", None
    min_clear: float | None = None
    clear_with, clear_at = "", None
    freeze = 0.0
    intrusion = 0.0
    collided = False
    fault, fault_with, fault_at = "", "", None
    fault_rc: float | None = None
    fault_pc: float | None = None

    # forward hold: crediting dt to the sample that ends it would count the straddling window as post-settle
    for i, s in enumerate(samples):
        dt = (samples[i + 1].t - s.t) if i + 1 < len(samples) else 0.0
        dt = max(0.0, dt)

        if s.t - t0 >= settle_s and _speed(s.robot_vel) < FREEZE_SPEED:
            freeze += dt

        intruding = False
        for name, (p, v, r) in s.peds.items():
            if wanted is not None and name not in wanted:
                continue

            clearance = math.dist(s.robot, p) - (s.robot_radius + r)
            if min_clear is None or clearance < min_clear:
                min_clear, clear_with, clear_at = clearance, name, s.t - t0
            if clearance <= 0.0:
                if not collided:
                    dx, dy = p[0] - s.robot[0], p[1] - s.robot[1]
                    norm = math.hypot(dx, dy) or 1.0
                    ux, uy = dx / norm, dy / norm
                    rc = s.robot_vel[0] * ux + s.robot_vel[1] * uy
                    pc = -(v[0] * ux + v[1] * uy)
                    r_moving, p_moving = rc >= FREEZE_SPEED, pc >= FREEZE_SPEED
                    fault = "mutual" if r_moving and p_moving else "robot" if r_moving else "pedestrian" if p_moving else "static"
                    fault_with, fault_at, fault_rc, fault_pc = name, s.t - t0, rc, pc
                collided = True
            if clearance < PERSONAL_SPACE:
                intruding = True

            ttc = time_to_collision(s.robot, s.robot_vel, s.robot_radius, p, v, r)
            if ttc is not None and (min_ttc is None or ttc < min_ttc):
                min_ttc, ttc_with, ttc_at = ttc, name, s.t - t0

        if intruding:
            intrusion += dt

    return Criticality(
        min_ttc_s=min_ttc,
        min_ttc_with=ttc_with,
        min_ttc_at=ttc_at,
        min_clearance_m=min_clear,
        min_clearance_with=clear_with,
        min_clearance_at=clear_at,
        pet_s=None if encounter is None else post_encroachment_time(samples, encounter),
        freeze_duration_s=freeze,
        intrusion_time_s=intrusion,
        collided=collided,
        collision_fault=fault,
        collision_with=fault_with,
        collision_at=fault_at,
        collision_robot_closing=fault_rc,
        collision_ped_closing=fault_pc,
        samples=len(samples),
        duration_s=samples[-1].t - t0,
    )


def _bare(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def formation_hold(samples: Sequence[Sample], members: Sequence[str], *, radius: float = 1.5, centre: Point | None = None) -> dict[str, Any]:
    wanted = {str(m) for m in members}
    out: dict[str, Any] = {"members": sorted(wanted), "seen": False, "formed_at": None, "held_s": None, "broke_at": None, "arrived": 0}
    if not wanted or not samples:
        return out
    t0 = samples[0].t
    ref: Point | None = centre
    formed_at: float | None = None
    held_until: float | None = None
    for sample in samples:
        present = {_bare(n): v[0] for n, v in sample.peds.items() if _bare(n) in wanted}
        if not present:
            continue
        out["seen"] = True
        if ref is not None:
            out["arrived"] = max(out["arrived"], sum(1 for p in present.values() if math.dist(p, ref) <= radius))
        if len(present) < len(wanted):
            if formed_at is None:
                continue
        if ref is None:
            ref = (sum(p[0] for p in present.values()) / len(present), sum(p[1] for p in present.values()) / len(present))
        together = all(math.dist(p, ref) <= radius for p in present.values())
        if formed_at is None:
            if together:
                formed_at = sample.t - t0
                held_until = formed_at
            continue
        if not together:
            out["broke_at"] = sample.t - t0
            break
        held_until = sample.t - t0
    if formed_at is not None:
        out["formed_at"] = formed_at
        out["held_s"] = (held_until or formed_at) - formed_at
    return out
