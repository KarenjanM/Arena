from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import attrs

Point = tuple[float, float]

# so a robot that has stopped still counts as arriving, slowly
ETA_MIN_SPEED_MPS = 0.1


@attrs.define
class Trigger:
    robot_within: float | None = None
    robot_beyond: float | None = None
    robot_eta: float | None = None
    # None: the centroid of the agents the plan owns, as placed, resolved by the caller
    of: str | Point | None = None

    @property
    def distance(self) -> float:
        return float(next(v for v in (self.robot_within, self.robot_beyond, self.robot_eta) if v is not None))

    def satisfied(self, distance: float | None, speed: float | None = None) -> bool:
        """An unknown distance (respawn window, unresolved reference) never satisfies: firing blind would report a case that did not run."""
        if distance is None or not math.isfinite(distance):
            return False
        if self.robot_eta is not None:
            return speed is not None and distance / max(speed, ETA_MIN_SPEED_MPS) <= self.robot_eta
        if self.robot_within is not None:
            return distance <= self.robot_within
        return distance >= self.robot_beyond

    def describe(self, reference: Point | None = None) -> str:
        where = f" of {reference[0]:.2f},{reference[1]:.2f}" if reference else ""
        if self.robot_eta is not None:
            return f"robot due in {self.robot_eta:g}s{where}"
        side = "within" if self.robot_within is not None else "beyond"
        return f"robot {side} {self.distance:g}m{where}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "robot_within": self.robot_within,
            "robot_beyond": self.robot_beyond,
            "robot_eta": self.robot_eta,
            "of": list(self.of) if isinstance(self.of, (list, tuple)) else self.of,
        }


def centroid(points: Sequence[Point]) -> Point | None:
    if not points:
        return None
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def distance_to(robot: Point | None, reference: Point | None) -> float | None:
    if robot is None or reference is None:
        return None
    return math.hypot(robot[0] - reference[0], robot[1] - reference[1])
