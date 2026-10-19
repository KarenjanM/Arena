from __future__ import annotations

import enum
import math
from typing import Any

import attrs

Point = tuple[float, float]

# successive shuffles land well apart instead of walking round a circle
_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


class Mode(enum.StrEnum):
    RALLY = "rally"
    """Send every owned agent to its target, once."""

    TRACK_ROBOT = "track_robot"
    """Re-issue the robot's current position as the goal."""

    SHUFFLE = "shuffle"
    """Re-issue a fresh point near where the agent started, every `period`."""

    HOLD = "hold"
    """Send each agent to its hold point, then release it to its resume point."""

    STIMULUS = "stimulus"
    """Raise a need on every owned agent, once."""


class WaypointError(Exception):
    pass


@attrs.define
class Plan:
    mode: Mode
    homes: dict[str, Point] = attrs.Factory(dict)
    at: float = 0.0
    period: float = 4.0
    radius: float = 3.0
    once: bool = False
    targets: dict[str, Point] = attrs.Factory(dict)
    duration: float = 0.0
    resume: dict[str, Point] = attrs.Factory(dict)
    points: dict[str, Point] = attrs.Factory(dict)
    need: str = ""
    intensity: float = 1.0
    label: str = ""

    def __attrs_post_init__(self) -> None:
        if not self.homes:
            raise WaypointError(f"{self.label or self.mode}: a plan owns no agents")
        if self.mode is Mode.RALLY and set(self.targets) != set(self.homes):
            raise WaypointError(f"{self.label or self.mode}: a rally needs a target for every owned agent")
        if self.mode is Mode.STIMULUS and not self.need:
            raise WaypointError(f"{self.label or self.mode}: a stimulus needs a need to raise")
        for name in self.resume:
            if name not in self.homes:
                raise WaypointError(f"{self.label or self.mode}: resume point for {name!r}, which the plan does not own")

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "mode": str(self.mode),
            "agents": sorted(self.homes),
            "at": self.at,
            "period": self.period,
            "radius": self.radius,
            "once": self.once,
            "duration": self.duration,
            "targets": {k: list(v) for k, v in sorted(self.targets.items())},
            **({"need": self.need, "intensity": self.intensity} if self.mode is Mode.STIMULUS else {}),
        }

    def shuffle_routes(self, step: int) -> dict[str, list[Point]]:
        """Deterministic in (agent index, issue count): a rerun shuffles identically."""
        out: dict[str, list[Point]] = {}
        for i, (name, (hx, hy)) in enumerate(sorted(self.homes.items())):
            angle = (step + i) * _GOLDEN_ANGLE
            out[name] = [(hx + self.radius * math.cos(angle), hy + self.radius * math.sin(angle))]
        return out

    def hold_routes(self) -> dict[str, list[Point]]:
        return {name: [self.points.get(name, home)] for name, home in self.homes.items()}

    def resume_routes(self) -> dict[str, list[Point]]:
        return {name: [pt] for name, pt in self.resume.items()}


@attrs.define
class WaypointOutcome:
    label: str = ""
    mode: str = ""
    agents: int = 0
    issued: int = 0
    trigger: str = ""
    # None with a non-empty trigger: the robot never came near enough and the effect did not run
    t_trigger: float | None = None
    t_first: float | None = None
    t_release: float | None = None
    failures: int = 0
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return attrs.asdict(self)
