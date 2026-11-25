"""Translate external handover phases into requests for the gesture layer."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Literal

import attrs

from arena_runtime.humans.gestures import SLOT_OVERLAY, Channel
from arena_runtime.humans.gestures.profiles.handover.handover_profile import handover_profile, is_handover_object
from arena_runtime.humans.support import BodyMeasurements

PHASE_TIMEOUT_S = {"attend": 3.0, "ready": 5.0, "reach": 3.0, "take": 2.0, "retract": 1.5}
HANDOVER_RETARGET_EPS_M = 0.01


@dataclass(frozen=True)
class _Phase:
    name: str
    at: tuple[float, float, float]
    object_kind: str
    deadline: float

    def apply(self, channels: tuple[Channel, ...], pose: tuple[float, float, float], body: BodyMeasurements, handedness: Literal["l", "r"]) -> tuple[Channel, ...]:
        if self.name == "retract":
            return tuple(channel for channel in channels if SLOT_OVERLAY.get(channel.slot) != "arm")
        reserved = {"head"} if self.name == "attend" else {"head", "arm"}
        channels = (*(channel for channel in channels if SLOT_OVERLAY.get(channel.slot) not in reserved), Channel("head", self.at))
        if self.name == "attend":
            return channels

        profile = handover_profile(self.object_kind, body, handedness)
        _, width, height = profile.size
        offset = (max(0.0, width / 2 - 0.035), -height / 2 - 0.015) if profile.side == "both" else (width / 2 + 0.012, 0.0)
        if self.name == "ready":
            ahead, lateral, _ = profile.at
            legs, proportions = body.legs, body.proportions
            pelvis = legs.hip + legs.thigh + legs.shin + legs.ankle
            elbow_height = pelvis + proportions.shoulders - proportions.upperarm
            palm_height = max(pelvis + 0.08, min(elbow_height - 0.07, self.at[2] + offset[1] - 0.08))
            ready = replace(profile, at=(0.6 * ahead, lateral, palm_height - offset[1]))
            channel = ready.channel(pose)
        else:
            grip = profile.grip_weight if self.name == "take" else 0.0
            channel = attrs.evolve(profile.channel(grip=grip), at=self.at)
        return (*channels, attrs.evolve(channel, retarget_eps=HANDOVER_RETARGET_EPS_M, handover=offset))


class HandoverTrigger:
    """Latest phase per pedestrian, with deadlines in simulation seconds."""

    def __init__(self) -> None:
        self._phases: dict[str, _Phase] = {}

    def clear(self) -> None:
        self._phases.clear()

    def retain(self, names: Iterable[str]) -> None:
        if self._phases:
            for name in self._phases.keys() - set(names):
                del self._phases[name]

    def command(self, pedestrian: str, phase: str, at: tuple[float, float, float], object_kind: str, now: float) -> None:
        if not pedestrian:
            raise ValueError("handover requires a pedestrian name")
        if phase not in PHASE_TIMEOUT_S:
            raise ValueError(f"unknown handover phase {phase!r}")
        if phase != "retract" and not all(math.isfinite(value) for value in at):
            raise ValueError("handover target must be finite")
        if phase in ("ready", "reach", "take") and not is_handover_object(object_kind):
            raise ValueError(f"unknown handover object {object_kind!r}")

        current = self._current(pedestrian, now)
        if current is not None and current.name == "retract" and phase in ("reach", "take"):
            return
        deadline = current.deadline if current is not None and current.name == phase else now + PHASE_TIMEOUT_S[phase]
        self._phases[pedestrian] = _Phase(phase, at, object_kind, deadline)

    def active(self, pedestrian: str, now: float) -> _Phase | None:
        current = self._current(pedestrian, now)
        return current if current is not None and now < current.deadline else None

    def _current(self, pedestrian: str, now: float) -> _Phase | None:
        current = self._phases.get(pedestrian)
        if current is not None and current.name != "retract" and now >= current.deadline:
            current = replace(current, name="retract", deadline=now + PHASE_TIMEOUT_S["retract"])
            self._phases[pedestrian] = current
        return current
