"""A preferred receiving pose from an example object, measured body and handedness."""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from arena_runtime.humans.gestures import Channel
from arena_runtime.humans.support import PALM_CENTER_FRACTION, BodyMeasurements


@dataclass(frozen=True)
class HandoverPose:
    """Handover point in the pedestrian's floor frame and the receiving hand selection."""

    at: tuple[float, float, float]
    side: str
    orientation: str

    def channel(self, pose: tuple[float, float, float] = (0.0, 0.0, 0.0), *, grip: bool = False) -> Channel:
        """A receive channel with the preferred point transformed into the world frame."""
        x, y, yaw = pose
        ahead, left, height = self.at
        c, s = math.cos(yaw), math.sin(yaw)
        return Channel(f"receive_{self.side}", (x + c * ahead - s * left, y + s * ahead + c * left, height), grip=grip, orientation=self.orientation)


@functools.cache
def _objects() -> dict:
    return yaml.safe_load(Path(__file__).with_name("objects.yaml").read_text(encoding="utf-8"))


def handover_profile(object_kind: str, body: BodyMeasurements, handedness: Literal['l', 'r']) -> HandoverPose:
    """An elbow-height receiving pose; example object entries explicitly choose one or two hands."""
    if handedness is None or handedness not in ("l", "r"):
        raise ValueError("handover handedness must be l or r")
    if object_kind not in _objects():
        raise ValueError(f"unknown handover object {object_kind!r}")
    obj = _objects()[object_kind]
    size_x, _, size_z = obj["size_m"]
    side = "both" if obj["hands"] == 2 else handedness
    orientation = "thumb_up" if obj["grip_axis"] == "z" else "palm_up"
    legs, proportions = body.legs, body.proportions
    pelvis = legs.hip + legs.thigh + legs.shin + legs.ankle
    elbow = pelvis + proportions.shoulders - proportions.upperarm
    palm = PALM_CENTER_FRACTION * proportions.hand
    height = elbow + (palm if orientation == "thumb_up" else size_z / 2)
    distance = proportions.forearm + palm + size_x / 2
    lateral = 0.0 if side == "both" else proportions.shoulder_width * (1 if side == "l" else -1)
    return HandoverPose((distance, lateral, height), side, orientation)
