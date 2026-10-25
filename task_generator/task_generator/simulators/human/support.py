"""Body frame pose of a ped sitting on a seat or lying on its back on a bed.

The body frame has its origin at the soles under the pelvis and its z axis up the body, renderers plant the lowest sole
point at body height 0.
"""

from __future__ import annotations

import math
import typing
from collections.abc import Mapping
from pathlib import Path

import attrs
import numpy as np
import yaml

from task_generator.simulators.human.pointing import skeleton as S
from task_generator.simulators.human.profile import JOINT_NAMES

ANNOTATION_FILE = "annotation.yaml"


@attrs.frozen
class Support:
    """Furniture a ped rests on, and its surface height above the ped's floor in m."""

    kind: typing.Literal["sit", "lie"]
    height: float


@attrs.frozen
class PelvisClearance:
    """Height of a character's pelvis joint above the surface it rests on in m, seated and lying on its back."""

    sit: float = attrs.field(converter=float)
    lie: float = attrs.field(converter=float)


@attrs.frozen
class Legs:
    """Leg geometry of a standing character in m: pelvis joint above the hip joints, thigh, shin, ankle joint above the soles."""

    hip: float = attrs.field(converter=float)
    thigh: float = attrs.field(converter=float)
    shin: float = attrs.field(converter=float)
    ankle: float = attrs.field(converter=float)


@attrs.frozen
class BodyMeasurements:
    """The asset annotation's measurements of a character."""

    clearance: PelvisClearance
    legs: Legs


TYPICAL_BODY = BodyMeasurements(clearance=PelvisClearance(sit=0.122, lie=0.117), legs=Legs(hip=0.015, thigh=0.36, shin=0.375, ankle=0.08))


def load_body(model_uri: str) -> BodyMeasurements | None:
    """Measurements in the annotation beside an actor SDF, None without the file or the measurements."""
    if not model_uri:
        return None
    path = Path(model_uri).with_name(ANNOTATION_FILE)
    if not path.is_file():
        return None
    annotation = yaml.safe_load(path.read_text()) or {}
    clearance, legs = annotation.get("pelvis_clearance"), annotation.get("legs")
    if not clearance or not legs:
        return None
    return BodyMeasurements(clearance=PelvisClearance(**clearance), legs=Legs(**legs))


def planted_pelvis(angles: Mapping[str, float], legs: Legs) -> float:
    """Height of the pelvis joint above the soles in the pose the angles give."""
    body = S.Body()
    body.thigh, body.tibia, body.hip_height = legs.thigh, legs.shin, legs.thigh + legs.shin
    pos, _ = S.fk(dict.fromkeys(JOINT_NAMES, 0.0) | dict(angles), body)
    soles = min(pos["l_ankle"][2], pos["r_ankle"][2]) - legs.ankle
    return float(legs.hip + pos["pelvis"][2] - soles)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of xyzw quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


_SUPINE = np.array([0.0, -math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)])


def supported_pose(support: Support, body: BodyMeasurements, angles: Mapping[str, float], x: float, y: float, yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """Body frame position and xyzw orientation of a ped parked at (x, y, yaw) on its support.

    A sitter faces yaw. A lying ped has its pelvis over (x, y), its head pointing away from yaw and its feet toward it.
    """
    pelvis = planted_pelvis(angles, body.legs)
    upright = np.array([0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)])
    if support.kind == "lie":
        position = np.array([x + pelvis * math.cos(yaw), y + pelvis * math.sin(yaw), support.height + body.clearance.lie])
        return position, _quat_mul(upright, _SUPINE)
    return np.array([x, y, max(0.0, support.height + body.clearance.sit - pelvis)]), upright
