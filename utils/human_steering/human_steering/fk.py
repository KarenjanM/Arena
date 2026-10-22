"""FK preview: xacro-render the human rig once, then project a composed pose.

Pipeline: one-time xacro of human_description/urdf/human-tpl.xacro (subprocess)
-> minimal xml.etree parse of joints/origins/axes -> wire-to-URDF conversion via
rviz_utils.hri.rig -> forward kinematics -> front/side orthographic projection.

The FK math itself (forward_kinematics/project_front/project_side) is plain
Python (no numpy, no ROS) so it stays testable on a synthetic joint tree even
where xacro/human_description/rviz_utils are not installed. Only `preview()`
needs the real pipeline.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import os
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence

try:
    from rviz_utils.hri.rig import semantic_to_rig
except ImportError:  # pragma: no cover - exercised only without a sourced ROS install
    semantic_to_rig = None

PREVIEW_ID = "fk_preview"
PREVIEW_HEIGHT_M = 1.65

_ZERO3 = (0.0, 0.0, 0.0)
_IDENTITY_R = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


@dataclasses.dataclass(frozen=True)
class Joint:
    """One URDF joint: static origin (xyz/rpy) plus rotation axis for revolute/continuous joints."""

    name: str
    parent: str
    child: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    kind: str


def _mat3_mul(a: tuple, b: tuple) -> tuple:
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def _mat3_vec(a: tuple, v: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = (sum(a[i][k] * v[k] for k in range(3)) for i in range(3))
    return (x, y, z)


def _vec3_add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _rotation_rpy(rpy: tuple[float, float, float]) -> tuple:
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _rotation_axis_angle(axis: tuple[float, float, float], angle: float) -> tuple:
    """Rodrigues' rotation formula about an arbitrary axis."""
    x, y, z = axis
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < 1e-9:
        return _IDENTITY_R
    x, y, z = x / norm, y / norm, z / norm
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return (
        (t * x * x + c, t * x * y - s * z, t * x * z + s * y),
        (t * x * y + s * z, t * y * y + c, t * y * z - s * x),
        (t * x * z - s * y, t * y * z + s * x, t * z * z + c),
    )


def _floats3(text: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not text:
        return default
    parts = [float(v) for v in text.split()]
    if len(parts) != 3:
        return default
    return (parts[0], parts[1], parts[2])


def parse_urdf(urdf_xml: str) -> tuple[str, dict[str, Joint]]:
    """Parse every <joint> (name/parent/child/origin/axis/type) out of a URDF string.

    Returns (root_link_name, joints_by_name). The root link is whichever link
    is never named as a joint's child.
    """
    root_elem = ET.fromstring(urdf_xml)  # noqa: S314
    joints: dict[str, Joint] = {}
    children: set[str] = set()
    all_links = {link.get("name", "") for link in root_elem.findall("link")}

    for joint_elem in root_elem.findall("joint"):
        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")
        if parent_elem is None or child_elem is None:
            continue
        name = joint_elem.get("name", "")
        origin_elem = joint_elem.find("origin")
        axis_elem = joint_elem.find("axis")
        joint = Joint(
            name=name,
            parent=parent_elem.get("link", ""),
            child=child_elem.get("link", ""),
            xyz=_floats3(origin_elem.get("xyz") if origin_elem is not None else None, _ZERO3),
            rpy=_floats3(origin_elem.get("rpy") if origin_elem is not None else None, _ZERO3),
            axis=_floats3(axis_elem.get("xyz") if axis_elem is not None else None, (1.0, 0.0, 0.0)),
            kind=joint_elem.get("type", "fixed"),
        )
        joints[name] = joint
        children.add(joint.child)

    roots = all_links - children
    root = next(iter(roots)) if roots else next(iter(all_links), "")
    return root, joints


def forward_kinematics(
    root: str,
    joints: Mapping[str, Joint],
    angles: Mapping[str, float],
) -> dict[str, tuple[float, float, float]]:
    """World-frame position of every link. angles keys are URDF (already-suffixed)
    joint names, joints absent from angles (fixed joints, unposed DOF) contribute
    only their static origin."""
    by_parent: dict[str, list[Joint]] = {}
    for joint in joints.values():
        by_parent.setdefault(joint.parent, []).append(joint)

    positions: dict[str, tuple[float, float, float]] = {root: _ZERO3}
    frames: dict[str, tuple[tuple, tuple[float, float, float]]] = {root: (_IDENTITY_R, _ZERO3)}
    stack = [root]
    while stack:
        parent = stack.pop()
        parent_r, parent_t = frames[parent]
        for joint in by_parent.get(parent, ()):
            origin_r = _rotation_rpy(joint.rpy)
            origin_t = _vec3_add(_mat3_vec(parent_r, joint.xyz), parent_t)
            combined_r = _mat3_mul(parent_r, origin_r)
            angle = angles.get(joint.name, 0.0)
            if joint.kind in ("revolute", "continuous") and angle != 0.0:
                combined_r = _mat3_mul(combined_r, _rotation_axis_angle(joint.axis, angle))
            frames[joint.child] = (combined_r, origin_t)
            positions[joint.child] = origin_t
            stack.append(joint.child)
    return positions


def project_front(positions: Mapping[str, tuple[float, float, float]]) -> dict[str, tuple[float, float]]:
    """Front view (facing the ped): horizontal = -y, vertical = z."""
    return {name: (-y, z) for name, (_x, y, z) in positions.items()}


def project_side(positions: Mapping[str, tuple[float, float, float]]) -> dict[str, tuple[float, float]]:
    """Side view: horizontal = x (forward), vertical = z (up)."""
    return {name: (x, z) for name, (x, _y, z) in positions.items()}


def _share_dir(package: str) -> str:
    from ament_index_python.packages import get_package_share_directory

    return get_package_share_directory(package)


def render_urdf(body_id: str = PREVIEW_ID, height: float = PREVIEW_HEIGHT_M) -> str:
    """One-time xacro render of human_description/urdf/human-tpl.xacro via subprocess."""
    xacro_path = os.path.join(_share_dir("human_description"), "urdf", "human-tpl.xacro")
    result = subprocess.run(  # noqa: S603
        ["xacro", xacro_path, f"id:={body_id}", f"height:={height}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


@functools.lru_cache(maxsize=4)
def _cached_tree(body_id: str) -> tuple[str, dict[str, Joint]]:
    return parse_urdf(render_urdf(body_id))


Segment = tuple[tuple[float, float], tuple[float, float]]


def _bone_segments(points: Mapping[str, tuple[float, float]], joints: Mapping[str, Joint]) -> list[Segment]:
    return [(points[j.parent], points[j.child]) for j in joints.values() if j.parent in points and j.child in points]


def preview(
    angles: Mapping[str, float],
    body_id: str = PREVIEW_ID,
) -> tuple[list[Segment], list[Segment]]:
    """Full pipeline: composed semantic angles -> rig conversion -> FK -> (front, side).

    angles keys are bare wire joint names (GaitGenerator.JOINT_NAMES), as read
    back from the bus. Returns (front_bones, side_bones): each a list of 2D line
    segments, one per URDF joint, ready to draw.
    """
    if semantic_to_rig is None:
        raise RuntimeError("rviz_utils.hri.rig is unavailable (source the ROS install)")
    root, joints = _cached_tree(body_id)
    names: Sequence[str] = list(angles.keys())
    raw_values = semantic_to_rig(names, [angles[n] for n in names])
    urdf_angles = {f"{bare}_{body_id}": value for bare, value in zip(names, raw_values, strict=True)}
    positions = forward_kinematics(root, joints, urdf_angles)
    front, side = project_front(positions), project_side(positions)
    return _bone_segments(front, joints), _bone_segments(side, joints)
