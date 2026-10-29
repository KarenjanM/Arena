"""Driven joints: wire joints a bundle's rig.yaml declares, turned by the ped's motion instead of its gait phase."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import attrs
import yaml

RIG_FILE = "rig.yaml"


@attrs.frozen
class Roller:
    """A wheel `lateral` meters left of the root that rolls without slipping on `radius`."""

    joint: str
    radius: float
    lateral: float


def parse_drivers(rig: object, source: str) -> tuple[Roller, ...]:
    """Distance drivers in file order; velocity drivers are computed by the caller."""
    if not isinstance(rig, dict):
        raise ValueError(f"{source}: a rig file is a mapping, got {type(rig).__name__}")
    section = rig.get("drivers", {})
    if not isinstance(section, dict):
        raise ValueError(f"{source}: drivers takes {{<wire joint>: {{from: distance, radius: <m>, lateral: <m>}}}}, got {section!r}")
    rollers: list[Roller] = []
    for joint, spec in section.items():
        if isinstance(spec, dict) and spec.get("from") == "velocity":
            continue
        if not isinstance(spec, dict) or spec.get("from") != "distance":
            raise ValueError(f"{source}: driver {joint!r} takes {{from: distance, radius: <m>, lateral: <m>}}, got {spec!r}")
        radius = float(spec.get("radius", 0.0))
        if radius <= 0.0:
            raise ValueError(f"{source}: driver {joint!r} needs a radius above 0 m, got {spec.get('radius')!r}")
        rollers.append(Roller(joint=str(joint), radius=radius, lateral=float(spec.get("lateral", 0.0))))
    return tuple(rollers)


def load_drivers(model_uri: str) -> tuple[Roller, ...]:
    """Drivers of the rig.yaml beside an actor SDF, none without the file."""
    if not model_uri:
        return ()
    rig_path = Path(model_uri).with_name(RIG_FILE)
    if not rig_path.is_file():
        return ()
    return parse_drivers(yaml.safe_load(rig_path.read_text()), str(rig_path))


def has_joystick(model_uri: str) -> bool:
    """Whether the model declares both joystick axes as velocity drivers."""
    if not model_uri:
        return False
    rig_path = Path(model_uri).with_name(RIG_FILE)
    if not rig_path.is_file():
        return False
    rig = yaml.safe_load(rig_path.read_text())
    if not isinstance(rig, dict) or not isinstance(rig.get("drivers", {}), dict):
        raise ValueError(f"{rig_path}: rig and drivers must be mappings")
    drivers = rig.get("drivers", {})
    return all(isinstance(drivers.get(joint), dict) and drivers[joint].get("from") == "velocity" for joint in ("joystick_pitch", "joystick_roll"))


def with_driven(names: Sequence[str], positions: Sequence[float], angles: Mapping[str, float]) -> tuple[list[str], list[float]]:
    """Joint names and positions with the driven angles set, same-named entries replaced."""
    merged = dict(zip(names, positions, strict=True))
    merged.update(angles)
    return list(merged), list(merged.values())


class DrivenJoints:
    """Per-ped driver angles in radians, integrated from successive poses and never wrapped."""

    def __init__(self) -> None:
        self._pose: dict[int, tuple[float, float, float]] = {}
        self._angles: dict[int, dict[str, float]] = {}

    def advance(self, agent_id: int, drivers: Sequence[Roller], x: float, y: float, yaw: float) -> dict[str, float]:
        """Roll every driver over the step from the ped's last pose, return joint name to angle."""
        previous = self._pose.get(agent_id)
        self._pose[agent_id] = (x, y, yaw)
        forward = turn = 0.0
        if previous is not None:
            turn = math.remainder(yaw - previous[2], math.tau)
            heading = previous[2] + turn / 2.0
            forward = (x - previous[0]) * math.cos(heading) + (y - previous[1]) * math.sin(heading)
        angles = self._angles.setdefault(agent_id, {})
        for driver in drivers:
            angles[driver.joint] = angles.get(driver.joint, 0.0) + (forward - turn * driver.lateral) / driver.radius
        return {driver.joint: angles[driver.joint] for driver in drivers}

    def known(self) -> set[int]:
        """Ids with integrated state."""
        return set(self._pose)

    def forget(self, agent_id: int) -> None:
        """Drop a despawned ped's pose and angles."""
        self._pose.pop(agent_id, None)
        self._angles.pop(agent_id, None)


class JoystickDriver:
    """Electric-chair joystick and matching right-arm poses from speed and sampled yaw rate."""

    # Absolute radians for arenian_wheelchair_electric + seated_wheelchair_electric.
    # Palm follows joystick roll; four-view mesh checks leave 5–7 mm clearance.
    _HAND_JOINTS = ("r_y_shoulder", "r_p_shoulder", "r_r_shoulder", "r_elbow", "r_r_wrist", "r_wrist")
    _HAND_POSES = {
        # (joystick pitch, roll): shoulder yaw/pitch/roll, elbow, forearm twist, wrist
        (0.0, 0.0): (1.19558, -0.60731, 1.00868, 1.50426, -0.60197, 0.10176),  # neutral
        (0.6, 0.0): (1.47985, -0.4042, 1.40343, 1.2382, -0.43725, -0.02839),  # forward
        (-0.6, 0.0): (0.78838, -0.64308, 0.76448, 1.65762, -0.48019, -0.00116),  # reverse
        (0.6, -0.4): (1.57441, -0.58008, 1.07303, 1.32812, -0.33197, 0.29005),  # forward left
        (0.6, 0.4): (1.47733, -0.43338, 1.52064, 1.11716, -0.81707, -0.16962),  # forward right
        (0.0, -0.6): (1.70538, -0.91595, 0.68625, 1.57246, -0.53479, 0.70789),  # pivot left
        (0.0, 0.6): (1.1626, -0.58367, 1.17998, 1.27694, -1.14152, -0.1606),  # pivot right
    }

    def __init__(self) -> None:
        self._previous: dict[int, tuple[float, float, float]] = {}

    def advance(self, agent_id: int, yaw: float, speed: float, pose_sec: float) -> dict[str, float]:
        """Sample yaw over 0.2 s of pose time, holding the last rate between samples."""
        previous = self._previous.get(agent_id)
        if previous is None or pose_sec < previous[1]:
            angular_speed = 0.0
            self._previous[agent_id] = (yaw, pose_sec, angular_speed)
        else:
            previous_yaw, previous_time, angular_speed = previous
            dt = pose_sec - previous_time
            if dt >= 0.2 - 1e-9:
                delta_yaw = math.remainder(yaw - previous_yaw, math.tau)
                angular_speed = delta_yaw / dt
                self._previous[agent_id] = (yaw, pose_sec, angular_speed)

        pitch = roll = 0.0
        if speed > 0.02:
            pitch = 0.6
            if angular_speed > 0.1:
                roll = -0.4
            elif angular_speed < -0.1:
                roll = 0.4
        elif speed < -0.02:
            pitch = -0.6
        else:
            if angular_speed > 0.02:
                roll = -0.6
            elif angular_speed < -0.02:
                roll = 0.6
        hand = dict(zip(self._HAND_JOINTS, self._HAND_POSES[pitch, roll], strict=True))
        return {"joystick_pitch": pitch, "joystick_roll": roll, **hand}

    def known(self) -> set[int]:
        """Ids with sampled joystick state."""
        return set(self._previous)

    def forget(self, agent_id: int) -> None:
        """Drop a despawned ped's sample."""
        self._previous.pop(agent_id, None)

    def clear(self) -> None:
        """Discard all samples when the scenario resets."""
        self._previous.clear()
