from __future__ import annotations

import array
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .errors import ToolError

if TYPE_CHECKING:
    from geometry_msgs.msg import Pose


def jsonable(value: object) -> object:
    """Plain JSON values from message dicts: arrays to lists, NaN and inf to None, bytes to ints."""
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | array.array):
        return [jsonable(item) for item in value]
    if isinstance(value, bytes | bytearray):
        return list(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if _is_numpy(value):
        return jsonable(value.tolist())
    return value


def _is_numpy(value: object) -> bool:
    return type(value).__module__ == "numpy"


def to_dict(msg: object) -> dict[str, Any]:
    """Every field of a ROS message as plain JSON values."""
    from rosidl_runtime_py.convert import message_to_ordereddict

    out = jsonable(message_to_ordereddict(msg))
    assert isinstance(out, dict)
    return out


def fill[MsgT](msg_type: type[MsgT], data: Mapping[str, Any] | None) -> MsgT:
    """A message of msg_type with the given fields set, rejecting unknown ones."""
    from rosidl_runtime_py.set_message import set_message_fields

    msg = msg_type()
    if data:
        try:
            set_message_fields(msg, dict(data))
        except (AttributeError, TypeError, ValueError) as exc:
            fields = ", ".join(msg_type.get_fields_and_field_types())
            raise ToolError(f"cannot build {msg_type.__name__} from {dict(data)!r}: {exc}", hint=f"fields: {fields}") from exc
    return msg


def constants(msg_type: type, prefix: str = "") -> dict[Any, str]:
    """value -> name for the uppercase constants a message declares, optionally only those with prefix."""
    out: dict[Any, str] = {}
    for name in dir(msg_type):
        if name.isupper() and name.startswith(prefix) and not name.startswith("_"):
            value = getattr(msg_type, name)
            if isinstance(value, int | str | float):
                out.setdefault(value, name[len(prefix) :].lower())
    return out


def seconds(stamp: object) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def pose(x: float, y: float, yaw: float = 0.0, z: float = 0.0) -> Pose:
    from geometry_msgs.msg import Pose

    msg = Pose()
    msg.position.x, msg.position.y, msg.position.z = float(x), float(y), float(z)
    msg.orientation.z = math.sin(yaw / 2.0)
    msg.orientation.w = math.cos(yaw / 2.0)
    return msg


def yaw_of(orientation: object) -> float:
    return math.atan2(2.0 * (orientation.w * orientation.z + orientation.x * orientation.y), 1.0 - 2.0 * (orientation.y**2 + orientation.z**2))


def pose_dict(msg: object) -> dict[str, float]:
    """x, y, z and yaw of a geometry_msgs Pose."""
    return {"x": msg.position.x, "y": msg.position.y, "z": msg.position.z, "yaw": yaw_of(msg.orientation)}
