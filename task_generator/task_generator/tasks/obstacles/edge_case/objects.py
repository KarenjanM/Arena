"""Scripted object timelines: objects that spawn and despawn mid-episode, placed by route fraction or absolute pose."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

import attrs

from .pathing import point_at_arc, route_length

Point = tuple[float, float]

SPAWN = "spawn"
DESPAWN = "despawn"
_ACTIONS = (SPAWN, DESPAWN)

_EVENT_KEYS = frozenset({"action", "entity", "model", "at", "place", "note"})
_AT_KEYS = frozenset({"t", "route_fraction"})
_PLACE_KEYS = frozenset({"route_fraction", "lateral_offset", "pose", "yaw"})


class TimelineError(Exception):
    pass


@attrs.frozen
class ObjectEvent:
    action: str
    entity: str
    model: str = ""
    at_t: float | None = None
    at_fraction: float | None = None
    place_fraction: float | None = None
    lateral_offset: float = 0.0
    pose: Point | None = None
    yaw: float | None = None
    note: str = ""


@attrs.frozen
class ResolvedEvent:
    event: ObjectEvent
    t: float
    pose: Point | None = None
    yaw: float = 0.0
    robot_arc: float = 0.0
    reveal_distance_m: float | None = None

    @property
    def action(self) -> str:
        return self.event.action

    @property
    def entity(self) -> str:
        return self.event.entity


def _as_float(value: object, where: str) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise TimelineError(f"{where}: expected a number, got {value!r}") from None


def _parse_at(raw: object, where: str) -> tuple[float | None, float | None]:
    if not isinstance(raw, Mapping):
        raise TimelineError(f"{where}.at: expected a mapping with `t` or `route_fraction`")
    unknown = set(raw) - _AT_KEYS
    if unknown:
        raise TimelineError(f"{where}.at: unknown key(s) {sorted(unknown)}; allowed: {sorted(_AT_KEYS)}")
    if not raw:
        raise TimelineError(f"{where}.at: give either `t` (sim seconds) or `route_fraction`")
    if len(raw) > 1:
        raise TimelineError(f"{where}.at: give `t` or `route_fraction`, not both - they would disagree")

    if "t" in raw:
        t = _as_float(raw["t"], f"{where}.at.t")
        if t < 0:
            raise TimelineError(f"{where}.at.t: must not be negative, got {t}")
        return t, None

    fraction = _as_float(raw["route_fraction"], f"{where}.at.route_fraction")
    if not 0.0 <= fraction <= 1.0:
        raise TimelineError(f"{where}.at.route_fraction: must be within [0, 1], got {fraction}")
    return None, fraction


def _parse_place(raw: object, where: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TimelineError(f"{where}.place: expected a mapping")
    unknown = set(raw) - _PLACE_KEYS
    if unknown:
        raise TimelineError(f"{where}.place: unknown key(s) {sorted(unknown)}; allowed: {sorted(_PLACE_KEYS)}")

    out: dict[str, Any] = {}
    if "pose" in raw:
        pose = raw["pose"]
        if not isinstance(pose, Sequence) or isinstance(pose, (str, bytes)) or len(pose) < 2:
            raise TimelineError(f"{where}.place.pose: expected [x, y] or [x, y, yaw]")
        out["pose"] = (_as_float(pose[0], f"{where}.place.pose[0]"), _as_float(pose[1], f"{where}.place.pose[1]"))
        if len(pose) > 2:
            out["yaw"] = _as_float(pose[2], f"{where}.place.pose[2]")
    if "route_fraction" in raw:
        if "pose" in out:
            raise TimelineError(f"{where}.place: give `pose` or `route_fraction`, not both")
        fraction = _as_float(raw["route_fraction"], f"{where}.place.route_fraction")
        if not 0.0 <= fraction <= 1.0:
            raise TimelineError(f"{where}.place.route_fraction: must be within [0, 1], got {fraction}")
        out["place_fraction"] = fraction
    if "lateral_offset" in raw:
        out["lateral_offset"] = _as_float(raw["lateral_offset"], f"{where}.place.lateral_offset")
    if "yaw" in raw:
        out["yaw"] = _as_float(raw["yaw"], f"{where}.place.yaw")
    return out


def parse_event(raw: object, index: int) -> ObjectEvent:
    where = f"events[{index}]"
    if not isinstance(raw, Mapping):
        raise TimelineError(f"{where}: expected a mapping, got {type(raw).__name__}")

    unknown = set(raw) - _EVENT_KEYS
    if unknown:
        raise TimelineError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(_EVENT_KEYS)}")

    action = str(raw.get("action", SPAWN)).strip().lower()
    if action not in _ACTIONS:
        raise TimelineError(f"{where}.action: expected one of {list(_ACTIONS)}, got {action!r}")

    entity = str(raw.get("entity", "") or "").strip()
    if not entity:
        raise TimelineError(f"{where}.entity: every event needs a name, so a despawn can refer to its spawn")

    model = str(raw.get("model", "") or "").strip()
    if action == SPAWN and not model:
        raise TimelineError(f"{where}.model: a spawn needs a model")
    if action == DESPAWN and model:
        raise TimelineError(f"{where}.model: a despawn names an entity, not a model")

    at_t, at_fraction = _parse_at(raw.get("at"), where)
    place = _parse_place(raw.get("place"), where)
    if action == DESPAWN and place:
        raise TimelineError(f"{where}.place: a despawn has no placement - it removes what the spawn placed")

    return ObjectEvent(
        action=action,
        entity=entity,
        model=model,
        at_t=at_t,
        at_fraction=at_fraction,
        note=str(raw.get("note", "") or ""),
        **place,
    )


def parse_timeline(document: object, *, source: str = "<timeline>") -> list[ObjectEvent]:
    if document is None:
        raise TimelineError(f"{source}: timeline is empty")
    if isinstance(document, Sequence) and not isinstance(document, (str, bytes)):
        document = {"events": document}
    if not isinstance(document, Mapping):
        raise TimelineError(f"{source}: timeline must be a mapping with `events:`, or a bare list of events")

    unknown = set(document) - {"events"}
    if unknown:
        raise TimelineError(f"{source}: unknown key(s) {sorted(unknown)}; a timeline holds only `events`")

    raw_events = document.get("events")
    if not isinstance(raw_events, Sequence) or isinstance(raw_events, (str, bytes)):
        raise TimelineError(f"{source}: `events` must be a list")
    if not raw_events:
        raise TimelineError(f"{source}: timeline has no events - omit the file rather than shipping an empty one")

    events = [parse_event(raw, i) for i, raw in enumerate(raw_events)]

    spawned: set[str] = set()
    for i, event in enumerate(events):
        if event.action == SPAWN:
            if event.entity in spawned:
                raise TimelineError(f"events[{i}]: {event.entity!r} is spawned twice; use a distinct entity name")
            spawned.add(event.entity)
        elif event.entity not in spawned:
            raise TimelineError(f"events[{i}]: despawns {event.entity!r}, which no earlier event spawns")

    return events


_TIMELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "timelines")


def resolve_timeline_path(name_or_path: str, search: Sequence[str] = ()) -> str:
    if os.path.isfile(name_or_path):
        return os.path.abspath(name_or_path)

    relative = name_or_path.lstrip("./") if name_or_path.startswith("./") else name_or_path
    for directory in search:
        candidate = os.path.join(directory, relative)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    base = os.path.basename(name_or_path)
    stem = base if os.path.splitext(base)[1] else f"{base}.yaml"

    share = _installed_timeline(stem)
    if share is not None and os.path.isfile(share):
        return share

    local = os.path.join(_TIMELINE_DIR, stem)
    if os.path.isfile(local):
        return local

    known = ", ".join(available_timelines()) or "(none installed)"
    raise TimelineError(f"object timeline {name_or_path!r} not found; shipped timelines: {known}")


def available_timelines() -> list[str]:
    if not os.path.isdir(_TIMELINE_DIR):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(_TIMELINE_DIR) if f.endswith((".yaml", ".yml")))


def _installed_timeline(stem: str) -> str | None:
    try:
        from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
    except ImportError:  # ament is absent outside a sourced workspace
        return None
    try:
        return os.path.join(get_package_share_directory("task_generator"), "edge_case", "timelines", stem)
    except PackageNotFoundError:
        return None


def load_timeline(name_or_path: str, search: Sequence[str] = ()) -> list[ObjectEvent]:
    import yaml

    path = resolve_timeline_path(name_or_path, search)
    try:
        with open(path) as fh:
            document = yaml.safe_load(fh)
    except OSError as e:
        raise TimelineError(f"cannot read object timeline {path}: {e}") from e
    except yaml.YAMLError as e:
        raise TimelineError(f"object timeline {path} is not valid YAML: {e}") from e
    return parse_timeline(document, source=path)


def resolve(
    events: Sequence[ObjectEvent],
    route: Sequence[Point],
    *,
    robot_speed: float,
    delay: float = 0.0,
) -> list[ResolvedEvent]:
    """`delay` shifts fire times after fractions become times, so an event keeps its place on the route.
    `robot_speed` is nominal, not measured, which is why `reveal_distance_m` is recorded per event."""
    if len(route) < 2:
        raise TimelineError("resolving a route-relative timeline needs a route of at least two points")
    speed = max(float(robot_speed), 1e-6)
    total = route_length(route)
    shift = max(0.0, float(delay))

    resolved: list[ResolvedEvent] = []
    poses: dict[str, Point] = {}

    for event in events:
        if event.at_t is not None:
            t = event.at_t
        else:
            t = (event.at_fraction or 0.0) * total / speed

        robot_arc = min(total, max(0.0, t * speed))
        t += shift

        pose: Point | None = None
        yaw = event.yaw if event.yaw is not None else 0.0
        if event.action == SPAWN:
            if event.pose is not None:
                pose = event.pose
            else:
                fraction = event.place_fraction
                if fraction is None:
                    fraction = event.at_fraction if event.at_fraction is not None else robot_arc / total if total else 0.0
                point, heading = point_at_arc(route, fraction * total)
                pose = _offset(point, heading, event.lateral_offset)
                if event.yaw is None:
                    yaw = math.atan2(-heading[1], -heading[0])
            poses[event.entity] = pose
        else:
            pose = poses.get(event.entity)

        reveal = None
        if event.action == SPAWN and pose is not None:
            robot_point, _ = point_at_arc(route, robot_arc)
            reveal = math.dist(robot_point, pose)

        resolved.append(
            ResolvedEvent(
                event=event,
                t=t,
                pose=pose if event.action == SPAWN else None,
                yaw=yaw,
                robot_arc=robot_arc,
                reveal_distance_m=reveal,
            )
        )

    # stable: a despawn authored after its spawn at the same t stays after it
    resolved.sort(key=lambda r: r.t)
    return resolved


def _offset(point: Point, heading: Point, lateral: float) -> Point:
    if not lateral:
        return point
    return (point[0] - heading[1] * lateral, point[1] + heading[0] * lateral)
