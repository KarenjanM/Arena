from __future__ import annotations

import enum
import math
from collections.abc import Mapping, Sequence
from typing import Any

import attrs

Point = tuple[float, float]


class EffectError(Exception):
    pass


SCOPE_ALL = "all"
SCOPE_INJECTED = "injected"

# a `route_fraction` onset is resolved at reset into `robot_within` this many metres of that route point
ROUTE_TRIGGER_M = 1.5


class Kind(enum.StrEnum):
    INTERCEPT = "intercept"
    RALLY = "rally"
    SHUFFLE = "shuffle"
    SCATTER = "scatter"
    TRACK_ROBOT = "track_robot"
    HOLD = "hold"
    STIMULUS = "stimulus"


class After(enum.StrEnum):
    CONTINUE = "continue"
    STOP_FACING = "stop_facing"
    FOLLOW = "follow"
    VEER = "veer"
    STAND = "stand"


@attrs.frozen
class When:
    at: float = 0.0
    robot_within: float | None = None
    robot_beyond: float | None = None
    of: str | Point | None = None
    route_fraction: float | None = None

    @property
    def gated(self) -> bool:
        return self.robot_within is not None or self.robot_beyond is not None or self.route_fraction is not None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"at": self.at}
        if self.robot_within is not None:
            out["robot_within"] = self.robot_within
        if self.robot_beyond is not None:
            out["robot_beyond"] = self.robot_beyond
        if self.route_fraction is not None:
            out["route_fraction"] = self.route_fraction
        if self.of is not None:
            out["of"] = list(self.of) if isinstance(self.of, (list, tuple)) else self.of
        return out


_WHEN_KEYS = frozenset({"at", "robot_within", "robot_beyond", "of", "route_fraction"})


def _number(spec: Mapping[str, Any], key: str, default: float, *, where: str, floor: float | None = None, positive: bool = False) -> float:
    raw = spec.get(key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise EffectError(f"{where}.{key}: {raw!r} is not a number") from None
    if not math.isfinite(value):
        raise EffectError(f"{where}.{key}: {value!r} is not finite")
    if positive and value <= 0:
        raise EffectError(f"{where}.{key}: must be positive, got {value!r}")
    if floor is not None:
        value = max(floor, value)
    return value


def parse_when(spec: object, *, where: str = "when") -> When:
    if spec is None:
        return When()
    if not isinstance(spec, Mapping):
        raise EffectError(f"{where}: expected a mapping, got {type(spec).__name__}")
    unknown = sorted(set(spec) - _WHEN_KEYS)
    if unknown:
        raise EffectError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(_WHEN_KEYS)}")
    gates = [k for k in ("robot_within", "robot_beyond", "route_fraction") if spec.get(k) is not None]
    if len(gates) > 1:
        raise EffectError(f"{where}: set one of robot_within / robot_beyond / route_fraction, not {gates}")
    out = When(at=_number(spec, "at", 0.0, where=where, floor=0.0))
    if "robot_within" in gates:
        out = attrs.evolve(out, robot_within=_number(spec, "robot_within", 0.0, where=where, positive=True))
    if "robot_beyond" in gates:
        out = attrs.evolve(out, robot_beyond=_number(spec, "robot_beyond", 0.0, where=where, positive=True))
    if "route_fraction" in gates:
        f = _number(spec, "route_fraction", 0.5, where=where)
        if not 0.0 < f < 1.0:
            raise EffectError(f"{where}.route_fraction: must be strictly between 0 and 1, got {f}")
        out = attrs.evolve(out, route_fraction=f)
    if spec.get("of") is not None:
        if not gates or "route_fraction" in gates:
            raise EffectError(f"{where}.of: only applies with robot_within / robot_beyond")
        out = attrs.evolve(out, of=_target(spec["of"], where=f"{where}.of"))
    return out


def _target(raw: object, *, where: str) -> str | Point:
    if isinstance(raw, str):
        name = raw.strip()
        if not name:
            raise EffectError(f"{where}: empty target")
        return name
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        try:
            return (float(raw[0]), float(raw[1]))
        except (TypeError, ValueError):
            pass
    raise EffectError(f"{where}: {raw!r} is neither a zone name nor an [x, y] pair")


Scope = str | tuple[str, ...]


def parse_scope(raw: object, *, where: str) -> Scope:
    if raw is None:
        return SCOPE_ALL
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in (SCOPE_ALL, SCOPE_INJECTED):
            return value
        raise EffectError(f"{where}.scope: {raw!r} is not 'all', 'injected' or a list of agent names")
    if isinstance(raw, (list, tuple)):
        names = tuple(str(n).strip() for n in raw)
        if not names or any(not n for n in names):
            raise EffectError(f"{where}.scope: a name list must be non-empty and hold no blanks")
        return names
    raise EffectError(f"{where}.scope: expected 'all', 'injected' or a list, got {type(raw).__name__}")


@attrs.frozen(kw_only=True)
class Effect:
    kind: Kind
    when: When = When()
    label: str = ""

    def as_dict(self) -> dict[str, Any]:
        out = attrs.asdict(self, recurse=False)
        out["type"] = str(self.kind)
        out.pop("kind")
        out["when"] = self.when.as_dict()
        for key, value in list(out.items()):
            if isinstance(value, enum.Enum):
                out[key] = str(value)
            elif isinstance(value, tuple):
                out[key] = [list(v) if isinstance(v, tuple) else v for v in value]
        return out


@attrs.frozen(kw_only=True)
class Intercept(Effect):
    kind: Kind = Kind.INTERCEPT
    profile: str = ""
    model: str = ""
    # degrees to the robot's heading at the encounter: 180 head-on, 90 crossing, 0 overtaking
    angle: float = 180.0
    route_fraction: float = 0.5
    speed: float = 0.0
    # seconds the pedestrian arrives after the robot (negative: before)
    offset: float = 0.0
    lead_out: float = 3.0
    # max angular deviation when hunting free space; 0 aborts instead
    search: float = 90.0
    after: After = After.CONTINUE
    duration: float = 0.0
    veer_to: str | Point | None = None
    waypoint_mode: str = "reverse"


@attrs.frozen(kw_only=True)
class Rally(Effect):
    kind: Kind = Kind.RALLY
    scope: Scope = SCOPE_ALL
    targets: tuple[str | Point, ...] = ()


@attrs.frozen(kw_only=True)
class Shuffle(Effect):
    kind: Kind = Kind.SHUFFLE
    scope: Scope = SCOPE_ALL
    period: float = 4.0
    radius: float = 3.0
    once: bool = False


@attrs.frozen(kw_only=True)
class TrackRobot(Effect):
    kind: Kind = Kind.TRACK_ROBOT
    scope: Scope = SCOPE_ALL
    period: float = 2.0
    duration: float = 0.0


@attrs.frozen(kw_only=True)
class Hold(Effect):
    kind: Kind = Kind.HOLD
    scope: Scope = SCOPE_ALL
    duration: float = 10.0


@attrs.frozen(kw_only=True)
class Stimulus(Effect):
    kind: Kind = Kind.STIMULUS
    scope: Scope = SCOPE_ALL
    need: str = ""
    intensity: float = 1.0


# every effect but an intercept rewrites agents already placed
ScopedEffect = Rally | Shuffle | TrackRobot | Hold | Stimulus

_COMMON = frozenset({"type", "when", "label"})
_KEYS: dict[Kind, frozenset[str]] = {
    Kind.INTERCEPT: _COMMON | {"profile", "model", "angle", "route_fraction", "speed", "offset", "lead_out", "search", "after", "duration", "veer_to", "waypoint_mode"},
    Kind.RALLY: _COMMON | {"scope", "target", "targets"},
    Kind.SHUFFLE: _COMMON | {"scope", "period", "radius", "once"},
    Kind.SCATTER: _COMMON | {"scope", "radius"},
    Kind.TRACK_ROBOT: _COMMON | {"scope", "period", "duration"},
    Kind.HOLD: _COMMON | {"scope", "duration"},
    Kind.STIMULUS: _COMMON | {"scope", "need", "intensity"},
}

_WAYPOINT_MODES = ("reverse", "once", "repeat", "random")
# below about a second an agent only ever turns around
MIN_PERIOD_S = 1.0


def parse_effect(spec: object, *, index: int = 0) -> Effect:
    where = f"effects[{index}]"
    if not isinstance(spec, Mapping):
        raise EffectError(f"{where}: expected a mapping, got {type(spec).__name__}")
    raw_kind = str(spec.get("type", "")).strip().lower()
    try:
        kind = Kind(raw_kind)
    except ValueError:
        raise EffectError(f"{where}.type: {raw_kind!r} is not one of {[k.value for k in Kind]}") from None

    unknown = sorted(set(spec) - _KEYS[kind])
    if unknown:
        raise EffectError(f"{where} ({kind}): unknown key(s) {unknown}; allowed: {sorted(_KEYS[kind])}")

    when = parse_when(spec.get("when"), where=f"{where}.when")
    label = str(spec.get("label", "") or "")

    if kind is Kind.INTERCEPT:
        raw_after = str(spec.get("after", After.CONTINUE)).strip().lower()
        try:
            after = After(raw_after)
        except ValueError:
            raise EffectError(f"{where}.after: {raw_after!r} is not one of {[a.value for a in After]}") from None
        fraction = _number(spec, "route_fraction", 0.5, where=where)
        if not 0.0 < fraction < 1.0:
            raise EffectError(f"{where}.route_fraction: must be strictly between 0 and 1, got {fraction}")
        mode = str(spec.get("waypoint_mode", "reverse")).strip().lower()
        if mode not in _WAYPOINT_MODES:
            raise EffectError(f"{where}.waypoint_mode: {mode!r} is not one of {list(_WAYPOINT_MODES)}")
        veer_to = spec.get("veer_to")
        if after is After.VEER and veer_to is None:
            raise EffectError(f"{where}: after 'veer' needs a 'veer_to' zone or [x, y]")
        if after is not After.VEER and veer_to is not None:
            raise EffectError(f"{where}.veer_to: only applies to after 'veer'")
        return Intercept(
            when=when,
            label=label,
            profile=str(spec.get("profile", "") or ""),
            model=str(spec.get("model", "") or ""),
            angle=_number(spec, "angle", 180.0, where=where),
            route_fraction=fraction,
            speed=_number(spec, "speed", 0.0, where=where, floor=0.0),
            offset=_number(spec, "offset", 0.0, where=where),
            lead_out=_number(spec, "lead_out", 3.0, where=where, positive=True),
            search=_number(spec, "search", 90.0, where=where, floor=0.0),
            after=after,
            duration=_number(spec, "duration", 0.0, where=where, floor=0.0),
            veer_to=None if veer_to is None else _target(veer_to, where=f"{where}.veer_to"),
            waypoint_mode=mode,
        )

    scope = parse_scope(spec.get("scope"), where=where)

    if kind is Kind.RALLY:
        raw_targets = spec.get("targets")
        if raw_targets is None and spec.get("target") is not None:
            raw_targets = [spec["target"]]
        if not isinstance(raw_targets, (list, tuple)) or not raw_targets:
            raise EffectError(f"{where}: a rally needs a 'target' (zone name or [x, y]) or a 'targets' list")
        targets = tuple(_target(t, where=f"{where}.targets[{i}]") for i, t in enumerate(raw_targets))
        return Rally(when=when, label=label, scope=scope, targets=targets)

    if kind in (Kind.SHUFFLE, Kind.SCATTER):
        return Shuffle(
            when=when,
            label=label,
            scope=scope,
            period=_number(spec, "period", 4.0, where=where, floor=MIN_PERIOD_S),
            radius=_number(spec, "radius", 3.0 if kind is Kind.SHUFFLE else 2.0, where=where, floor=0.5),
            once=kind is Kind.SCATTER or bool(spec.get("once", False)),
        )

    if kind is Kind.TRACK_ROBOT:
        return TrackRobot(
            when=when,
            label=label,
            scope=scope,
            period=_number(spec, "period", 2.0, where=where, floor=MIN_PERIOD_S),
            duration=_number(spec, "duration", 0.0, where=where, floor=0.0),
        )

    if kind is Kind.STIMULUS:
        need = str(spec.get("need", "") or "").strip()
        if not need:
            raise EffectError(f"{where}: a stimulus needs a 'need' - the agent type's need it raises")
        intensity = _number(spec, "intensity", 1.0, where=where)
        if not 0.0 <= intensity <= 1.0:
            raise EffectError(f"{where}.intensity: must be within [0, 1], got {intensity}")
        return Stimulus(when=when, label=label, scope=scope, need=need, intensity=intensity)

    return Hold(when=when, label=label, scope=scope, duration=_number(spec, "duration", 10.0, where=where, positive=True))


def parse_effects(raw: object) -> tuple[Effect, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, Mapping)):
        raise EffectError(f"effects: expected a list, got {type(raw).__name__}")
    return tuple(parse_effect(item, index=i) for i, item in enumerate(raw))
