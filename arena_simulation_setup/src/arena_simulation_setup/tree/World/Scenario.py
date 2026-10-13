import functools
import itertools
import os
import typing
import warnings
from collections.abc import Iterable

import attrs
import yaml

from arena_simulation_setup.shared import DynamicObstacle, EpisodeCondition, Light, Obstacle, Pose, Position, Sound, TaskPhase
from arena_simulation_setup.shared.conditions import parse_atom
from arena_simulation_setup.tree import Identifier, PathView
from arena_simulation_setup.utils.cattrs import ArenaConverter, Parseable, converter

ROBOT_KEYS = frozenset({"start", "level", "goal_level", "phases", "conditions"})


@attrs.define
class RobotGoal(Parseable):
    start: Pose = attrs.field(converter=Pose.converter)
    level: str = attrs.field(default="")
    start_zone: str = attrs.field(default="")
    goal_level: str = attrs.field(default="")
    phases: list[TaskPhase] = attrs.field(factory=list)
    conditions: list[EpisodeCondition] = attrs.field(factory=list)

    @classmethod
    def parse(cls, obj: dict) -> "RobotGoal":
        if "start" not in obj:
            raise ValueError("robot requires 'start'")
        unknown = sorted(set(obj) - ROBOT_KEYS)
        if unknown:
            raise ValueError(f"robot has unknown keys {unknown}, expected some of {sorted(ROBOT_KEYS)}")
        return cls(
            level=str(obj.get("level", "")),
            goal_level=str(obj.get("goal_level", "")),
            start=Pose.parse(obj["start"]),
            start_zone=obj["start"] if isinstance(obj["start"], str) else "",
            phases=[TaskPhase.parse(p) for p in obj.get("phases") or []],
            conditions=[EpisodeCondition.parse(c) for c in obj.get("conditions", [])],
        )

    def phase_list(self) -> list[TaskPhase]:
        return list(self.phases)


@attrs.define
class RegionAssignment:
    type: str = ""  # "source" | "sink"
    polygon: list[Position] = attrs.field(factory=list)  # zone corners (resolved from ref by zone_converter)
    config: dict = attrs.Factory(dict)  # type-specific params


@attrs.define
class TimelineEntry(Parseable):
    """One scenario timeline entry: exactly one trigger (at/every/when) plus a set action list."""

    set: list[dict] = attrs.field(factory=list)
    at: float | None = None
    every: float | None = None
    offset: float = 0.0
    until: float | None = None
    when: dict | str | None = None

    def __attrs_post_init__(self) -> None:
        triggers = [name for name, value in (("at", self.at), ("every", self.every), ("when", self.when)) if value is not None]
        if len(triggers) != 1:
            raise ValueError(f"timeline entry must set exactly one of 'at', 'every', 'when'; got {triggers}")
        if isinstance(self.when, str):
            parse_atom(self.when)


SCENARIO_FORMAT = "arena-scenario/2"
SCENARIO_KEYS = frozenset({"format", "objects", "humans", "sounds", "lights", "robots", "regions", "timeline", "conditions"})


@attrs.define
class Scenario:
    objects: list[Obstacle] = attrs.field(factory=list)
    humans: list[DynamicObstacle] = attrs.field(factory=list)
    sounds: list[Sound] = attrs.field(factory=list)
    lights: list[Light] = attrs.field(factory=list)
    robots: list[RobotGoal] = attrs.field(factory=list)
    regions: dict[str, RegionAssignment] = attrs.field(factory=dict)
    timeline: list[TimelineEntry] = attrs.field(factory=list)
    conditions: list[EpisodeCondition] = attrs.field(factory=list)

    def __attrs_post_init__(self) -> None:
        names: set[str] = set()
        for light in self.lights:
            if light.spec.ambient:
                raise ValueError(f'scenario light {light.name!r} uses fixture {light.fixture!r}, which belongs in the level lights of the world')
            if light.name in names:
                raise ValueError(f'duplicate scenario light {light.name!r}')
            names.add(light.name)


def parse_scenario(raw: object, c: ArenaConverter = converter) -> Scenario:
    """A Scenario from the mapping of a format 2 scenario.yaml."""
    if not isinstance(raw, dict):
        raise ValueError(f"a scenario is a mapping, got {raw!r}")
    if raw.get("format") != SCENARIO_FORMAT:
        raise ValueError(f"format: expected {SCENARIO_FORMAT!r}, got {raw.get('format')!r}")
    unknown = sorted(set(raw) - SCENARIO_KEYS)
    if unknown:
        raise ValueError(f"unknown keys {unknown}, expected some of {sorted(SCENARIO_KEYS)}")
    fields: dict = {}
    for key, kind in (("objects", Obstacle), ("humans", DynamicObstacle), ("sounds", Sound), ("lights", Light), ("timeline", TimelineEntry), ("conditions", EpisodeCondition)):
        items = raw.get(key) or []
        parsed = []
        for index, item in enumerate(items):
            try:
                parsed.append(c.structure(item, kind))
            except Exception as exc:
                raise ValueError(f"{key}[{index}]: {exc}") from exc
        fields[key] = parsed
    robots = []
    for index, item in enumerate(raw.get("robots") or []):
        try:
            robots.append(c.structure(item, RobotGoal))
        except Exception as exc:
            raise ValueError(f"robots[{index}]: {exc}") from exc
    fields["robots"] = robots
    fields["regions"] = c.structure(raw.get("regions") or {}, dict[str, RegionAssignment])
    return Scenario(**fields)


def upgrade_scenario(raw: object) -> tuple[dict, list[str]]:
    """A format 2 scenario mapping and the upgrade warnings for a format 1 scenario mapping."""
    from .legacy import _Upgrade

    if not isinstance(raw, dict):
        raise ValueError(f"a scenario is a mapping, got {raw!r}")
    up = _Upgrade()
    raw = dict(raw)
    raw.pop("format", None)
    wrapper = raw.pop("obstacles", None) or {}
    statics = [*(raw.pop("static", None) or []), *(wrapper.get("static") or []), *(wrapper.get("interactive") or [])]
    dynamics = [*(raw.pop("dynamic", None) or []), *(wrapper.get("dynamic") or [])]
    out: dict = {"format": SCENARIO_FORMAT}
    objects = []
    for index, item in enumerate(statics):
        level = item.get("level_id")
        obj = up.obj(f"static[{index}]", item)
        if level:
            obj["level"] = str(level)
        objects.append(obj)
    if objects:
        out["objects"] = objects
    humans = [up.human(item, keep_level=True) for item in dynamics]
    if humans:
        out["humans"] = humans
    if raw.get("sounds"):
        out["sounds"] = [up.sound(f"sounds[{index}]", item) for index, item in enumerate(raw.pop("sounds"))]
    if raw.get("lights"):
        out["lights"] = [up.light(item) for item in raw.pop("lights")]
    robots = []
    for item in raw.pop("robots", None) or []:
        robot = dict(item)
        if "start_floor" in robot:
            robot["level"] = robot.pop("start_floor")
        if "goal_floor" in robot:
            robot["goal_level"] = robot.pop("goal_floor")
        goal = robot.pop("goal", None)
        if goal is not None and not robot.get("phases"):
            robot["phases"] = [{"goto": goal}]
        for key in ("level", "goal_level"):
            if key in robot and not robot[key]:
                robot.pop(key)
        robots.append(robot)
    if robots:
        out["robots"] = robots
    for key in ("regions", "timeline", "conditions"):
        if raw.get(key):
            out[key] = raw.pop(key)
    for key in ("sounds", "lights", "regions", "timeline", "conditions"):
        raw.pop(key, None)
    if raw:
        up.warn(f"dropped scenario keys {sorted(raw)} that format 2 does not know")
    return out, up.warnings


class ScenarioView(PathView):
    _names: typing.ClassVar[Iterable[str]] = [
        "scenario.yaml",
        "scenario.json",
    ]

    @functools.cached_property
    def scenario_path(self) -> str:
        """
        Get the path to the scenario file.
        """
        prefix = functools.partial(os.path.join, self.path)
        scenario = next((p for p in map(prefix, self._names) if os.path.isfile(p)), prefix(next(iter(self._names))))
        return scenario

    def read(self) -> dict:
        """The format 2 mapping of the scenario on disk, format 1 upgraded."""
        with open(self.scenario_path) as f:
            raw = yaml.safe_load(f)
        if isinstance(raw, dict) and "format" in raw:
            return raw
        upgraded, _ = upgrade_scenario(raw)
        warnings.warn(f"scenario {self.scenario_path} is in format 1, migrate it with touch_world --migrate", DeprecationWarning, stacklevel=2)
        return upgraded

    def migrate(self) -> list[str]:
        """Rewrite a format 1 scenario in place as format 2 `scenario.yaml`. Returns the upgrade warnings, empty for a format 2 one."""
        from .write import to_yaml

        with open(self.scenario_path) as f:
            raw = yaml.safe_load(f)
        if isinstance(raw, dict) and "format" in raw:
            return []
        upgraded, notes = upgrade_scenario(raw)
        target = os.path.join(self.path, "scenario.yaml")
        with open(target, "w") as f:
            f.write(to_yaml(upgraded))
        if self.scenario_path != target:
            os.remove(self.scenario_path)
        return notes

    def identifiers(self, converter: ArenaConverter = converter) -> Iterable[Identifier]:
        """Every asset this scenario references. Pass the world's zone converter to read a
        scenario that addresses zones by name."""
        from arena_simulation_setup.tree.assets.Sound import SoundIdentifier

        scenario = self.load(converter=converter)
        for obstacle in itertools.chain(scenario.objects, scenario.humans):
            yield obstacle.model
        for sound in scenario.sounds:
            yield SoundIdentifier.parse(sound.asset_id)

    def load(self, converter: ArenaConverter = converter) -> Scenario:
        try:
            scenario = parse_scenario(self.read(), converter)
        except Exception as exc:
            raise RuntimeError(f"Scenario {self.scenario_path}: {exc}") from exc
        for obj in itertools.chain(scenario.objects, scenario.humans):
            obj.included_from = self.path
        return scenario
