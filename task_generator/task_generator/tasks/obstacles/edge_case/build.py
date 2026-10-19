"""A case built against one episode: the agents an intercept adds and the route rewrites the effects arm, from the block, the base population and the scene."""

from __future__ import annotations

import math
from collections.abc import Container, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs
import shapely
from arena_simulation_setup.tree.assets.Human import HumanIdentifier

from task_generator.shared import DynamicObstacle, Orientation, Pose, Position
from task_generator.simulators.human.arena_humansim.agent import resolve_agent_type_path
from task_generator.tasks.obstacles._validation import IsValid

from . import ROBOT_SPEED_MPS
from .agent_types import AgentTypeError, agent_block, load
from .effects import ROUTE_TRIGGER_M, SCOPE_ALL, SCOPE_INJECTED, After, Hold, Intercept, Rally, ScopedEffect, Shuffle, Stimulus, TrackRobot, When
from .geometry import GeometryError, encounter_fractions, solve_encounter
from .pathing import point_at_arc, route_length
from .scenario_block import Block
from .triggers import Trigger, centroid
from .waypoints import Mode as WaypointMode
from .waypoints import Plan, Point, WaypointError

if TYPE_CHECKING:
    from arena_humansim.core.agents import AgentType

# released open-loop from far away, walkers crossed 8-28 s before the robot (2026-09 campaign)
LEAD_IN_M = 4.0
STAND_TRIGGER_M = 3.0

INJECT_PREFIX = "edge"
FALLBACK_TYPE = "adult"
FALLBACK_MODEL = "arenian"

ArmedPlan = tuple[Plan, Trigger | None, Point | None]


class CaseAborted(RuntimeError):
    pass


@attrs.define
class AgentTypes:
    """Agent types by declared name, scenario-local profiles found in `scenario_dirs`; loaded once per mode."""

    scenario_dirs: Sequence[str] = ()
    _cache: dict[str, AgentType] = attrs.field(factory=dict)

    @staticmethod
    def declared(obs: DynamicObstacle) -> str | None:
        if isinstance(obs.agent, str):
            return obs.agent
        if isinstance(obs.agent, dict):
            return str(obs.agent.get("agent_type", FALLBACK_TYPE))
        return None

    def of(self, obs: DynamicObstacle) -> AgentType:
        resolved = resolve_agent_type_path(self.declared(obs) or FALLBACK_TYPE, obs.included_from)
        if resolved not in self._cache:
            self._cache[resolved] = load(resolved)
        return self._cache[resolved]

    def profile(self, configured: str) -> str:
        candidate = Path(configured)
        if candidate.is_absolute() or candidate.suffix != ".yaml" or candidate.is_file():
            return configured
        for scenario_dir in self.scenario_dirs:
            found = Path(scenario_dir) / candidate
            if found.is_file():
                return str(found)
        return configured


@attrs.frozen(kw_only=True)
class Scene:
    """The episode a case is built against, in the abstract frame: the robot's planned route (or why there is none) and the world."""

    route: tuple[Point, ...] = ()
    no_route: str = "no robot route"
    zones: Mapping[str, tuple[Point, ...]] = attrs.field(factory=dict)
    is_valid: IsValid | None = None
    # walls only: the forbidden discs round the robot's start and goal would make every route into a room read as blocked
    is_traversable: IsValid | None = None
    robot_speed: float = ROBOT_SPEED_MPS
    robot_hold: float = 0.0
    types: AgentTypes = attrs.field(factory=AgentTypes)

    def robot_route(self) -> list[Point]:
        if not self.route:
            raise CaseAborted(self.no_route)
        return list(self.route)

    def speed(self) -> float:
        if self.robot_speed <= 0.0:
            raise CaseAborted(f"task.edge_case.robot_speed must be positive, got {self.robot_speed}")
        return self.robot_speed

    def point(self, target: str | Point, *, where: str) -> Point:
        if not isinstance(target, str):
            try:
                return (float(target[0]), float(target[1]))
            except (TypeError, ValueError, IndexError):
                raise CaseAborted(f"{where}: {target!r} is neither a zone name nor an [x, y] pair") from None
        if target not in self.zones:
            raise CaseAborted(f"{target!r} is not a zone of this world; known zones: {sorted(self.zones)}")
        polygon = shapely.Polygon(self.zones[target])
        if polygon.is_empty:
            raise CaseAborted(f"zone {target!r} has no usable polygon")
        return (float(polygon.centroid.x), float(polygon.centroid.y))


@attrs.frozen
class Case:
    injected: list[DynamicObstacle]
    plans: list[ArmedPlan]
    geometry: dict[str, Any]
    invalid_spawns: tuple[str, ...]


def build(block: Block, dynamic: Sequence[DynamicObstacle], scene: Scene) -> Case:
    """The intercepts first, so every other effect can scope them; raises `CaseAborted` with the reason a case cannot run."""
    injected: list[DynamicObstacle] = []
    geometry: dict[str, Any] = {}
    plans: list[ArmedPlan] = []
    for index, effect in enumerate(block.effects):
        if isinstance(effect, Intercept):
            obs, geo, extra = inject(effect, index, len(injected), dynamic, scene)
            injected.append(obs)
            plans.extend(extra)
            if not geometry:
                geometry = {**geo, "target_agent": obs.name}
    if not geometry.get("robot_route") and scene.route:
        route = list(scene.route)
        geometry.update({"robot_leg": (route[0], route[-1]), "robot_route": tuple(route), "route_length": route_length(route)})
    combined = [*dynamic, *injected]
    for index, effect in enumerate(block.effects):
        if isinstance(effect, ScopedEffect):
            plans.append(plan_for(effect, index, combined, scene))
    invalid = invalid_spawns(combined, scene.is_valid, fatal={o.name for o in injected})
    return Case(injected=injected, plans=plans, geometry=geometry, invalid_spawns=invalid)


def invalid_spawns(obstacles: Sequence[DynamicObstacle], is_valid: IsValid | None, fatal: Container[str]) -> tuple[str, ...]:
    """Agents placed outside free space; one in `fatal` aborts the case, the rest are recorded."""
    if is_valid is None:
        return ()
    invalid: list[str] = []
    for obs in obstacles:
        pt = Position(x=obs.pose.position.x, y=obs.pose.position.y)
        if not is_valid(pt):
            invalid.append(obs.name)
            if obs.name in fatal:
                raise CaseAborted(f"edge-case agent {obs.name!r} spawn ({pt.x:.2f}, {pt.y:.2f}) is not in free space")
    return tuple(invalid)


def owned(scope: str | tuple[str, ...], dynamic: Sequence[DynamicObstacle], *, where: str) -> dict[str, Point]:
    by_name = {str(a.name): (float(a.pose.position.x), float(a.pose.position.y)) for a in dynamic}
    if scope == SCOPE_ALL:
        out = dict(by_name)
    elif scope == SCOPE_INJECTED:
        out = {n: p for n, p in by_name.items() if n.startswith(f"{INJECT_PREFIX}_")}
    else:
        missing = [n for n in scope if n not in by_name]
        if missing:
            raise CaseAborted(f"{where}: scope names agent(s) this episode did not place: {missing}; placed: {sorted(by_name)}")
        out = {n: by_name[n] for n in scope}
    if not out:
        raise CaseAborted(f"{where}: scope {scope!r} matched none of the {len(dynamic)} agent(s) this episode placed")
    return out


def gate(when: When, scene: Scene, *, default_reference: Point | None) -> tuple[Trigger | None, Point | None, float]:
    if when.route_fraction is not None:
        route = scene.robot_route()
        point, _heading = point_at_arc(route, when.route_fraction * route_length(route))
        return Trigger(robot_within=ROUTE_TRIGGER_M), point, when.at
    if when.robot_within is None and when.robot_beyond is None:
        return None, None, when.at
    trigger = Trigger(robot_within=when.robot_within, robot_beyond=when.robot_beyond, of=when.of)
    reference = scene.point(when.of, where="when.of") if when.of is not None else default_reference
    if reference is None:
        raise CaseAborted("a proximity gate has no reference point: the effect owns no agents and names no `of`")
    return trigger, reference, when.at


def plan_for(effect: ScopedEffect, index: int, dynamic: Sequence[DynamicObstacle], scene: Scene) -> ArmedPlan:
    where = f"effects[{index}] ({effect.kind})"
    label = effect.label or f"{effect.kind}#{index}"
    homes = owned(effect.scope, dynamic, where=where)
    trigger, reference, at = gate(effect.when, scene, default_reference=centroid(list(homes.values())))
    try:
        if isinstance(effect, Rally):
            points = [scene.point(t, where=f"{where}.targets") for t in effect.targets]
            targets = {name: points[i % len(points)] for i, name in enumerate(sorted(homes))}
            plan = Plan(mode=WaypointMode.RALLY, homes=homes, at=at, targets=targets, label=label)
        elif isinstance(effect, Shuffle):
            plan = Plan(mode=WaypointMode.SHUFFLE, homes=homes, at=at, period=effect.period, radius=effect.radius, once=effect.once, label=label)
        elif isinstance(effect, TrackRobot):
            plan = Plan(mode=WaypointMode.TRACK_ROBOT, homes=homes, at=at, period=effect.period, duration=effect.duration, label=label)
        elif isinstance(effect, Hold):
            plan = Plan(mode=WaypointMode.HOLD, homes=homes, at=at, duration=effect.duration, resume=goals_of(homes, dynamic), label=label)
        else:
            check_reacts(effect, homes, dynamic, scene.types, where=where)
            plan = Plan(mode=WaypointMode.STIMULUS, homes=homes, at=at, need=effect.need, intensity=effect.intensity, label=label)
    except WaypointError as exc:
        raise CaseAborted(f"{where}: {exc}") from None
    return plan, trigger, reference


def check_reacts(effect: Stimulus, homes: Mapping[str, Point], dynamic: Sequence[DynamicObstacle], types: AgentTypes, *, where: str) -> None:
    deaf: list[str] = []
    for agent in dynamic:
        if str(agent.name) not in homes:
            continue
        try:
            definition = types.of(agent)
        except AgentTypeError as exc:
            raise CaseAborted(f"{where}: {agent.name}: {exc}") from None
        if definition.mode != "behavior_tree" or effect.need not in definition.needs:
            deaf.append(f"{agent.name} ({types.declared(agent) or FALLBACK_TYPE})")
    if deaf:
        raise CaseAborted(f"{where}: need {effect.need!r} reaches no behaviour tree on {deaf}")


def goals_of(homes: Mapping[str, Point], dynamic: Sequence[DynamicObstacle]) -> dict[str, Point]:
    out: dict[str, Point] = {}
    for agent in dynamic:
        if str(agent.name) in homes and agent.waypoints:
            wp = agent.waypoints[0]
            out[str(agent.name)] = (float(wp.x), float(wp.y))
    return out


def inject_profile(effect: Intercept, dynamic: Sequence[DynamicObstacle], types: AgentTypes) -> tuple[str, AgentType]:
    """The effect's profile, else the alphabetically first type of the population, else `adult`."""
    if effect.profile:
        resolved = types.profile(effect.profile)
        try:
            return resolved, load(resolved)
        except AgentTypeError as exc:
            raise CaseAborted(f"intercept profile {effect.profile!r}: {exc}") from None
    seen: list[tuple[str, AgentType]] = []
    for obs in dynamic:
        declared = types.declared(obs)
        if declared is None:
            continue
        try:
            seen.append((resolve_agent_type_path(declared, obs.included_from), types.of(obs)))
        except AgentTypeError:
            continue
    if seen:
        return min(seen, key=lambda pair: pair[0])
    return FALLBACK_TYPE, load(FALLBACK_TYPE)


def inject_model(effect: Intercept, dynamic: Sequence[DynamicObstacle]) -> HumanIdentifier:
    if effect.model:
        return HumanIdentifier.parse(effect.model)
    models = [o.model for o in dynamic]
    if not models:
        return HumanIdentifier.parse(FALLBACK_MODEL)
    return min(models, key=lambda m: m.name)


def inject(effect: Intercept, index: int, ordinal: int, dynamic: Sequence[DynamicObstacle], scene: Scene) -> tuple[DynamicObstacle, dict[str, Any], list[ArmedPlan]]:
    """One pedestrian placed to meet the robot at `effect.route_fraction`, its geometry record and the plans that release it."""
    where = f"effects[{index}] (intercept)"
    name = f"{INJECT_PREFIX}_{ordinal}"
    profile_ref, definition = inject_profile(effect, dynamic, scene.types)
    speed = effect.speed or definition.desired_velocity.mean
    model = inject_model(effect, dynamic)
    route = scene.robot_route()
    plans: list[ArmedPlan] = []
    geometry: dict[str, Any] = {
        "inject_type": profile_ref,
        "inject_model": model.name,
        "waypoint_mode": effect.waypoint_mode,
        "after": str(effect.after),
        "ped_speed": speed,
        "robot_leg": (route[0], route[-1]),
        "robot_route": tuple(route),
        "route_length": route_length(route),
        "route_planned": len(route) > 2,
        "approach_angle_requested": effect.angle,
        "encounter_at_requested": effect.route_fraction,
        "co_arrival_offset": effect.offset,
    }

    if effect.after is After.STAND:
        total = route_length(route)
        spot: Point | None = None
        achieved = effect.route_fraction
        for fraction in encounter_fractions(effect.route_fraction):
            point, heading = point_at_arc(route, fraction * total)
            if scene.is_valid is None or scene.is_valid(Position(x=point[0], y=point[1])):
                spot, achieved = point, fraction
                break
        if spot is None:
            raise CaseAborted(f"{where}: no free point on the robot's route to stand on")
        yaw = math.atan2(-heading[1], -heading[0])
        t_arrival = achieved * total / scene.speed()
        release = point_at_arc(route, min(total, achieved * total + effect.lead_out))[0]
        obs = DynamicObstacle(
            name=name,
            model=model,
            pose=Pose(Position(x=spot[0], y=spot[1]), orientation=Orientation.from_yaw(yaw)),
            waypoints=[Position(x=spot[0], y=spot[1])],
            velocity=speed,
            agent=agent_block(profile_ref, definition, speed),
            waypoint_mode="once",
        )
        geometry.update(
            {
                "place": "stand",
                "approach_angle_achieved": effect.angle,
                "encounter_at_achieved": achieved,
                "encounter_point": spot,
                "t_encounter": t_arrival,
                "designed_ttc": t_arrival,
                "spawn_pose": spot,
                "ped_goal": spot,
                "geometry_retries": 0,
            }
        )
        if effect.duration > 0:
            plans.append((Plan(mode=WaypointMode.HOLD, homes={name: spot}, at=0.0, duration=effect.duration, resume={name: release}, label=f"{name}:stand"), Trigger(robot_within=STAND_TRIGGER_M), spot))
        return obs, geometry, plans

    try:
        enc = solve_encounter(
            route,
            robot_speed=scene.speed(),
            ped_speed=speed,
            approach_angle=effect.angle,
            encounter_at=effect.route_fraction,
            co_arrival_offset=effect.offset,
            lead_out=effect.lead_out,
            angle_search=effect.search,
            is_valid=scene.is_traversable,
            spawn_valid=scene.is_valid,
            max_lead_in=LEAD_IN_M,
        )
    except GeometryError as exc:
        raise CaseAborted(f"{where}: {exc}") from None

    goal, spot = enc.goal, Position(x=enc.spawn[0], y=enc.spawn[1])
    walk = math.dist(enc.spawn, enc.encounter) / speed
    # released when the robot is due as soon as the walker, so a slow or late robot still meets it
    release = Trigger(robot_eta=round(max(walk - effect.offset, 0.5), 2))
    yaw = math.atan2(enc.encounter[1] - enc.spawn[1], enc.encounter[0] - enc.spawn[0])
    obs = DynamicObstacle(
        name=name,
        model=model,
        pose=Pose(spot, orientation=Orientation.from_yaw(yaw)),
        waypoints=[spot],
        velocity=speed,
        agent=agent_block(profile_ref, definition, speed),
        waypoint_mode="once",
    )
    geometry.update(
        {
            "place": "intercept",
            "approach_angle_achieved": enc.angle_achieved,
            "encounter_at_achieved": enc.encounter_at_achieved,
            "encounter_point": enc.encounter,
            "t_encounter": enc.t_encounter,
            "designed_ttc": enc.designed_ttc,
            "designed_pet": enc.designed_pet,
            "lead_out_achieved": enc.lead_out_achieved,
            "spawn_pose": enc.spawn,
            "ped_goal": goal,
            "geometry_retries": enc.retries,
            "release_eta": release.robot_eta,
            "robot_hold": scene.robot_hold,
        }
    )
    if effect.after is After.VEER:
        assert effect.veer_to is not None
        veer = scene.point(effect.veer_to, where=f"{where}.veer_to")
        plan = Plan(mode=WaypointMode.HOLD, homes={name: enc.spawn}, at=0.0, duration=walk + 0.5, points={name: enc.encounter}, resume={name: veer}, label=f"{name}:veer")
    elif effect.after is After.STOP_FACING:
        plan = Plan(mode=WaypointMode.HOLD, homes={name: enc.spawn}, at=0.0, duration=walk + effect.duration, points={name: enc.encounter}, resume={name: goal}, label=f"{name}:stop_facing")
    elif effect.after is After.FOLLOW:
        plan = Plan(mode=WaypointMode.TRACK_ROBOT, homes={name: enc.spawn}, at=0.0, period=2.0, duration=walk + effect.duration, resume={name: goal}, label=f"{name}:follow")
    else:
        plan = Plan(mode=WaypointMode.RALLY, homes={name: enc.spawn}, at=0.0, targets={name: goal}, label=f"{name}:release")
    plans.append((plan, release, enc.encounter))
    return obs, geometry, plans
