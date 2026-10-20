from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import attrs
from arena_rclpy_mixins.ROSParamServer import ROSParamT

from task_generator.constants import Constants
from task_generator.shared import Orientation, Pose, Position
from task_generator.tasks.obstacles import Obstacles, TM_Obstacles
from task_generator.tasks.obstacles._validation import in_map_bounds, make_is_traversable, make_is_valid
from task_generator.tasks.robots.request import GoToPhase

from . import ROBOT_SPEED_MPS
from .agent_types import AgentTypeError
from .build import AgentTypes, ArmedPlan, CaseAborted, Scene, build
from .criticality import formation_hold
from .geometry import GeometryError
from .object_driver import DEFAULT_RATE_HZ as OBJECT_RATE_HZ
from .object_driver import ObjectDriver
from .objects import ObjectEvent, ResolvedEvent, TimelineError
from .objects import load_timeline as load_object_timeline
from .objects import resolve as resolve_objects
from .pathing import PathError, plan_route, route_length
from .provenance import CaseLog, CaseRecord
from .scenario_block import Block, BlockError, scenario_path
from .scenario_block import read as read_block
from .scoring import DEFAULT_RATE_HZ, SCOPE_AUTO, EpisodeScorer, score_episode, score_scope, write_score
from .waypoint_driver import DEFAULT_RATE_HZ as WAYPOINT_RATE_HZ
from .waypoint_driver import ROBOT_DEPART_M, WaypointDriver, Work
from .waypoints import Mode as WaypointMode
from .waypoints import Point, WaypointError

if TYPE_CHECKING:
    pass

_ROBOT_TRAVERSALS_PARAM = "task.edge_case_robot.traversals"


@attrs.define()
class EdgeCaseConfig:
    read_scenario_block: ROSParamT[bool]
    prompt: ROSParamT[str]
    prompt_base: ROSParamT[str]
    record_dir: ROSParamT[str]
    score: ROSParamT[bool]
    score_rate_hz: ROSParamT[float]
    score_scope: ROSParamT[str]
    objects: ROSParamT[str]
    objects_rate_hz: ROSParamT[float]
    robot_speed: ROSParamT[float]


class TM_EdgeCase(TM_Obstacles):
    _config: EdgeCaseConfig

    def __init__(self, **kwargs: object) -> None:
        TM_Obstacles.__init__(self, **kwargs)

        self._config = EdgeCaseConfig(
            read_scenario_block=self.node.ROSParam[bool](self.namespace("read_scenario_block"), value=True),
            prompt=self.node.ROSParam[str](self.namespace("prompt"), value=""),
            prompt_base=self.node.ROSParam[str](self.namespace("prompt_base"), value=""),
            record_dir=self.node.ROSParam[str](self.namespace("record_dir"), value=""),
            score=self.node.ROSParam[bool](self.namespace("score"), value=True),
            score_rate_hz=self.node.ROSParam[float](self.namespace("score_rate_hz"), value=DEFAULT_RATE_HZ),
            score_scope=self.node.ROSParam[str](self.namespace("score_scope"), value=SCOPE_AUTO),
            objects=self.node.ROSParam[str](self.namespace("objects"), value=""),
            objects_rate_hz=self.node.ROSParam[float](self.namespace("objects_rate_hz"), value=OBJECT_RATE_HZ),
            robot_speed=self.node.ROSParam[float](self.namespace("robot_speed"), value=ROBOT_SPEED_MPS),
        )

        self._block: Block | None = None
        self._block_error: BlockError | None = None
        self._last_block_path: str = ""
        self._prompt_scenarios: dict[str, str] = {}

        self._scorer: EpisodeScorer | None = None
        self._pending_case: CaseRecord | None = None

        self._objects: ObjectDriver | None = None
        self._timeline_cache: tuple[str, list[ObjectEvent]] | None = None

        self._waypoints: WaypointDriver | None = None
        self._loops: list[asyncio.Task[None]] = []
        self._work: set[asyncio.Task[object]] = set()

        self._base_cache: TM_Obstacles | None = None
        self._types = AgentTypes()

    def _base(self) -> TM_Obstacles:
        from task_generator.tasks.registry import OBSTACLES_MODES  # noqa: PLC0415

        if self._base_cache is not None:
            return self._base_cache
        key = Constants.TaskMode.TM_Obstacles.SCENARIO
        if key not in OBSTACLES_MODES:
            raise CaseAborted("the scenario obstacle mode is not registered")
        cls = OBSTACLES_MODES.get(key)
        meta = OBSTACLES_MODES.meta(key)
        self._base_cache = cls(ctx=self._ctx, namespace=meta.namespace, node=self.node)
        return self._base_cache

    def _scenario_name(self) -> str:
        return str(self.node.rosparam[str].get("task.scenario.file", "") or "").strip()

    def _scenario_dirs(self) -> list[str]:
        if not self._ctx.world_manager.loaded_world:
            return []
        scenarios = self._ctx.world_manager.loaded_world_path / "scenarios"
        scenario = self._scenario_name()
        dirs = [str(scenarios / scenario)] if scenario else []
        dirs.append(str(scenarios))
        return dirs

    def _ensure_scorer(self) -> EpisodeScorer | None:
        if not self._config.score.value:
            self._teardown_scorer()
            return None
        if self._scorer is not None:
            return self._scorer
        robots = self._ctx.robots
        if not robots:
            return None
        _, manager = next(iter(robots.items()))
        scorer = EpisodeScorer(manager, rate_hz=float(self._config.score_rate_hz.value or DEFAULT_RATE_HZ))
        self.node.executor.add_node(scorer)
        self._scorer = scorer
        self._logger.info(f"edge_case: scoring active at {self._config.score_rate_hz.value} Hz -> {self._case_log().path.parent / 'scores.jsonl'}")
        return scorer

    def _teardown_scorer(self) -> None:
        if self._scorer is None:
            return
        self.node.executor.remove_node(self._scorer)
        self._scorer.shutdown()
        self._scorer = None

    def flush_score(self) -> None:
        scorer, case = self._scorer, self._pending_case
        self._pending_case = None
        if scorer is None or case is None:
            return
        try:
            own = {str(n) for step in ("C", "D") for n in ((self._block.owned.get(step) if self._block else None) or ())}
            scope, agents, note = score_scope(self._config.score_scope.value, case, own, scorer.samples)
            if note:
                self._logger.warning(f"edge_case: {note}")
            route = case.robot_route or case.robot_leg or ()
            goal = self._realized(route[-1]) if route else None
            result, ambient, progress = score_episode(scorer.samples, encounter=case.encounter_point, agents=agents, goal=goal, route_length=case.route_length or 0.0)
            if not result.observed:
                self._logger.warning("edge_case: episode produced no usable samples; no score written")
                return
            extra = {**self._object_outcomes(), **self._waypoint_outcomes(), **progress}
            members = list((self._block.owned.get("D") if self._block else None) or ())
            if members and self._block is not None:
                spec = dict(self._block.formation)
                centre = self._realized(spec["centre"]) if spec.get("centre") else None
                extra["formation"] = formation_hold(scorer.samples, members, radius=float(spec.get("radius", 1.5)), centre=centre)
            write_score(self._case_log().path.parent, case, result, scope=scope, scored_agents=agents, ambient=ambient, extra=extra)
            self._logger.info(f"edge_case: scored episode {case.episode_id} scope={scope} min_ttc={result.min_ttc_s} min_clearance={result.min_clearance_m}")
        except Exception as exc:
            self._logger.warning(f"edge_case: scoring this episode failed ({exc!r})")

    def _realized(self, point: Sequence[float]) -> Point:
        placed = self._ctx.environment_manager.realize(Pose(Position(x=float(point[0]), y=float(point[1]))))
        return (float(placed.position.x), float(placed.position.y))

    def _arm_score(self) -> None:
        scorer = self._ensure_scorer()
        if scorer is not None:
            scorer.reset()

    def _write_case(self, record: CaseRecord) -> None:
        self._pending_case = record
        self._case_log().write(record)

    def _case_log(self) -> CaseLog:
        return CaseLog(self._config.record_dir.value or None)

    def _timeline_source(self) -> str:
        return (self._config.objects.value or "").strip() or (self._block.objects if self._block else "")

    def _timeline(self) -> list[ObjectEvent]:
        path = self._timeline_source()
        if not path:
            self._timeline_cache = None
            return []
        if self._timeline_cache is not None and self._timeline_cache[0] == path:
            return self._timeline_cache[1]
        events = load_object_timeline(path, self._scenario_dirs())
        self._timeline_cache = (path, events)
        return events

    def _object_driver(self) -> ObjectDriver:
        if self._objects is None:
            self._objects = ObjectDriver(spawn=self._spawn_object, despawn=self._despawn_object, submit=self._submit, logger=self._logger)
            self._drive(self._objects.tick, float(self._config.objects_rate_hz.value or OBJECT_RATE_HZ))
        return self._objects

    def _waypoint_driver(self) -> WaypointDriver:
        if self._waypoints is None:
            self._waypoints = WaypointDriver(reroute=self._reroute, robot_pose=self._robot_here, submit=self._submit, logger=self._logger, stimulus=self._stimulate_agents, depart_after_m=ROBOT_DEPART_M)
            self._drive(self._waypoints.tick, WAYPOINT_RATE_HZ)
        return self._waypoints

    def _drive(self, tick: Callable[[float], None], rate_hz: float) -> None:
        """Ticks a driver on sim time from the node's event loop, so a paused simulation pauses it."""

        async def loop() -> None:
            try:
                with self.node.sim_time_rate(rate_hz) as (done, rate):
                    while not done.is_set():
                        await rate.get()
                        tick(self.node.sim_time.to_seconds())
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                self._logger.error(f"edge_case: driver loop stopped ({exc!r})")

        self._loops.append(asyncio.get_running_loop().create_task(loop()))

    def _submit(self, work: Work) -> None:
        task = asyncio.get_running_loop().create_task(work)
        self._work.add(task)
        task.add_done_callback(self._work.discard)

    def _teardown_drivers(self) -> None:
        for task in (*self._loops, *self._work):
            task.cancel()
        self._loops.clear()
        self._objects = self._waypoints = None

    async def _spawn_object(self, event: ResolvedEvent) -> str:
        from task_generator.tasks.obstacles import ObstacleKind  # noqa: PLC0415

        assert event.pose is not None
        pose = Pose(Position(float(event.pose[0]), float(event.pose[1])), orientation=Orientation.from_yaw(float(event.yaw)))
        return (await self.extend(ObstacleKind.STATIC, event.event.model, pose)).name

    async def _despawn_object(self, event: ResolvedEvent, entity_id: str) -> bool:
        del event
        return await self.retract(entity_id)

    def _arm_objects(self, scene: Scene) -> list[dict[str, Any]]:
        try:
            events = self._timeline()
        except TimelineError as exc:
            raise CaseAborted(str(exc)) from None
        if not events:
            if self._objects is not None:
                self._objects.arm([])
            return []
        needs_route = any(e.at_fraction is not None or (e.pose is None and e.action != "despawn") for e in events)
        route: Sequence[Point] = ()
        if needs_route:
            try:
                route = scene.robot_route()
            except CaseAborted as exc:
                raise CaseAborted(f"the object timeline is route-relative and the route could not be planned: {exc}") from None
        try:
            resolved = resolve_objects(events, route, robot_speed=scene.speed(), delay=0.0)
        except TimelineError as exc:
            raise CaseAborted(str(exc)) from None
        self._object_driver().arm(resolved)
        self._logger.info(f"edge_case: object timeline armed, {len(resolved)} event(s) over {resolved[-1].t:.1f}s")
        return [{"entity": r.entity, "action": r.action, "model": r.event.model, "t": r.t, "pose": r.pose, "reveal_distance_m": r.reveal_distance_m, "note": r.event.note} for r in resolved]

    def _object_outcomes(self) -> dict[str, Any]:
        if self._objects is None:
            return {}
        outcomes = self._objects.outcomes()
        if not outcomes:
            return {}
        return {"object_events": [o.as_dict() for o in outcomes], "objects_pending": self._objects.pending()}

    def _reroute(self, routes: dict[str, Sequence[Point]]) -> Work:
        return self._ctx.environment_manager.set_agent_waypoints(routes)

    async def _stimulate_agents(self, names: Sequence[str], need: str, intensity: float) -> int:
        return sum([await self._ctx.environment_manager.notify_stimulus(name, need, intensity) for name in names])

    def _robot_start(self) -> Point | None:
        robots = self._ctx.robots
        if not robots:
            return None
        _, manager = next(iter(robots.items()))
        if not manager.runner.phases:
            return None
        start = manager.start_pos
        return (float(start.position.x), float(start.position.y))

    def _robot_here(self) -> Point | None:
        robots = self._ctx.robots
        if not robots:
            return None
        _, manager = next(iter(robots.items()))
        pose = manager.pose
        if pose is None:
            return None
        abstract = self._ctx.environment_manager.ezilear(pose)
        return (float(abstract.position.x), float(abstract.position.y))

    def _arm_plans(self, entries: Sequence[ArmedPlan]) -> None:
        if not entries:
            if self._waypoints is not None:
                self._waypoints.arm([])
            return
        self._waypoint_driver().arm(list(entries), origin=self._robot_start())
        for plan, trigger, reference in entries:
            gated = f" once {trigger.describe(reference)}" if trigger else ""
            self._logger.warning(f"edge_case: armed {plan.label or plan.mode} over {len(plan.homes)} agent(s), first rewrite at t={plan.at:.1f}s{gated}" + (f", every {plan.period:g}s" if plan.mode in (WaypointMode.SHUFFLE, WaypointMode.TRACK_ROBOT) and not plan.once else ""))

    def _waypoint_outcomes(self) -> dict[str, Any]:
        if self._waypoints is None:
            return {}
        outcomes = self._waypoints.outcomes()
        if not outcomes:
            return {}
        return {"waypoint_outcomes": [o.as_dict() for o in outcomes]}

    def _robot_route(self) -> list[Point]:
        robots = self._ctx.robots
        if not robots:
            raise CaseAborted("an effect needs the robot's route, but no robot is present")
        name, manager = next(iter(robots.items()))
        goals = [state.phase.pose for state in manager.runner.phases if isinstance(state.phase, GoToPhase) and state.phase.pose is not None]
        if not goals:
            raise CaseAborted(f"robot {name!r} has no goal at obstacle-generation time, so there is no route to design against. Use tm_robots:=edge_case (or :=scenario); tm_robots:=explore assigns goals lazily.")
        # the request's start, which the robot mode submitted with it; the robot is teleported there only after the obstacles spawn
        start = (float(manager.start_pos.position.x), float(manager.start_pos.position.y))
        goal = (float(goals[-1].position.x), float(goals[-1].position.y))
        if math.dist(start, goal) < 1e-6:
            raise CaseAborted(f"robot {name!r} has no distinct goal at obstacle-generation time, so there is no route to design against.")
        world_map = self._ctx.world_manager.map
        if world_map is None:
            return [start, goal]
        for label, pt in (("start", start), ("goal", goal)):
            if not in_map_bounds(world_map, Position(x=pt[0], y=pt[1])):
                raise CaseAborted(f"robot {name!r} {label} ({pt[0]:.2f}, {pt[1]:.2f}) is outside the map entirely")
        try:
            route = plan_route(world_map, start, goal, is_valid=make_is_traversable(world_map, self.node.conf.Obstacles.SAFE_DIST.value))
        except PathError as exc:
            raise CaseAborted(f"robot {name!r} has no navigable route from ({start[0]:.2f}, {start[1]:.2f}) to ({goal[0]:.2f}, {goal[1]:.2f}): {exc}") from exc
        if len(route) > 2:
            self._logger.info(f"edge_case: planned the robot's route - {len(route)} waypoint(s), {route_length(route):.2f} m")
        return route

    def _traversals(self) -> int | None:
        if self.node.has_parameter(_ROBOT_TRAVERSALS_PARAM):
            return int(self.node.get_parameter(_ROBOT_TRAVERSALS_PARAM).value)
        return None

    def _robot_hold(self) -> float:
        return max(0.0, float(self.node.rosparam[float].get("task.edge_case_robot.hold", 0.0) or 0.0))

    def _scene(self) -> Scene:
        world_map = self._ctx.world_manager.map
        safe_dist = self.node.conf.Obstacles.SAFE_DIST.value
        try:
            route, no_route = tuple(self._robot_route()), ""
        except CaseAborted as exc:
            route, no_route = (), str(exc)
            self._logger.info(f"edge_case: no route recorded for scoring ({exc})")
        self._types.scenario_dirs = self._scenario_dirs()
        return Scene(
            route=route,
            no_route=no_route,
            zones={str(z.name): tuple((float(c.x), float(c.y)) for c in z.polygon) for z in self._ctx.world_manager.world_compacted().zones},
            is_valid=make_is_valid(world_map, safe_dist),
            is_traversable=None if world_map is None else make_is_traversable(world_map, safe_dist),
            robot_speed=float(self._config.robot_speed.value),
            robot_hold=self._robot_hold(),
            types=self._types,
        )

    def _apply_prompt(self) -> None:
        text = " ".join(str(self._config.prompt.value or "").split())
        if not text:
            return
        current = self._scenario_name()
        wanted = self._prompt_scenarios.get(text)
        if wanted and wanted == current:
            return
        if wanted:
            self.node.rosparam[str].set("task.scenario.file", wanted)
            return

        import hashlib  # noqa: PLC0415

        base = str(self._config.prompt_base.value or "").strip() or (current.split("__", 1)[0] if "__rviz_" in current else current)
        world_root = self._ctx.world_manager.loaded_world_path
        base_path = world_root / "scenarios" / base / "scenario.yaml"
        if not base_path.is_file():
            raise CaseAborted(f"prompt: base scenario {base!r} not found at {base_path}")
        try:
            from arena_evaluation.generation.promptgen import generate, write_result  # noqa: PLC0415
            from arena_evaluation.generation.promptgen.prompts import Prompt  # noqa: PLC0415
            from arena_evaluation.generation.promptgen.world import WorldContext  # noqa: PLC0415
            from arena_evaluation.generation.propose import load_overrides  # noqa: PLC0415
        except ImportError as exc:
            raise CaseAborted(f"prompt: the generator (arena_evaluation.generation.promptgen) is not importable: {exc}") from None
        from arena_evaluation.generation.formats import read_scenario  # noqa: PLC0415

        base_doc = read_scenario(base_path)
        digest = hashlib.sha256(f"{base}\0{text}".encode()).hexdigest()[:8]
        prompt = Prompt(id=f"rviz_{digest}", text=text, targets=(), oracle={}, scope="rviz", world=str(self._ctx.world_manager.loaded_world))
        self._logger.warning(f"edge_case: realising prompt {prompt.id} on {base!r}: {text}")
        try:
            world = WorldContext.load(world_root, base_doc, load_overrides())
            result = generate(prompt, world=world, base_id=base, base_doc=base_doc, base_dir=base_path.parent)
        except Exception as exc:  # noqa: BLE001 - any generator failure is one abort with the reason
            raise CaseAborted(f"prompt: generation failed: {exc!r}") from None
        if result.record.status == "infeasible":
            raise CaseAborted("prompt: " + "; ".join(f"{f.step}: {f.reason}" for f in result.record.failures))
        out = write_result(result, world_root)
        if out is None:
            raise CaseAborted("prompt: nothing to write")
        for note in result.record.warnings:
            self._logger.info(f"edge_case: prompt - {note}")
        for failure in result.record.failures:
            self._logger.warning(f"edge_case: prompt - {failure.step}: {failure.reason}")
        self._prompt_scenarios[text] = out.name
        self.node.rosparam[str].set("task.scenario.file", out.name)
        self._logger.warning(f"edge_case: prompt realised as {out.name} [{', '.join(result.record.steps) or 'no steps'}] -> task.scenario.file")

    def _read_block(self) -> Block | None:
        if not bool(self._config.read_scenario_block.value):
            return None
        world = self._ctx.world_manager.loaded_world
        scenario = self._scenario_name()
        path = scenario_path(world, scenario)
        if path is None:
            if scenario:
                self._logger.debug(f"edge_case: no scenario file for {world}/{scenario}; block not read")
            return None
        block = read_block(path)
        first_time = str(path) != self._last_block_path
        self._last_block_path = str(path)
        if block is None:
            if first_time:
                self._logger.info(f"edge_case: {path} carries no 'edge_case' block - base population only")
        elif first_time:
            self._logger.warning(f"edge_case: {path} -> {block.describe()}")
        return block

    async def reset(self, **kwargs: object) -> Obstacles:
        # scored here because TM_Obstacles has no episode-end hook
        self.flush_score()
        try:
            self._apply_prompt()
        except CaseAborted as exc:
            self._logger.error(f"edge_case: {exc}")
            self._ctx.abort_episode(f"edge_case: {exc}")
            self._block = None
            return [], []
        try:
            self._block, self._block_error = self._read_block(), None
        except BlockError as exc:
            self._block, self._block_error = None, exc
        try:
            return await self._reset(**kwargs)
        finally:
            self._arm_score()

    async def teardown(self) -> None:
        self.flush_score()
        self._teardown_scorer()
        self._teardown_drivers()
        if self._base_cache is not None:
            base = self._base_cache
            self._base_cache = None
            await base.teardown()
        await super().teardown()

    async def _reset(self, **kwargs: object) -> Obstacles:
        base = self._base()
        static, dynamic = await base.reset(**kwargs)
        static, dynamic = list(static), list(dynamic)
        self.pending_regions[:] = base.pending_regions

        episode = self.node._episodes
        block = self._block or Block()
        base_record: dict[str, Any] = {
            "run_seed": episode.run_seed,
            "episode_id": episode.current.episode_id,
            "world": self._ctx.world_manager.loaded_world,
            "seed": episode.current.seed,
            "scenario": self._scenario_name(),
            "case_id": block.id,
            "base": block.base,
            "prompt_id": block.prompt_id,
            "steps": tuple(block.steps),
            "effects": tuple(e.as_dict() for e in block.effects),
        }

        try:
            if self._block_error is not None:
                raise CaseAborted(str(self._block_error))
            scene = self._scene()
            object_events = self._arm_objects(scene)
            if object_events:
                base_record["object_timeline"] = self._timeline_source()
                base_record["object_events"] = tuple(object_events)
            case = build(block, dynamic, scene)
            if case.geometry.get("geometry_retries"):
                g = case.geometry
                self._logger.warning(f"edge_case: requested angle={g['approach_angle_requested']:g} at {g['encounter_at_requested']:g} was obstructed; solved at {g['approach_angle_achieved']:g} / {g['encounter_at_achieved']:g} after {g['geometry_retries']} retries")
            if case.invalid_spawns:
                self._logger.warning(f"edge_case: {len(case.invalid_spawns)} agent(s) spawn outside free space ({', '.join(case.invalid_spawns)}); recorded, not fatal")
            injected, plans = case.injected, case.plans
            self._arm_plans(plans)

            status = "ok" if not block.empty else "baseline"
            self._write_case(
                CaseRecord(
                    **base_record,
                    **case.geometry,
                    injected=len(injected),
                    invalid_spawns=case.invalid_spawns,
                    plans=tuple(p.as_dict() for p, _t, _r in plans),
                    traversals_seen=self._traversals(),
                    status=status,
                )
            )
            if block.empty:
                self._logger.info("edge_case: base population only, nothing added")
            else:
                self._logger.warning(f"edge_case: {block.id or 'case'} built - {len(injected)} injected, {len(plans)} plan(s), {len(dynamic)} base agent(s)")
            return static, [*dynamic, *injected]

        except (CaseAborted, AgentTypeError, GeometryError, WaypointError) as exc:
            self._write_case(CaseRecord(**base_record, status="aborted", reason=str(exc)))
            self._logger.error(f"edge_case: {exc}")
            self._ctx.abort_episode(f"edge_case: {exc}")
            return static, dynamic
