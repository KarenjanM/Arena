import asyncio
from collections.abc import Sequence

import attrs
from arena_rclpy_mixins.ROSParamServer import ROSParamT
from arena_simulation_setup.tree.World import WorldIdentifier

from task_generator.shared import Pose, Position, PositionRadius
from task_generator.tasks.obstacles._validation import make_is_valid
from task_generator.tasks.registry import default_scenario
from task_generator.tasks.robots import TM_Robots
from task_generator.tasks.robots.request import GoToPhase, LoopPhase, TaskPhase, TaskRequest

# a robot not spawned yet is assumed this wide
NOMINAL_ROBOT_RADIUS = 0.3


def spawn_clearance(safe_distances: Sequence[float], spawn_safe_dist: float) -> float:
    """Free space a robot's start and goal need: the widest managed robot's, else a nominal robot's."""
    return max(safe_distances) if safe_distances else NOMINAL_ROBOT_RADIUS + spawn_safe_dist


class TM_Scenario(TM_Robots):
    _config: ROSParamT[str]
    _linger_after_completion: ROSParamT[bool]
    _idle_robots: set[str]

    async def reset(self) -> None:
        await super().reset()

        self._idle_robots = set()

        clearance = spawn_clearance([r.safe_distance for r in self._ctx.robots.values()], self.node.conf.Robot.SPAWN_ROBOT_SAFE_DIST.value)
        is_valid = make_is_valid(self._ctx.world_manager.map, clearance)
        zone_conv = self._ctx.world_manager.world_compacted().zone_converter(
            self.node.conf.General.RNG.stream("robots", "scenario"),
            is_valid=is_valid,
        )
        scenario_view = WorldIdentifier(self._ctx.world_manager.loaded_world).resolve_sync().scenario(self._config.value).resolve_sync()
        scenario = scenario_view.load(converter=zone_conv)
        SCENARIO_ROBOTS = scenario.robots
        declared_pedestrians = frozenset(human.name for human in scenario.humans)

        managed_robots = list(self._ctx.robots.values())

        scenario_robots_length = len(SCENARIO_ROBOTS)
        setup_robot_length = len(managed_robots)

        if setup_robot_length > scenario_robots_length:
            managed_robots = managed_robots[:scenario_robots_length]
            self._logger.warn("Robot setup contains more robots than the scenario file.", once=True)

        if scenario_robots_length > setup_robot_length:
            SCENARIO_ROBOTS = SCENARIO_ROBOTS[:setup_robot_length]
            self._logger.warn("Scenario file contains more robots than setup.", once=True)

        # Shift level-tagged poses from level-local to the flattened map frame.
        level_origins = self._ctx.world_manager.map.level_origins

        def _to_map_frame(pose: Pose, level: str) -> Pose:
            if not level:
                return pose
            ox, oy = level_origins.get(level, (0.0, 0.0))
            return Pose(
                position=Position(pose.position.x + ox, pose.position.y + oy, pose.position.z),
                orientation=pose.orientation,
            )

        def localize(phase: TaskPhase, goal_level: str, forbidden: list[PositionRadius], safe_distance: float) -> TaskPhase:
            if isinstance(phase, LoopPhase):
                return attrs.evolve(phase, phases=[localize(inner, goal_level, forbidden, safe_distance) for inner in phase.phases])
            if isinstance(phase, GoToPhase) and phase.pose is not None:
                goto_pose = _to_map_frame(phase.pose, goal_level)
                forbidden.append(PositionRadius(x=goto_pose.position.x, y=goto_pose.position.y, radius=safe_distance))
                return attrs.evolve(phase, pose=goto_pose)
            return phase

        for robot, config in zip(managed_robots, SCENARIO_ROBOTS, strict=False):
            start_pose = _to_map_frame(config.start, "" if config.start_zone else config.level)
            self._start_poses[robot.name] = start_pose

            forbidden: list[PositionRadius] = [
                PositionRadius(x=start_pose.position.x, y=start_pose.position.y, radius=robot.safe_distance),
            ]
            phases = [localize(phase, config.goal_level, forbidden, robot.safe_distance) for phase in config.phases]

            # An empty phase list intentionally describes a stationary robot.
            # Do not submit it to RobotManager: an empty TaskRequest has no
            # dispatchable phase.  Keep the scenario active until its normal
            # timeout (or until another non-idle robot finishes) so that
            # recordings can capture pedestrians passing a parked robot.
            if phases:
                await robot.submit_task(TaskRequest(phases=phases, conditions=list(config.conditions)), start_pose, pedestrians=declared_pedestrians)
            else:
                self._idle_robots.add(robot.name)
            self._ctx.world_manager.forbid(forbidden)

    @property
    async def done(self) -> bool:
        """Treat empty scenario phase lists as parked, not completed, robots."""
        active_robots = [manager for name, manager in self._ctx.robots.items() if name not in self._idle_robots]
        if not active_robots:
            return False
        if self._linger_after_completion.value:
            return False
        return all(await asyncio.gather(*(manager.is_done for manager in active_robots)))

    def __init__(self, **kwargs: object) -> None:
        TM_Robots.__init__(self, **kwargs)

        self._config = self.node.ROSParam[str](
            self.namespace('file'),
            default_scenario(self._ctx.world_manager.loaded_world),
        )
        self._linger_after_completion = self.node.ROSParam[bool](
            self.namespace('linger_after_completion'),
            False,
        )
