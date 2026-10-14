"""Patrol-with-service: the robot loops the scenario's checkpoints until it perceives a pedestrian's call (call clip published, within sensor range, not behind a wall)."""

from __future__ import annotations

import math

import numpy as np
from arena_people_msgs.msg import Pedestrians
from arena_rclpy_mixins.shared import Namespace
from arena_simulation_setup.tree.World import WorldIdentifier
from arena_simulation_setup.tree.World.Scenario import ScenarioGotoPhase
from rclpy.qos import qos_profile_sensor_data

from task_generator.manager.world_manager.utils import WorldOccupancy
from task_generator.shared import Orientation, Pose, Position
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, default_scenario
from task_generator.tasks.robots import TM_Robots
from task_generator.tasks.robots.request import GoToPhase, TaskRequest

from . import DEFAULT_CLIP

_SCENARIO_NS = _REGISTRY_NAMESPACE("scenario")


def _short_name(name: str) -> str:
    """Pedestrians reach the topic namespaced by their environment (`env_0/caller`)."""
    return name.rsplit("/", 1)[-1]


class TM_Patrol(TM_Robots):
    """Loops the scenario's checkpoints; diverts to a pedestrian whose call it can perceive."""

    async def reset(self) -> None:
        await super().reset()

        world = WorldIdentifier(self._ctx.world_manager.loaded_world)
        zone_conv = self._ctx.world_manager.world_compacted().zone_converter(self.node.conf.General.RNG.stream("robots", "patrol"))
        name = self._scenario_file.value or default_scenario(self._ctx.world_manager.loaded_world)
        scenario = world.resolve_sync().scenario(name).resolve_sync().load(converter=zone_conv)

        if not self._service_agent.value:
            raise ValueError(f"patrol: set {self.namespace('service_agent')} to the pedestrian whose call the robot answers")
        self._route = []
        self._route_local = []
        self._frame_offset = (0.0, 0.0)
        self._index = 0
        self._current_goal = None
        self._walls_cache = None
        self._served_since = None
        self._visible_since = None
        self._diverted = False
        self._divert_pending = False
        self._service_goal = None
        self._peds = None

        robots = list(self._ctx.robots.values())
        for robot, config in zip(robots, scenario.robots, strict=False):
            self._start_poses[robot.name] = config.start
            self._scenario_start = config.start
            self._route_local = [phase.goto for phase in config.phase_list() if isinstance(phase, ScenarioGotoPhase)]
            await robot.move(config.start)
            break  # one robot patrols

        if self._subscription is None:
            self._subscription = self.node.create_subscription(
                Pedestrians,
                Namespace(self.node.get_namespace())("arena_peds"),
                self._on_peds,
                qos_profile_sensor_data,
            )

    def _adopt_frame(self, spawn_pose: Pose) -> None:
        """Shift the route by the offset between the robot's reported pose and the scenario start, when it exceeds a meter."""
        ox = spawn_pose.position.x - self._scenario_start.position.x
        oy = spawn_pose.position.y - self._scenario_start.position.y
        if math.hypot(ox, oy) < 1.0:
            ox = oy = 0.0
        self._frame_offset = (ox, oy)
        self._route = [Pose(position=Position(pose.position.x + ox, pose.position.y + oy, pose.position.z), orientation=pose.orientation) for pose in self._route_local]
        self._logger.info(f"patrol route in the robot's frame, offset ({ox:.2f}, {oy:.2f}): {[(round(p.position.x, 1), round(p.position.y, 1)) for p in self._route]}")

    ARRIVED_M = 1.2

    def _next_checkpoint(self) -> TaskRequest:
        """One checkpoint per request, advanced on arrival rather than on the adapter's verdict."""
        i = self._index % len(self._route)
        self._current_goal = self._route[i]
        self._index += 1
        # the adapter adds the env offset itself, so the goal is handed over in the scenario frame
        return TaskRequest(phases=[GoToPhase(pose=self._route_local[i])])

    def _on_peds(self, msg: Pedestrians) -> None:
        """Keep the latest pedestrians, everything else happens on the episode tick."""
        self._peds = msg

    def _look(self, pose: Pose) -> None:
        """Latch arrival and a perceived call, from the tick."""
        here = (pose.position.x, pose.position.y)
        wanted = self._service_agent.value

        target = self._agent_position(wanted)
        if target is not None and math.dist(here, (target.x, target.y)) <= self._service_radius.value:
            if self._served_since is None:
                self._served_since = self._now()
                self._logger.info(f"reached {wanted or 'the caller'} at {math.dist(here, (target.x, target.y)):.2f} m")
        elif self._service_dwell.value > 0.0:
            self._served_since = None

        if self._diverted or not self._route:
            return
        calling = self._calling()
        if calling is None or not self._perceives(pose, calling[1]):
            self._visible_since = None
            return
        now = self._now()
        if self._visible_since is None:
            self._visible_since = now
        if now - self._visible_since >= self._recognition_dwell.value:
            self._service_goal = self._service_pose(pose, calling[1])
            self._divert_pending = True
            self._logger.info(f"call from {calling[0]} perceived at {math.dist(here, (calling[1].x, calling[1].y)):.1f} m")

    def _now(self) -> float:
        return self.node.sim_time.sec + self.node.sim_time.nanosec * 1e-9

    # -- perception ----------------------------------------------------------------------------

    def _calling(self) -> tuple[str, Position] | None:
        """The pedestrian publishing the call clip right now, if any."""
        msg = self._peds
        if msg is None:
            return None
        wanted = self._service_agent.value
        for ped in msg.pedestrians:
            if _short_name(ped.name) != wanted:
                continue
            if any(g.clip == self._clip.value for g in ped.gestures):
                return ped.name, Position(ped.pose.position.x, ped.pose.position.y, ped.pose.position.z)
        return None

    def _agent_position(self, name: str) -> Position | None:
        msg = self._peds
        if msg is None:
            return None
        for ped in msg.pedestrians:
            if _short_name(ped.name) == name:
                return Position(ped.pose.position.x, ped.pose.position.y, ped.pose.position.z)
        return None

    def _walls(self) -> np.ndarray:
        """The wall layer as a boolean grid, resolved once."""
        if self._walls_cache is None:
            self._walls_cache = WorldOccupancy.fullish(self._ctx.world_manager.map.occupancy.walls)
        return self._walls_cache

    def _line_of_sight(self, a: Position, b: Position) -> bool:
        """False when a wall cell lies on the segment, so a call from inside a room does not carry through it."""
        world_map = self._ctx.world_manager.map
        walls = self._walls()
        steps = max(2, int(math.dist((a.x, a.y), (b.x, b.y)) / max(world_map.resolution, 1e-3)) + 1)
        xs = np.linspace(a.x, b.x, steps)
        ys = np.linspace(a.y, b.y, steps)
        rows, cols = [], []
        for x, y in zip(xs, ys, strict=True):
            r, c = world_map.tf_pos2grid(Position(float(x), float(y), 0.0))
            rows.append(int(r))
            cols.append(int(c))
        rows = np.clip(np.array(rows), 0, walls.shape[0] - 1)
        cols = np.clip(np.array(cols), 0, walls.shape[1] - 1)
        return not bool(walls[rows, cols].any())

    def _to_scenario_frame(self, p: Position) -> Position:
        """Robot-frame position -> the frame of the world map and its wall grid."""
        ox, oy = self._frame_offset
        return Position(p.x - ox, p.y - oy, p.z)

    def _to_scenario_pose(self, pose: Pose) -> Pose:
        return Pose(position=self._to_scenario_frame(pose.position), orientation=pose.orientation)

    def _perceives(self, robot_pose: Pose, ped: Position) -> bool:
        if math.dist((robot_pose.position.x, robot_pose.position.y), (ped.x, ped.y)) > self._sensor_range.value:
            return False
        return not self._require_los.value or self._line_of_sight(self._to_scenario_frame(robot_pose.position), self._to_scenario_frame(ped))

    # -- episode -------------------------------------------------------------------------------

    @property
    async def done(self) -> bool:
        """Answer a perceived call, keep the loop running, and end the episode once the caller is served."""
        robots = list(self._ctx.robots.values())
        if not robots or not self._route_local:
            return False
        robot = robots[0]
        pose = robot.pose
        if pose is None:
            return False

        if not self._route:
            self._adopt_frame(pose)
            await robot.submit_task(self._next_checkpoint())
            return False

        self._look(pose)
        now = self._now()

        if self._served_since is not None and now - self._served_since >= self._service_dwell.value:
            return True

        if self._divert_pending and not self._diverted and self._service_goal is not None:
            await robot.submit_task(TaskRequest(phases=[GoToPhase(pose=self._to_scenario_pose(self._service_goal))]))
            self._diverted = True
            self._divert_pending = False
            self._logger.info(f"goal overridden by the caller at ({self._service_goal.position.x:.1f}, {self._service_goal.position.y:.1f})")

        if not self._diverted and self._current_goal is not None:
            here = (pose.position.x, pose.position.y)
            goal = (self._current_goal.position.x, self._current_goal.position.y)
            if math.dist(here, goal) <= self.ARRIVED_M:
                await robot.submit_task(self._next_checkpoint())

        return False

    def _service_pose(self, robot_pose: Pose, ped: Position) -> Pose:
        """A pose service_standoff short of the pedestrian, on the robot's side, facing them."""
        dx, dy = ped.x - robot_pose.position.x, ped.y - robot_pose.position.y
        distance = math.hypot(dx, dy)
        yaw = math.atan2(dy, dx)
        standoff = min(self._service_standoff.value, distance)
        return Pose(
            position=Position(ped.x - math.cos(yaw) * standoff, ped.y - math.sin(yaw) * standoff, 0.0),
            orientation=Orientation.from_yaw(yaw),
        )

    def __init__(self, **kwargs: object) -> None:
        TM_Robots.__init__(self, **kwargs)
        self._subscription = None
        self._peds: Pedestrians | None = None
        self._route: list[Pose] = []
        self._route_local: list[Pose] = []
        self._frame_offset: tuple[float, float] = (0.0, 0.0)
        self._index = 0
        self._current_goal: Pose | None = None
        self._walls_cache: np.ndarray | None = None
        self._scenario_start = Pose()
        self._service_goal: Pose | None = None
        self._divert_pending = False
        self._served_since: float | None = None
        self._visible_since: float | None = None
        self._diverted = False

        self._scenario_file = self.node.ROSParam[str](_SCENARIO_NS("file"), "")
        self._service_agent = self.node.ROSParam[str](self.namespace("service_agent"), "")
        self._clip = self.node.ROSParam[str](self.namespace("clip"), DEFAULT_CLIP)
        self._service_radius = self.node.ROSParam[float](self.namespace("service_radius"), 2.5)
        self._service_dwell = self.node.ROSParam[float](self.namespace("service_dwell"), 0.0)
        self._service_standoff = self.node.ROSParam[float](self.namespace("service_standoff"), 0.8)
        self._sensor_range = self.node.ROSParam[float](self.namespace("sensor_range"), 12.0)
        self._require_los = self.node.ROSParam[bool](self.namespace("require_line_of_sight"), True)
        self._recognition_dwell = self.node.ROSParam[float](self.namespace("recognition_dwell"), 0.5)
