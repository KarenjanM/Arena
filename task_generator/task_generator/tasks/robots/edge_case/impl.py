from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence

import attrs
from arena_rclpy_mixins.ROSParamServer import ROSParamT

from task_generator.manager.robot_manager.robot_manager import RobotManager
from task_generator.shared import Pose
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, default_scenario
from task_generator.tasks.robots import TM_Robots
from task_generator.tasks.robots.request import GoToPhase, TaskPhase
from task_generator.tasks.robots.scenario.impl import TM_Scenario

_NEVER = -1.0


@attrs.define
class _Progress:
    best_distance: float = math.inf
    last_progress_at: float = _NEVER
    phase_key: tuple[float, float] | None = None

    def reset(self, now: float) -> None:
        self.best_distance = math.inf
        self.last_progress_at = now
        self.phase_key = None


def plan_legs(start: Pose, phases: Sequence[TaskPhase], *, traversals: int, retries: int, hold: float, accept_within: float, goal_tolerance: float) -> list[TaskPhase]:
    """The scenario route's last goal as alternating legs from `start`, after an optional hold; a route without a goal is kept."""
    goal = next((p.pose for p in reversed(phases) if isinstance(p, GoToPhase) and p.pose is not None), None)
    if goal is None:
        return list(phases)
    # nav2 will not plan into a goal cell inside the robot's inflation, so a leg within `accept_within` is met
    tolerance = max(accept_within, goal_tolerance) if accept_within > 0.0 else None
    legs = TM_EdgeCase.traversal_phases(start, goal, traversals, retries, tolerance)
    # a pose-less phase holds where the robot stands, inside its own tolerance, so the watchdog leaves it alone
    return [GoToPhase(hold_time=hold), *legs] if hold > 0.0 else legs


def leg_blocked(progress: _Progress, pose: Pose, phase: GoToPhase, now: float, *, min_improvement: float, timeout: float) -> float | None:
    """The distance still to go when the robot made no progress on its leg for `timeout`, else None."""
    assert phase.pose is not None
    goal = phase.pose
    key = (goal.position.x, goal.position.y)
    if progress.phase_key != key:
        progress.reset(now)
        progress.phase_key = key
    distance = math.dist((pose.position.x, pose.position.y), key)
    # inside the leg's tolerance the robot is holding or arriving, not obstructed
    if distance <= (phase.tolerance_radius or 0.0):
        return None
    return distance if TM_EdgeCase.is_blocked(progress, distance, now, min_improvement=min_improvement, timeout=timeout) else None


class TM_EdgeCase(TM_Scenario):
    """Unlike the base class, a timeout with work left ends the episode FAILED rather than SUCCESS."""

    _traversals: ROSParamT[int]
    _blocked_distance: ROSParamT[float]
    _blocked_timeout: ROSParamT[float]
    _on_blocked: ROSParamT[str]

    @staticmethod
    def traversal_phases(start: Pose, goal: Pose, traversals: int, retries: int = 1, tolerance: float | None = None) -> list[TaskPhase]:
        """``2N - 1`` alternating goal/start legs, each ``retries`` times; one traversal matches the scenario mode."""
        if traversals < 1:
            raise ValueError(f"traversals must be >= 1, got {traversals}")
        if retries < 1:
            raise ValueError(f"retries must be >= 1, got {retries}")
        phases: list[TaskPhase] = []
        for i in range(2 * traversals - 1):
            phases.extend(GoToPhase(pose=goal if i % 2 == 0 else start, tolerance_radius=tolerance) for _ in range(retries))
        return phases

    @staticmethod
    def is_blocked(
        progress: _Progress,
        distance: float,
        now: float,
        *,
        min_improvement: float,
        timeout: float,
    ) -> bool:
        if progress.last_progress_at == _NEVER:
            progress.last_progress_at = now

        if distance < progress.best_distance - min_improvement:
            progress.best_distance = distance
            progress.last_progress_at = now
            return False

        return (now - progress.last_progress_at) >= timeout

    def _goal_for(self, robot: RobotManager) -> GoToPhase | None:
        phase = robot.active_dispatch_phase()
        return phase if isinstance(phase, GoToPhase) and phase.pose is not None else None

    # same figure as the obstacles mode's case clock, `waypoint_driver.ROBOT_DEPART_M`
    DEPART_M = 0.3

    def _log_departures(self) -> None:
        now = float(self.node.sim_time.sec) + self.node.sim_time.nanosec * 1e-9
        for name, robot in self._ctx.robots.items():
            if name in self._departed or robot.pose is None or not robot.placed:
                continue
            start = robot.start_pos
            here = self._ctx.environment_manager.ezilear(robot.pose)
            moved = math.dist((here.position.x, here.position.y), (start.position.x, start.position.y))
            if moved >= self.DEPART_M:
                self._departed.add(name)
                self._logger.warning(f"edge_case robots: {name!r} under way ({moved:.2f} m from its start) {now - self._last_reset:.1f}s after the reset")

    def _check_blocked(self) -> str | None:
        now = float(self.node.sim_time.sec) + self.node.sim_time.nanosec * 1e-9
        for name, robot in self._ctx.robots.items():
            pose = robot.pose
            phase = self._goal_for(robot)
            if pose is None or phase is None or phase.pose is None:
                continue
            progress = self._progress.setdefault(name, _Progress())
            distance = leg_blocked(progress, pose, phase, now, min_improvement=self._blocked_distance.value, timeout=self._blocked_timeout.value)
            if distance is not None:
                goal = phase.pose
                return f"robot {name!r} blocked: no progress toward ({goal.position.x:.2f}, {goal.position.y:.2f}) for {self._blocked_timeout.value:.0f}s, still {distance:.2f}m away"
        return None

    def _plan(self, start: Pose, phases: list[TaskPhase]) -> list[TaskPhase]:
        traversals = max(1, int(self._traversals.value))
        retries = max(1, int(self._retries.value))
        hold = max(0.0, float(self._hold.value))
        planned = plan_legs(start, phases, traversals=traversals, retries=retries, hold=hold, accept_within=float(self._accept_within.value), goal_tolerance=float(self.node.conf.Robot.GOAL_TOLERANCE_RADIUS.value))
        self._logger.info(f"edge_case robots: {f'holds {hold:.0f}s at its start, then ' if hold > 0.0 else '-> '}{traversals} traversal(s) x {retries} attempt(s), {len(planned) - (hold > 0.0)} leg(s)")
        return planned

    async def reset(self, **kwargs: object) -> None:
        await super().reset(**kwargs)
        self._last_reset = self.node.sim_time.sec
        self._progress = {}
        self._departed: set[str] = set()

    def _log_outcomes(self) -> None:
        """A phase also ends when nav2 gives up, so the analysis tools read the arrival rate from this line."""
        for name, robot in self._ctx.robots.items():
            goals = [state.phase.pose for state in robot.runner.phases if isinstance(state.phase, GoToPhase) and state.phase.pose is not None]
            pose = robot.pose
            if not goals or pose is None:
                self._logger.warning(f"edge_case robots: {name!r} request spent with no goal to measure against ({len(robot.runner.phases)} phase(s), pose {'known' if pose is not None else 'unknown'})")
                continue
            goal = self._ctx.environment_manager.realize(goals[-1])
            distance = math.hypot(pose.position.x - goal.position.x, pose.position.y - goal.position.y)
            tolerance = max(float(self._accept_within.value), float(self.node.conf.Robot.GOAL_TOLERANCE_RADIUS.value))
            elapsed = self.node.sim_time.sec - self._last_reset
            if distance <= tolerance:
                self._logger.warning(f"edge_case robots: {name!r} reached its goal ({distance:.2f} m off) after {elapsed:.0f}s")
            else:
                self._logger.warning(f"edge_case robots: {name!r} gave up {distance:.2f} m short of ({goal.position.x:.2f}, {goal.position.y:.2f}) after {elapsed:.0f}s: every attempt ended without arriving")

    @property
    async def done(self) -> bool:
        self._log_departures()
        reason = self._check_blocked()
        if reason is not None:
            if self._on_blocked.value == "abort":
                self._logger.warning(f"edge_case robots: {reason}")
                self._ctx.abort_episode(reason)
                return True
            self._logger.warning(f"edge_case robots: {reason} (on_blocked=continue, not ending the episode)")

        if not self._ctx.robots:
            return False

        finished = all(await asyncio.gather(*(robot.is_done for robot in self._ctx.robots.values())))
        if finished:
            self._log_outcomes()
            return True

        elapsed = self.node.sim_time.sec - self._last_reset
        if elapsed > self.node.conf.Robot.TIMEOUT.value:
            self._ctx.abort_episode(f"timed out after {elapsed}s with traversals incomplete")
            return True

        return False

    def __init__(self, **kwargs: object) -> None:
        # skips TM_Scenario.__init__ to read the shared `task.scenario.file`, so route and crowd share one scenario
        TM_Robots.__init__(self, **kwargs)
        self._config = self.node.ROSParam[str](
            _REGISTRY_NAMESPACE("scenario")("file"),
            default_scenario(self._ctx.world_manager.loaded_world),
        )

        self._progress: dict[str, _Progress] = {}
        self._departed: set[str] = set()
        self._traversals = self.node.ROSParam[int](self.namespace("traversals"), 1)
        self._retries = self.node.ROSParam[int](self.namespace("retries"), 1)
        self._hold = self.node.ROSParam[float](self.namespace("hold"), 0.0)
        self._accept_within = self.node.ROSParam[float](self.namespace("accept_within"), 0.6)
        self._blocked_distance = self.node.ROSParam[float](self.namespace("blocked_distance"), 0.25)
        self._blocked_timeout = self.node.ROSParam[float](self.namespace("blocked_timeout"), 15.0)
        self._on_blocked = self.node.ROSParam[str](self.namespace("on_blocked"), "abort")
