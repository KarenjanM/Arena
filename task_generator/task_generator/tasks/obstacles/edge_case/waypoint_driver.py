from __future__ import annotations

import collections
from collections.abc import Callable, Coroutine, Mapping, Sequence
from typing import TYPE_CHECKING

import attrs

from .triggers import Trigger, distance_to
from .waypoints import Mode, Plan, Point, WaypointOutcome

if TYPE_CHECKING:
    from rclpy.impl.rcutils_logger import RcutilsLogger

Work = Coroutine[None, None, object]

# slower than the object driver: every re-issue costs the pedestrian backend a path recomputation
DEFAULT_RATE_HZ = 2.0

ROBOT_DEPART_M = 0.3
# a jump this large between ticks is the reset placing the robot, which moves the origin rather than starting the clock
ROBOT_TELEPORT_M = 1.5
ROBOT_SETTLE_S = 10.0
SPEED_WINDOW_S = 2.0


@attrs.define
class Armed:
    plan: Plan
    trigger: Trigger | None = None
    reference: Point | None = None
    armed_at: float | None = None
    t0: float | None = None
    last: float | None = None
    issues: int = 0
    released: bool = False
    done: bool = False
    outcome: WaypointOutcome = attrs.Factory(WaypointOutcome)


class WaypointDriver:
    """Fires armed route-rewrite plans against sim time once the robot is under way; `tick` is called by the mode's sim-time loop."""

    def __init__(
        self,
        *,
        reroute: Callable[[Mapping[str, Sequence[Point]]], Work],
        robot_pose: Callable[[], Point | None],
        submit: Callable[[Work], None],
        logger: RcutilsLogger,
        stimulus: Callable[[Sequence[str], str, float], Work] | None = None,
        depart_after_m: float | None = None,
    ) -> None:
        self._reroute = reroute
        self._stimulus = stimulus
        self._robot_pose = robot_pose
        self._submit = submit
        self._armed: list[Armed] = []
        self._depart_after_m = depart_after_m
        self._origin: Point | None = None
        self._origin_given = False
        self._settled = False
        self._last_seen: Point | None = None
        self._track: collections.deque[tuple[float, Point]] = collections.deque()
        self._armed_at: float | None = None
        self.clock_started_at: float | None = None
        self._logger = logger

    def arm(self, entries: Sequence[tuple[Plan, Trigger | None, Point | None]], origin: Point | None = None) -> None:
        self._armed = [
            Armed(
                plan=plan,
                trigger=trigger,
                reference=reference,
                outcome=WaypointOutcome(
                    label=plan.label,
                    mode=str(plan.mode),
                    agents=len(plan.homes),
                    trigger=trigger.describe(reference) if trigger else "",
                ),
            )
            for plan, trigger, reference in entries
        ]
        self._origin = origin
        self._origin_given = origin is not None
        self._settled = False
        self._last_seen = None
        self._armed_at = None
        self.clock_started_at = None

    def outcomes(self) -> list[WaypointOutcome]:
        return [a.outcome for a in self._armed]

    def tick(self, now: float) -> None:
        if not self._armed:
            return
        if now <= 0.0:
            return
        robot = self._robot_pose()
        speed = self._robot_speed(now, robot)
        if not self._robot_under_way(now, robot):
            return
        for armed in self._armed:
            if not armed.done:
                self._tick_one(armed, now, robot, speed)

    def _robot_speed(self, now: float, robot: Point | None) -> float | None:
        if robot is None:
            return None
        if self._track and (distance_to(robot, self._track[-1][1]) or 0.0) >= ROBOT_TELEPORT_M:
            self._track.clear()
        self._track.append((now, robot))
        while now - self._track[0][0] > SPEED_WINDOW_S:
            self._track.popleft()
        span = now - self._track[0][0]
        if span < SPEED_WINDOW_S / 2:
            return None
        points = [p for _, p in self._track]
        return sum(distance_to(b, a) or 0.0 for a, b in zip(points, points[1:], strict=False)) / span

    def _robot_under_way(self, now: float, robot: Point | None) -> bool:
        """The case clock starts once the robot has moved `depart_after_m` from its start, so an onset counts from the drive, not the reset."""
        if self.clock_started_at is not None:
            return True
        if self._depart_after_m is None:
            self.clock_started_at = now
            return True
        if self._armed_at is None:
            self._armed_at = now
        if robot is None:
            return False
        jumped = self._last_seen is not None and (distance_to(robot, self._last_seen) or 0.0) >= ROBOT_TELEPORT_M
        self._last_seen = robot
        if self._origin_given and not self._settled:
            # plans are armed before the reset has placed the robot
            near = distance_to(robot, self._origin)
            if near is not None and near < self._depart_after_m:
                self._settled = True
            elif now - self._armed_at >= ROBOT_SETTLE_S:
                self._logger.warning(f"edge_case: robot never seen on its start {self._origin} ({near:.2f} m off after {ROBOT_SETTLE_S:.0f}s); taking its departure from here")
                self._origin_given = False
                self._origin = robot
            return False
        if not self._origin_given and (self._origin is None or jumped):
            self._origin = robot
            return False
        moved = distance_to(robot, self._origin)
        if moved is None or moved < self._depart_after_m:
            return False
        self.clock_started_at = now
        self._logger.warning(f"edge_case: robot under way ({moved:.2f} m from its start, {now - self._armed_at:.1f}s after the plans were armed) at sim t={now:.1f}s - the case clock starts now")
        return True

    def _tick_one(self, armed: Armed, now: float, robot: Point | None, speed: float | None = None) -> None:
        plan = armed.plan
        if armed.armed_at is None:
            armed.armed_at = now

        if armed.t0 is None:
            if armed.trigger is not None:
                distance = distance_to(robot, armed.reference)
                if not armed.trigger.satisfied(distance, speed):
                    return
                armed.outcome.t_trigger = now - armed.armed_at
                self._logger.warning(f"edge_case: {plan.label or plan.mode} trigger fired at t={armed.outcome.t_trigger:.1f}s - {armed.trigger.describe(armed.reference)}, robot at {distance:.2f}m")
            armed.t0 = now
            return

        elapsed = now - armed.t0
        if elapsed < plan.at:
            return

        if plan.mode in (Mode.HOLD, Mode.TRACK_ROBOT) and armed.issues and plan.duration > 0 and elapsed >= plan.at + plan.duration:
            armed.done = True
            armed.outcome.t_release = elapsed
            routes = plan.resume_routes()
            if routes:
                self._send(armed, routes, elapsed)
            return

        if armed.last is not None and elapsed - armed.last < plan.period:
            return

        if plan.mode is Mode.STIMULUS:
            armed.done = True
            armed.issues += 1
            armed.outcome.t_first = elapsed
            self._send_once(armed, elapsed)
            return

        routes = self._routes(armed, robot)
        if routes is None:
            return

        armed.last = elapsed
        armed.issues += 1
        if armed.outcome.t_first is None:
            armed.outcome.t_first = elapsed
        if plan.mode is Mode.RALLY or (plan.mode is Mode.SHUFFLE and plan.once):
            armed.done = True
        if plan.mode is Mode.HOLD and plan.duration <= 0:
            armed.done = True
        self._send(armed, routes, elapsed)

    def _routes(self, armed: Armed, robot: Point | None) -> dict[str, list[Point]] | None:
        plan = armed.plan
        if plan.mode is Mode.RALLY:
            return {name: [plan.targets[name]] for name in plan.homes}
        if plan.mode is Mode.TRACK_ROBOT:
            if robot is None:
                return None
            return {name: [robot] for name in plan.homes}
        if plan.mode is Mode.HOLD:
            if armed.issues:
                return None
            return plan.hold_routes()
        return plan.shuffle_routes(armed.issues)

    def _send_once(self, armed: Armed, elapsed: float) -> None:
        if self._stimulus is None:
            armed.outcome.failures += 1
            armed.outcome.reason = f"no {armed.plan.mode} hook"
            return
        self._submit(self._run_once(armed, elapsed))

    async def _run_once(self, armed: Armed, elapsed: float) -> None:
        plan = armed.plan
        wanted = len(plan.homes)
        try:
            assert self._stimulus is not None
            ok = await self._stimulus(sorted(plan.homes), plan.need, plan.intensity)
        except Exception as exc:
            armed.outcome.failures += 1
            self._logger.warning(f"edge_case: {plan.mode} at t={elapsed:.1f}s failed ({exc!r})")
            return
        n = int(ok) if isinstance(ok, int) and not isinstance(ok, bool) else (wanted if ok else 0)
        armed.outcome.issued += n
        if n < wanted:
            armed.outcome.failures += wanted - n
        self._logger.warning(f"edge_case: {plan.label or plan.mode} at t={elapsed:.1f}s - {n} of {wanted} agent(s) stimulated ({plan.need})")

    def _send(self, armed: Armed, routes: Mapping[str, Sequence[Point]], elapsed: float) -> None:
        self._submit(self._run(armed, routes, elapsed))

    async def _run(self, armed: Armed, routes: Mapping[str, Sequence[Point]], elapsed: float) -> None:
        try:
            ok = await self._reroute(routes)
        except Exception as exc:
            armed.outcome.failures += 1
            self._logger.warning(f"edge_case: reroute at t={elapsed:.1f}s failed ({exc!r})")
            return
        if not ok:
            armed.outcome.failures += 1
            self._logger.warning(f"edge_case: reroute at t={elapsed:.1f}s reached no agent - {len(routes)} name(s) did not resolve")
            return
        armed.outcome.issued += len(routes)
        if armed.outcome.issued == len(routes):
            self._logger.warning(f"edge_case: {armed.plan.label or armed.plan.mode} fired at t={elapsed:.1f}s - {len(routes)} agent(s) rerouted")
