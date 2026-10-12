from __future__ import annotations

import asyncio

import rclpy.qos
from arena_rclpy_mixins import ArenaMixinNode
from arena_rclpy_mixins.Time import Time
from flatsim_msgs.srv import ControlWorld, DeletePrefix

from arena_runtime.sim import SimLifecycle, SimUnavailable

SERVER = '/flatsim'
SERVICE_QOS = rclpy.qos.QoSProfile(depth=100)

_EPS = 1e-6


def env_prefix(env_id: int) -> str:
    """Name prefix of everything env_id puts into the world."""
    return f'env_{env_id}/'


class FlatsimHost(SimLifecycle):
    """Pauses and steps the flatsim server, which owns /clock, and removes what a departed env left in its world."""

    CLOCK_TIMEOUT = 5.0

    def __init__(self, physics_dt: float, node: ArenaMixinNode) -> None:
        self._dt = physics_dt
        self._node = node
        self._control = node.create_client_wrapper(ControlWorld, f'{SERVER}/control_world', qos_profile=SERVICE_QOS)
        self._delete_prefix = node.create_client_wrapper(DeletePrefix, f'{SERVER}/delete_prefix', qos_profile=SERVICE_QOS)

    async def ensure_ready(self) -> None:
        await asyncio.gather(self._control.ensure(), self._delete_prefix.ensure())

    async def _send(self, pause: bool, steps: int = 0) -> Time:
        response = await self._control.call_timeout(ControlWorld.Request(pause=pause, multi_step=steps))
        if response is None:
            raise SimUnavailable(f'flatsim: no answer from {SERVER}/control_world')
        if not response.success:
            raise RuntimeError(f'flatsim: control_world(pause={pause}, multi_step={steps}) refused')
        return Time.from_msg(response.time)

    async def pause(self) -> bool:
        await self._send(True)
        return True

    async def unpause(self) -> bool:
        await self._send(False)
        return True

    async def step_seconds(self, seconds: float) -> float:
        """Take whole physics steps on the paused server, returns once /clock carries the last one."""
        n = round(seconds / self._dt)
        if seconds > 0:
            n = max(1, n)
        if n <= 0:
            return 0.0
        chunk = max(1, round(1.0 / self._dt))
        reached = self._node.sim_time
        for start in range(0, n, chunk):
            reached = await self._send(True, min(chunk, n - start))
        if not await self._node.await_sim_time(reached - Time.from_float(_EPS), freeze_timeout=self.CLOCK_TIMEOUT):
            raise SimUnavailable(f'flatsim: /clock silent for {self.CLOCK_TIMEOUT}s after a step')
        return n * self._dt

    def env_prefix(self, env_id: int) -> str:
        return env_prefix(env_id)

    async def cleanup_namespace(self, prefix: str) -> int:
        response = await self._delete_prefix.call_timeout(DeletePrefix.Request(prefix=prefix))
        if response is None:
            raise SimUnavailable(f'flatsim: no answer from {SERVER}/delete_prefix')
        return response.removed
