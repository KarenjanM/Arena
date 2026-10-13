from __future__ import annotations

import typing

import attrs
from geometry_msgs.msg import Pose as PoseMsg
from visualization_msgs.msg import Marker

from task_generator.interactive.hub import Apply, planar_marker, visual
from task_generator.shared import Obstacle, Pose

if typing.TYPE_CHECKING:
    from task_generator.node import TaskGenerator

_STATIC_COLOR = (0.2, 0.6, 0.9, 0.8)


class EntityHandles:
    """Interactive handles of runtime-spawned static obstacles, named by sim_path."""

    def __init__(self, node: TaskGenerator) -> None:
        self._node = node
        self._obstacles: dict[str, Obstacle] = {}

    def add(self, obstacle: Obstacle) -> None:
        name = obstacle.sim_path
        self._obstacles[name] = obstacle
        pose = self._node._realizer.realize(obstacle.pose, obstacle.level).to_msg()
        body = visual(Marker.CYLINDER, (0.6, 0.6, 0.02), _STATIC_COLOR)

        async def on_pose(moved: PoseMsg) -> None:
            await self._move(name, moved)

        self._node.markers.put(
            planar_marker(name, pose, description=obstacle.name, visuals=[body]),
            on_pose=on_pose,
            apply=Apply.LIVE,
        )

    def clear(self) -> None:
        self._obstacles.clear()

    async def _move(self, name: str, pose_msg: PoseMsg) -> None:
        obstacle = self._obstacles[name]
        environment = self._node._environment_manager
        local = self._node._realizer.ezilear(Pose.from_msg(pose_msg), obstacle.level)
        moved = attrs.evolve(obstacle, pose=local)
        await environment.move_obstacles([moved])
        self._obstacles[name] = moved
