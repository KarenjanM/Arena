import enum
import uuid
from collections.abc import Sequence

from arena_simulation_setup.shared import Light

from task_generator.shared import DynamicObstacle, Obstacle, Pose, Region
from task_generator.tasks.mode import TaskMode
from task_generator.tasks.obstacles._placement import random_placement

from . import edge_case, environment, parametrized, prompt, random, scenario

Obstacles = tuple[list[Obstacle], list[DynamicObstacle]]


@enum.unique
class ObstacleKind(enum.Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"


class TM_Obstacles(TaskMode):
    lights: Sequence[Light] = ()

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.pending_regions: list[Region] = []

    async def reset(self, *, seed: int) -> Obstacles:
        return [], []

    async def teardown(self) -> None:
        """Release everything this mode drives; called when the mode is replaced or the task ends."""

    async def extend(self, kind: ObstacleKind, model: str, pose: Pose | None = None, level: str = "") -> Obstacle | DynamicObstacle:
        resolved_pose = pose if pose is not None else await random_placement(self._ctx, level_id=level)
        name = f"ext_{model}_{uuid.uuid4().hex[:6]}"

        if kind is ObstacleKind.STATIC:
            obstacle: Obstacle | DynamicObstacle = Obstacle(name=name, model=model, pose=resolved_pose, level=level)
            await self._ctx.environment_manager.spawn_obstacles([obstacle])
        else:
            waypoints = self._ctx.world_manager.get_positions_on_map(n=2, safe_dist=0, forbid=False, level_id=level)
            obstacle = DynamicObstacle(name=name, model=model, waypoints=waypoints, pose=resolved_pose, level=level)
            await self._ctx.environment_manager.spawn_dynamic_obstacles([obstacle])
        return obstacle

    async def retract(self, entity_id: str) -> bool:
        removed, _ = await self._ctx.environment_manager.remove_obstacles_by_id([entity_id])
        return bool(removed)


__all__ = ["TM_Obstacles", "ObstacleKind", "Obstacles", "edge_case", "environment", "parametrized", "prompt", "random", "scenario"]
