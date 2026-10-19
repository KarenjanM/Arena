from collections.abc import Callable

import attrs

from task_generator.manager.environment_manager import EnvironmentManager
from task_generator.manager.robot_manager import RobotManager, RobotsManager
from task_generator.manager.world_manager.world_manager_ros import WorldManagerROS as WorldManager


@attrs.define
class TaskContext:
    environment_manager: EnvironmentManager
    robots_manager: RobotsManager
    world_manager: WorldManager
    abort_episode: Callable[[str], None]

    @property
    def robots(self) -> dict[str, RobotManager]:
        return self.robots_manager.managers
