import typing

from task_generator.tasks.mode import TaskMode, TaskModeKwargs

from . import clear_forbidden_zones, rviz_ui, sounds, zone_edit

if typing.TYPE_CHECKING:
    from task_generator.tasks.task import Task


class TM_ModuleKwargs(TaskModeKwargs):
    task: "Task"


class TM_Module(TaskMode):
    _task: "Task"

    def __init__(self, *, task: "Task", **kwargs: typing.Unpack[TaskModeKwargs]) -> None:
        super().__init__(**kwargs)
        self._task = task

    def before_reset(self): ...

    def after_reset(self): ...


__all__ = [
    "TM_Module",
    "clear_forbidden_zones",
    "rviz_ui",
    "sounds",
    "zone_edit",
]
