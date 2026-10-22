import typing

import rclpy.node
from arena_rclpy_mixins.shared import Namespace
from arena_runtime._node import NodeInterface

from task_generator.tasks.context import TaskContext


class Namespaced:
    _namespace: Namespace = Namespace('').ParamNamespace()

    def namespace(self, *path: str) -> Namespace:
        return self._namespace(*path)


class TaskModeKwargs(typing.TypedDict):
    ctx: TaskContext
    namespace: Namespace
    node: rclpy.node.Node


class TaskMode(NodeInterface, Namespaced):
    _ctx: TaskContext

    def __init__(self, *, ctx: TaskContext, namespace: Namespace, node: rclpy.node.Node) -> None:
        super().__init__(node=node)
        self._ctx = ctx
        self._namespace = namespace
