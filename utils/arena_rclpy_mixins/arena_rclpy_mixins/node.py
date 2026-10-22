from collections.abc import Callable
from typing import TYPE_CHECKING, TypedDict

import rclpy
import rclpy.node

from .Async import AsyncNode
from .LifecycleClient import AsyncLifecycleClient, LifecycleClient
from .ROSParamServer import ROSParamServer
from .ServiceNamespace import ServiceNamespace
from .spin import run_main
from .Time import TimeNode

if TYPE_CHECKING:
    import aiomonitor


class AiomonitorConfig(TypedDict, total=False):
    monitor_cls: "type[aiomonitor.Monitor]"
    host: str
    port: int
    console_port: int
    webui_port: int
    console_enabled: bool
    hook_task_factory: bool
    max_termination_history: int | None


class ArenaMixinNode(ROSParamServer, LifecycleClient, AsyncLifecycleClient, ServiceNamespace, AsyncNode, TimeNode, rclpy.node.Node):
    """Megaclass composing every arena_rclpy_mixins mixin onto rclpy.node.Node."""

    _shutdown_requester: Callable[[str], None] | None = None

    async def setup(self) -> None:
        """Called by `async_main` once the node is constructed and the executor is spinning.
        Override for non-lifecycle subclasses; lifecycle subclasses leave this as a no-op
        and let the lifecycle state machine drive configure/activate.
        """

    async def teardown(self) -> None:
        """Called by `async_main` on SIGINT/SIGTERM before rclpy shuts down.
        Override to run cleanup that requires the executor still spinning,
        e.g. final service calls to peer nodes.
        """

    def request_shutdown(self, reason: str) -> None:
        """Start the `async_main` teardown from any thread, as SIGINT would, or shut rclpy down outside `async_main`."""
        if self._shutdown_requester is None:
            rclpy.try_shutdown()
            return
        self._shutdown_requester(reason)

    def aiomonitor_config(self) -> AiomonitorConfig | None:
        """Override to customize aiomonitor.start_monitor kwargs, or return None to disable.
        Only consulted when run_main(aiomonitor=True). Default: empty dict (library defaults).
        """
        return {}

    @classmethod
    def run_main(cls, *args: object, aiomonitor: bool = False, **kwargs: object) -> None:
        """Standard entry: instantiate, spin, tear down cleanly. Subclass needs an async `setup()`."""
        run_main(cls, *args, aiomonitor=aiomonitor, **kwargs)
