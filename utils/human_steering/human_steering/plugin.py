"""rqt Plugin shell: --ns arg, QTimer-driven tick, wires panel + driver.

50 ms QTimer calls panel.tick(), no rclpy timer touches Qt-adjacent state.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from python_qt_binding.QtCore import QTimer
from rclpy.parameter import Parameter
from rqt_gui_py.plugin import Plugin

from human_steering.driver import resolve_namespace
from human_steering.panel import Panel

if TYPE_CHECKING:
    from rqt_gui.ros2_plugin_context import Ros2PluginContext

TICK_MS = 50


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="human_steering")
    parser.add_argument("--ns", dest="ns", default=None, help="env or node namespace to attach to (see arena viz)")
    parser.add_argument("--unlimited", action="store_true", help="pose sliders run 0..2pi and the driver skips joint-limit clamping")
    return parser.parse_args(argv)


class HumanSteering(Plugin):
    """rqt_gui_py plugin entry point, registered in plugin.xml."""

    def __init__(self, context: Ros2PluginContext) -> None:
        super().__init__(context)
        self.setObjectName("HumanSteering")

        self._node = context.node
        # the stream gate compares sim-time stamps, follow /clock
        self._node.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        args = _parse_args(context.argv())
        self._target_ns = args.ns
        self._attached = False

        self._panel = Panel(unlimited=args.unlimited)
        self._panel.setObjectName("HumanSteeringUi")
        self._panel.set_instance_number(context.serial_number())
        context.add_widget(self._panel)

        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(TICK_MS)

    def _tick(self) -> None:
        if not self._attached:
            namespaces = resolve_namespace(self._node, self._target_ns)
            if namespaces is not None:
                self._panel.attach(self._node, namespaces)
                self._attached = True
        self._panel.tick()

    def shutdown_plugin(self) -> None:
        self._timer.stop()
        self._panel.detach()
