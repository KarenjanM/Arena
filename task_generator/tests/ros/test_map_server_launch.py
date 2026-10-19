"""The map_server bringup lands in the task generator namespace, also when relaunched."""

from __future__ import annotations

import asyncio
import os
import threading
import time

import pytest


@pytest.fixture(scope="module", autouse=True)
def _require_ament() -> None:
    if "AMENT_PREFIX_PATH" not in os.environ:
        pytest.skip("AMENT_PREFIX_PATH unset: source install/setup.bash")


def _node_names(description) -> set[str]:
    import launch
    from launch_ros.actions import Node

    ctx = launch.LaunchContext()
    names: set[str] = set()

    def walk(entity) -> None:
        if isinstance(entity, Node):
            entity._perform_substitutions(ctx)
            names.add(entity.node_name)
        elif isinstance(entity, launch.LaunchDescription):
            for child in entity.entities:
                walk(child)
        elif isinstance(entity, launch.Action):
            for child in entity.execute(ctx) or []:
                walk(child)

    walk(description)
    return names


def test_map_server_bringup_is_under_the_namespace() -> None:
    from task_generator.manager.world_manager.world_manager_ros import map_server_launch_description

    assert _node_names(map_server_launch_description("/arena/env_3/task_generator_node")) == {
        "/arena/env_3/task_generator_node/map_server",
        "/arena/env_3/task_generator_node/lifecycle_manager_map_server",
    }


def test_relaunch_replaces_the_bringup_in_the_namespace() -> None:
    import lifecycle_msgs.msg
    import rclpy
    from arena_rclpy_mixins.LifecycleClient import AsyncLifecycleClient
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor

    from task_generator.manager.world_manager.world_manager_ros import MapServerHandler

    namespace = f"/map_server_relaunch_{os.getpid()}"
    context = Context()
    rclpy.init(context=context)

    async def main() -> tuple[bool, bool, bool, list[tuple[str, str]]]:
        node = AsyncLifecycleClient("probe", namespace=namespace, context=context)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        threading.Thread(target=executor.spin, daemon=True).start()
        handler = MapServerHandler(node=node)
        map_server = f"{namespace}/probe/map_server"
        active = lifecycle_msgs.msg.State.PRIMARY_STATE_ACTIVE
        try:
            await handler.launch_map_server()
            first = handler._map_server_launch
            assert first is not None
            first_active = await node.wait_for_lifecycle_state_async(map_server, active, timeout=90.0)
            await handler.launch_map_server()
            first_gone = first.task.done()
            relaunch_active = await node.wait_for_lifecycle_state_async(map_server, active, timeout=90.0)
            deadline = time.monotonic() + 10.0
            while True:
                map_servers = [(name, ns) for name, ns in node.get_node_names_and_namespaces() if name == "map_server" and ns in ("/", f"{namespace}/probe")]
                if len(map_servers) == 1 or time.monotonic() > deadline:
                    break
                await asyncio.sleep(0.5)
            return first_active, first_gone, relaunch_active, map_servers
        finally:
            await node.kill_launches()
            executor.shutdown()
            node.destroy_node()

    try:
        first_active, first_gone, relaunch_active, map_servers = asyncio.run(main())
    finally:
        rclpy.shutdown(context=context)

    assert first_active
    assert first_gone
    assert relaunch_active
    assert map_servers == [("map_server", f"{namespace}/probe")]
