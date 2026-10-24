from __future__ import annotations

import subprocess
import sys
import time
import uuid

import pytest

SERVER = """
import sys
import rclpy
from std_srvs.srv import Trigger
rclpy.init()
node = rclpy.create_node("supervisor_discovery_peer")
node.create_service(Trigger, sys.argv[1], lambda request, response: response)
print("READY", flush=True)
rclpy.spin(node)
"""


def test_supervisor_discovers_a_service_another_process_serves_and_gives_up_on_a_missing_one(rclpy_context):
    if rclpy_context is None:
        pytest.skip("ROS2 not available")
    from arena_bringup.supervisor import Supervisor

    service = f"/supervisor_probe_{uuid.uuid4().hex[:8]}/register_env"
    peer = subprocess.Popen([sys.executable, "-c", SERVER, service], stdout=subprocess.PIPE, text=True)
    node = rclpy_context.create_node(f"supervisor_probe_{uuid.uuid4().hex[:8]}")
    try:
        assert peer.stdout is not None
        assert peer.stdout.readline().strip() == "READY"
        sup = Supervisor(node)
        assert sup.discovers(service, 3.0, 15.0)
        started = time.monotonic()
        assert not sup.discovers(f"{service}_missing", 0.5, 15.0)
        assert time.monotonic() - started < 5.0
    finally:
        node.destroy_node()
        peer.terminate()
        peer.wait(timeout=10)
