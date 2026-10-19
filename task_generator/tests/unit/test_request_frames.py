from __future__ import annotations

import types

import pytest

geometry_msgs = pytest.importorskip("geometry_msgs.msg")

from task_generator.manager.realizer import Realizer
from task_generator.node import TaskGenerator, _is_map_frame


def _stamped(frame: str, x: float, y: float) -> object:
    msg = geometry_msgs.PoseStamped()
    msg.header.frame_id = frame
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.orientation.w = 1.0
    return msg


@pytest.mark.parametrize(("frame", "accepted"), [("", True), ("map", True), ("env_0/map", True), ("odom", False), ("env_0/jackal/base_link", False), ("mapper", False)])
def test_map_frames_are_the_plain_and_the_env_namespaced_map(frame, accepted):
    assert _is_map_frame(frame) is accepted


def test_spawn_poses_are_shifted_from_the_map_frame_into_the_world_frame():
    holder = types.SimpleNamespace(_realizer=Realizer(Realizer._Configuration(x=5.0, y=7.0, prefix="env_0")))
    pose = TaskGenerator._pose_from_request(holder, _stamped("env_0/map", 6.0, 10.0))
    assert (pose.position.x, pose.position.y) == pytest.approx((1.0, 3.0))
    with pytest.raises(ValueError, match="map frame"):
        TaskGenerator._pose_from_request(holder, _stamped("odom", 6.0, 10.0))
