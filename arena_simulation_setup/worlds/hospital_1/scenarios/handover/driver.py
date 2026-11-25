"""Publish the receiver phases for the hospital_1 handover scenario."""

from __future__ import annotations

from functools import lru_cache
from math import dist
from pathlib import Path

import numpy as np
import yaml

from arena_runtime.humans.handover import PHASE_TIMEOUT_S
from arena_runtime.humans.pointing import skeleton as human_fk
from arena_runtime.humans.support import PALM_CENTER_FRACTION, TYPICAL_BODY, load_body, skeleton
from arena_simulation_setup.shared.body import body_to_semantic, load_stance, quat_to_matrix

SCENARIO = Path(__file__).with_name("scenario.yaml")
TAKE_DISTANCE_M = 0.095
MAX_SAMPLE_AGE_S = 0.35


def xyz(point):
    return np.array([point.x, point.y, point.z])


def rotation(quaternion):
    return quat_to_matrix((quaternion.x, quaternion.y, quaternion.z, quaternion.w))


@lru_cache(maxsize=4)
def receiver_rig(model_uri):
    body = load_body(model_uri) or TYPICAL_BODY
    return skeleton(body.legs, body.proportions), PALM_CENTER_FRACTION * body.proportions.hand, load_stance(model_uri)


def right_palm(model_uri, rotations, root_offset):
    """Recover the right palm in body coordinates using Arena's measured skeleton."""
    body, palm_length, stance = receiver_rig(model_uri)
    angles = body_to_semantic(rotations, stance)
    points, frames = human_fk.fk(angles, body)
    return points["r_wrist"] + palm_length * (frames["r_hand"] @ human_fk.DOWN) + root_offset


class Driver:
    """One attempt, driven by fresh samples in the pedestrian map frame."""

    def __init__(self):
        self.phase = "approach"
        self.since = 0.0
        self.last_sample = None
        self.close_since = None
        self.timed_out = False

    def advance(self, now, tool, palm, goal):
        """Return the new phase only when it changes."""
        planar = dist(tool[:2], palm[:2])
        gap = dist(tool, palm)
        to_goal = dist(tool, goal)
        elapsed = now - self.since
        if self.last_sample is not None and now - self.last_sample > MAX_SAMPLE_AGE_S:
            self.close_since = None
        self.last_sample = now

        phase = self.phase
        if self.phase in ("attend", "ready", "reach") and elapsed >= PHASE_TIMEOUT_S[self.phase]:
            self.timed_out = True
            phase = "retract"
        elif self.phase == "approach" and planar <= 1.2:
            phase = "attend"
        elif self.phase == "attend" and elapsed >= 0.55 and planar <= 0.64:
            phase = "ready"
        elif self.phase == "ready" and elapsed >= 0.65 and gap <= 0.55 and to_goal <= 0.24:
            phase = "reach"
        elif self.phase == "reach":
            if gap <= TAKE_DISTANCE_M and to_goal <= 0.055:
                if self.close_since is None:
                    self.close_since = now
                elif now - self.close_since >= 0.2:
                    phase = "take"
            else:
                self.close_since = None
        elif self.phase == "take" and ((elapsed >= 0.55 and gap >= 0.13) or elapsed >= 1.85):
            phase = "retract"

        if phase != self.phase:
            self.phase, self.since = phase, now
            return phase
        return None


def main():
    import rclpy
    from arena_people_msgs.msg import Handover, Pedestrians
    from geometry_msgs.msg import Point
    from rclpy.node import Node
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformException, TransformListener

    scenario = yaml.safe_load(SCENARIO.read_text())
    receiver = scenario["humans"][0]["name"]
    reach = scenario["robots"][0]["phases"][1]
    target_in_base = np.array(reach["reach"][:3])

    class HandoverNode(Node):
        def __init__(self):
            super().__init__("handover_driver")
            prefix = self.get_namespace().rsplit("/", 1)[-1] or "env_0"
            self.base_frame = self.declare_parameter("base_frame", f'{prefix}/jackal/{reach["frame"]}').value
            self.tool_frame = self.declare_parameter("tool_frame", f"{prefix}/jackal/top_tool0").value
            self.buffer = Buffer(node=self)
            self.listener = TransformListener(self.buffer, self)
            self.publisher = self.create_publisher(Handover, "human/handover", 10)
            self.create_subscription(Pedestrians, "arena_peds", self.on_peds, 10)
            self.driver = Driver()
            self.target = None
            self.sample_stamp = None
            self.get_logger().info(f"Waiting for {receiver} and the {self.tool_frame} TF")

        def on_peds(self, msg):
            if self.driver.phase == "retract":
                return
            ped = next((ped for ped in msg.pedestrians if ped.name.rsplit("/", 1)[-1] == receiver), None)
            frame = msg.header.frame_id
            if not frame or ped is None or "right_wrist" not in ped.body.joints:
                return
            now = self.get_clock().now()
            ped_time = Time.from_msg(msg.header.stamp, clock_type=now.clock_type)
            if abs((now - ped_time).nanoseconds) > MAX_SAMPLE_AGE_S * 1e9:
                return
            try:
                tool = self.buffer.lookup_transform(frame, self.tool_frame, Time())
                tool_time = Time.from_msg(tool.header.stamp, clock_type=now.clock_type)
                if abs((now - tool_time).nanoseconds) > MAX_SAMPLE_AGE_S * 1e9:
                    return
                sample_stamp = min(ped_time.nanoseconds, tool_time.nanoseconds)
                if self.sample_stamp is not None and sample_stamp <= self.sample_stamp:
                    return
                if self.target is None:
                    base = self.buffer.lookup_transform(frame, self.base_frame, tool_time).transform
                    goal = xyz(base.translation) + rotation(base.rotation) @ target_in_base
                else:
                    goal = self.target
            except TransformException:
                return

            rotations = {
                joint: (q.x, q.y, q.z, q.w)
                for joint, q in zip(ped.body.joints, ped.body.rotations, strict=True)
            }
            local = right_palm(ped.model_uri, rotations, xyz(ped.body.root_offset))
            palm = xyz(ped.pose.position) + rotation(ped.pose.orientation) @ local
            tip = xyz(tool.transform.translation)
            self.sample_stamp = sample_stamp
            phase = self.driver.advance(sample_stamp * 1e-9, tip, palm, goal)
            if phase is None:
                return
            if phase == "reach":
                self.target = goal
            self.publisher.publish(Handover(
                pedestrian_name=ped.name, phase=phase, at=Point(x=float(goal[0]), y=float(goal[1]), z=float(goal[2])),
                object_kind="bottle",
            ))
            reason = " (phase timed out)" if self.driver.timed_out else ""
            self.get_logger().info(f"{receiver}: {phase}, tool to palm {dist(tip, palm):.3f} m{reason}")

    rclpy.init()
    node = HandoverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
