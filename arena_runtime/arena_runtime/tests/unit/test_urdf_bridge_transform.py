from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from arena_runtime.sim._urdf import transform_urdf_for_bridge

BRIDGE = 'joint_state_topic_hardware_interface/JointStateTopicSystem'
TOPICS = {'velocity': '/r/cmd_vel_joints', 'position': '/r/cmd_pos_joints'}
STATES = '/r/isaac_joint_states'

GZ_URDF = """<?xml version="1.0"?>
<robot name="r">
  <link name="base"/>
  <ros2_control name="GazeboSystem" type="system">
    <hardware>
      <plugin> gz_ros2_control/GazeboSimSystem </plugin>
    </hardware>
    <joint name="wheel_left">
      <command_interface name="velocity"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
    <joint name="arm">
      <command_interface name="position"/>
      <state_interface name="position"><param name="initial_value">0.5</param></state_interface>
      <state_interface name="effort"/>
    </joint>
    <joint name="caster">
      <state_interface name="position"/>
    </joint>
    <joint name="gripper">
      <command_interface name="effort"/>
      <state_interface name="position"/>
    </joint>
    <joint name="wheel_right">
      <command_interface name="velocity"/>
      <state_interface name="velocity"/>
    </joint>
  </ros2_control>
  <gazebo>
    <plugin filename="gz_ros2_control-system" name="gz_ros2_control::GazeboSimROS2ControlPlugin"><parameters>/c.yaml</parameters></plugin>
    <plugin/>
  </gazebo>
  <ros2_control name="generic" type="actuator">
    <hardware>
      <plugin>topic_based_ros2_control/TopicBasedSystem</plugin>
    </hardware>
    <joint name="lift"><command_interface name="position"/></joint>
  </ros2_control>
</robot>
"""


def _transform(tmp_path: Path, urdf: str, topics: dict[str, str] = TOPICS) -> ET.Element:
    src = tmp_path / 'in.urdf'
    dst = tmp_path / 'out.urdf'
    src.write_text(urdf)
    transform_urdf_for_bridge(str(src), topics, STATES, str(dst))
    return ET.parse(dst).getroot()


def _params(block: ET.Element) -> dict[str, str]:
    return {p.get('name'): p.text for p in block.find('hardware').findall('param')}


def _initial_values(joint: ET.Element) -> dict[str, list[str]]:
    return {si.get('name'): [p.text for p in si.findall('param[@name="initial_value"]')] for si in joint.findall('state_interface')}


def test_gazebo_block_splits_into_one_bridge_block_per_command_kind_in_place(tmp_path):
    root = _transform(tmp_path, GZ_URDF)
    assert [(c.tag, c.get('name')) for c in root] == [
        ('link', 'base'),
        ('ros2_control', 'GazeboSystem_velocity'),
        ('ros2_control', 'GazeboSystem_position'),
        ('gazebo', None),
        ('ros2_control', 'generic'),
    ]
    velocity, position = root[1], root[2]
    assert velocity.get('type') == position.get('type') == 'system'
    assert velocity.find('hardware/plugin').text == position.find('hardware/plugin').text == BRIDGE
    assert [j.get('name') for j in velocity.findall('joint')] == ['wheel_left', 'wheel_right']
    assert [j.get('name') for j in position.findall('joint')] == ['arm']


def test_each_bridge_block_publishes_its_own_command_topic_without_threshold(tmp_path):
    root = _transform(tmp_path, GZ_URDF)
    assert _params(root[1]) == {
        'joint_commands_topic': TOPICS['velocity'],
        'joint_states_topic': STATES,
        'trigger_joint_command_threshold': '0.0',
        'sum_wrapped_joint_states': 'true',
    }
    assert _params(root[2]) == {
        'joint_commands_topic': TOPICS['position'],
        'joint_states_topic': STATES,
        'trigger_joint_command_threshold': '0.0',
    }


def test_joints_without_a_bridged_command_kind_are_dropped(tmp_path):
    root = _transform(tmp_path, GZ_URDF)
    bridged = {j.get('name') for rc in root.findall('ros2_control') if rc.get('name') != 'generic' for j in rc.findall('joint')}
    assert bridged == {'wheel_left', 'wheel_right', 'arm'}


def test_every_bridged_joint_gets_one_effort_state_and_zero_initial_values(tmp_path):
    root = _transform(tmp_path, GZ_URDF)
    wheel_left, wheel_right = root[1].findall('joint')
    (arm,) = root[2].findall('joint')
    assert _initial_values(wheel_left) == {'position': ['0.0'], 'velocity': ['0.0'], 'effort': ['0.0']}
    assert _initial_values(wheel_right) == {'velocity': ['0.0'], 'effort': ['0.0']}
    assert [si.get('name') for si in arm.findall('state_interface')] == ['position', 'effort']
    assert _initial_values(arm) == {'position': ['0.5'], 'effort': ['0.0']}


def test_non_gazebo_plugins_and_blocks_are_left_untouched(tmp_path):
    root = _transform(tmp_path, GZ_URDF)
    gz_plugin, empty_plugin = root.find('gazebo').findall('plugin')
    assert gz_plugin.get('filename') == 'gz_ros2_control-system'
    assert gz_plugin.find('parameters').text == '/c.yaml'
    assert empty_plugin.text is None
    generic = root.find('ros2_control[@name="generic"]')
    assert generic.find('hardware/plugin').text == 'topic_based_ros2_control/TopicBasedSystem'
    assert generic.find('hardware/param') is None
    assert generic.find('joint/state_interface') is None


def test_output_carries_an_xml_declaration(tmp_path):
    _transform(tmp_path, GZ_URDF)
    assert (tmp_path / 'out.urdf').read_bytes().startswith(b"<?xml version='1.0' encoding='utf-8'?>")


def test_unnamed_block_defaults_to_bridge_system_naming(tmp_path):
    urdf = f"""<robot name="r">
  <ros2_control>
    <hardware><plugin>{BRIDGE}</plugin></hardware>
    <joint name="j"><command_interface name="position"/><state_interface name="position"/></joint>
  </ros2_control>
</robot>"""
    root = _transform(tmp_path, urdf)
    (block,) = root.findall('ros2_control')
    assert (block.get('name'), block.get('type')) == ('bridge_position', 'system')
    assert _params(block)['joint_commands_topic'] == TOPICS['position']


def test_nested_block_is_replaced_inside_its_own_parent(tmp_path):
    urdf = """<robot name="r">
  <xacro_scope>
    <first/>
    <ros2_control name="arm" type="system">
      <hardware><plugin>gz_ros2_control/GazeboSimSystem</plugin></hardware>
      <joint name="j"><command_interface name="position"/></joint>
    </ros2_control>
    <last/>
  </xacro_scope>
</robot>"""
    root = _transform(tmp_path, urdf)
    assert root.find('ros2_control') is None
    assert [c.tag for c in root.find('xacro_scope')] == ['first', 'ros2_control', 'last']
    assert root.find('xacro_scope/ros2_control').get('name') == 'arm_position'


def test_block_without_bridged_joints_is_removed(tmp_path):
    urdf = """<robot name="r">
  <ros2_control name="passive" type="system">
    <hardware><plugin>gz_ros2_control/GazeboSimSystem</plugin></hardware>
    <joint name="j"><state_interface name="position"/></joint>
  </ros2_control>
</robot>"""
    root = _transform(tmp_path, urdf)
    assert root.findall('ros2_control') == []


@pytest.mark.parametrize(
    'hardware',
    ['', '<hardware/>', '<hardware><plugin/></hardware>', '<hardware><plugin>other/System</plugin></hardware>'],
)
def test_blocks_without_the_gazebo_plugin_are_not_split(tmp_path, hardware):
    urdf = f"""<robot name="r">
  <ros2_control name="x" type="system">
    {hardware}
    <joint name="j"><command_interface name="position"/></joint>
  </ros2_control>
</robot>"""
    root = _transform(tmp_path, urdf)
    (block,) = root.findall('ros2_control')
    assert block.get('name') == 'x'
    assert block.find('joint/state_interface') is None
