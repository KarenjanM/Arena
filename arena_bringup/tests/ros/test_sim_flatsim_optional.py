"""sim.launch.py without the flatsim package: other sims launch and sim:=flatsim names the feature."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

_DRIVER = r"""
import json
import os
import sys

import launch
import launch.actions
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource

ctx = launch.LaunchContext()
ctx.launch_configurations.update(json.loads(sys.argv[1]))
path = os.path.join(get_package_share_directory("arena_bringup"), "launch", "simulator", "sim", "sim.launch.py")
selected = []
for entity in PythonLaunchDescriptionSource(path).get_launch_description(ctx).entities:
    if isinstance(entity, launch.actions.DeclareLaunchArgument):
        entity.execute(ctx)
    else:
        selected = entity.execute(ctx) or []
for action in selected:
    if isinstance(action, launch.actions.OpaqueFunction):
        action.execute(ctx)
print(json.dumps([type(action).__name__ for action in selected]))
"""

_STRIPPED = ("flatsim", "flatsim_msgs")


def _launch(sim: str, *, with_flatsim: bool) -> subprocess.CompletedProcess[str]:
    if "AMENT_PREFIX_PATH" not in os.environ:
        pytest.skip("AMENT_PREFIX_PATH unset: source install/setup.bash")
    env = dict(os.environ)
    if not with_flatsim:
        env["AMENT_PREFIX_PATH"] = os.pathsep.join(p for p in os.environ["AMENT_PREFIX_PATH"].split(os.pathsep) if os.path.basename(p) not in _STRIPPED)
    args = {"sim": sim, "use_sim_time": "true", "world": "map_empty"}
    return subprocess.run([sys.executable, "-c", _DRIVER, json.dumps(args)], env=env, capture_output=True, text=True, timeout=120, check=False)


def test_dummy_launches_without_flatsim() -> None:
    proc = _launch("dummy", with_flatsim=False)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == ["GroupAction"]


def test_flatsim_without_its_package_names_the_feature() -> None:
    proc = _launch("flatsim", with_flatsim=False)
    assert proc.returncode != 0
    assert "arena feature flatsim install" in proc.stderr


def test_flatsim_starts_the_server_through_the_feature() -> None:
    from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

    try:
        get_package_share_directory("flatsim")
    except PackageNotFoundError:
        pytest.skip("flatsim feature not installed")
    proc = _launch("flatsim", with_flatsim=True)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == ["OpaqueFunction", "ExecuteProcess"]
