"""In-process launches keep their environment actions off the process environment and still hand them to their children."""

from __future__ import annotations

import asyncio
import os
import pathlib

import launch
from launch.actions import ExecuteProcess, GroupAction, OpaqueFunction, ResetEnvironment, SetEnvironmentVariable, UnsetEnvironmentVariable

from arena_rclpy_mixins.Async import AsyncLaunchManager

_VAR = "ARENA_TEST_LAUNCH_ENVIRONMENT"


def _run(description: launch.LaunchDescription) -> None:
    async def main() -> None:
        await (await AsyncLaunchManager().launch_description(description))

    asyncio.run(main())


def _record(path: pathlib.Path) -> ExecuteProcess:
    return ExecuteProcess(cmd=["sh", "-c", f'printf %s "${{{_VAR}-unset}}" > {path}'])


def test_set_and_scoped_groups_leave_os_environ_untouched(tmp_path: pathlib.Path):
    before = dict(os.environ)
    _run(
        launch.LaunchDescription(
            [
                SetEnvironmentVariable(_VAR, "outer"),
                GroupAction([SetEnvironmentVariable(_VAR, "inner"), _record(tmp_path / "inner")], scoped=True),
                _record(tmp_path / "outer"),
            ]
        )
    )
    assert dict(os.environ) == before
    assert (tmp_path / "inner").read_text() == "inner"
    assert (tmp_path / "outer").read_text() == "outer"


def test_unset_and_reset_act_on_the_launch_environment_only(tmp_path: pathlib.Path):
    seen: list[str | None] = []
    os.environ[_VAR] = "process"
    try:
        _run(
            launch.LaunchDescription(
                [
                    UnsetEnvironmentVariable(_VAR),
                    OpaqueFunction(function=lambda _context: seen.append(os.environ.get(_VAR))),
                    _record(tmp_path / "unset"),
                    ResetEnvironment(),
                    _record(tmp_path / "reset"),
                ]
            )
        )
    finally:
        del os.environ[_VAR]
    assert seen == ["process"]
    assert (tmp_path / "unset").read_text() == "unset"
    assert (tmp_path / "reset").read_text() == "process"
