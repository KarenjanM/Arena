"""Tests for the `arena mcp` verb and the host launcher _meta/tools/arena-mcp."""

import importlib.util
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from arena_cli import cli
from arena_cli.complete import Context, _expand

META = Path(__file__).resolve().parents[2]
LAUNCHER = META / "tools" / "arena-mcp"
needs_arena_mcp = pytest.mark.skipif(importlib.util.find_spec("arena_mcp.server") is None, reason="arena_mcp is not installed in this environment")


def arena(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": f"{META}:{os.environ.get('PYTHONPATH', '')}", "ARENA_WS_DIR": str(tmp_path)}
    return subprocess.run([sys.executable, "-m", "arena_cli", *args], env=env, capture_output=True, text=True, check=False, timeout=120)


def launcher(*args: str, ws: str | None = None) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items() if key != "ARENA_WS_DIR"}
    if ws is not None:
        env["ARENA_WS_DIR"] = ws
    return subprocess.run(["bash", str(LAUNCHER), *args], env=env, capture_output=True, text=True, check=False, timeout=30)


def printed(result: subprocess.CompletedProcess[str]) -> dict[str, list[str]]:
    assert result.returncode == 0, result.stderr
    return {label: shlex.split(command) for label, _, command in (line.partition(": ") for line in result.stdout.splitlines())}


def candidates(words: list[str], cur: str = "") -> set[str]:
    return {value for value, _desc, _group in _expand(cli._VERBS["mcp"].complete, words, cur, Context("", {})).items}


def test_help_lists_the_forms_and_subcommands(tmp_path: Path) -> None:
    result = arena(tmp_path, "mcp", "--help")
    assert result.returncode == 0
    assert "arena mcp ls [DOMAIN ...] [--json]" in result.stdout
    assert "arena mcp serve [DOMAIN ...]" in result.stdout
    assert "_meta/tools/arena-mcp" in result.stdout
    assert "Commands:" in result.stdout and "serve" in result.stdout.split("Commands:")[1]


def test_subcommand_help_explains_it(tmp_path: Path) -> None:
    result = arena(tmp_path, "mcp", "serve", "--help")
    assert result.returncode == 0
    assert result.stdout.startswith("Usage: arena mcp serve")
    assert "Stdout carries only the protocol" in result.stdout


def test_bad_subcommand_lists_the_accepted_ones(tmp_path: Path) -> None:
    result = arena(tmp_path, "mcp", "start")
    assert result.returncode == 1
    assert "No such command 'start'. Commands: ls, serve" in result.stderr


def test_mcp_sits_in_the_attach_section(tmp_path: Path) -> None:
    result = arena(tmp_path, "--help")
    attach = result.stdout.split("Attach:")[1].split("\n\n")[0]
    assert "mcp" in attach and "Serve the Arena MCP domains" in attach


@needs_arena_mcp
def test_bare_call_shows_domains_and_the_registration_line(tmp_path: Path) -> None:
    result = arena(tmp_path, "mcp")
    assert result.returncode == 0, result.stderr
    assert any(line.split()[:1] == ["jobs"] and "served" in line for line in result.stdout.splitlines())
    assert "_meta/tools/arena-mcp serve [DOMAIN ...]" in result.stdout
    assert "python3 -m arena_mcp serve" in result.stdout


@needs_arena_mcp
def test_ls_of_an_unknown_domain_lists_the_installed_ones(tmp_path: Path) -> None:
    result = arena(tmp_path, "mcp", "ls", "no_such_domain")
    assert result.returncode != 0
    assert "unknown domain(s) no_such_domain, installed:" in result.stderr and "jobs" in result.stderr


@needs_arena_mcp
def test_completion_offers_subcommands_and_installed_domains() -> None:
    assert candidates([]) == {"ls", "serve"}
    assert "jobs" in candidates(["serve"])
    assert {"jobs", "--json"} <= candidates(["ls"])


def test_launcher_derives_the_dev_container_project_from_the_workspace() -> None:
    commands = printed(launcher("--print-command", ws="/somewhere/my_ws"))
    assert commands["lookup"] == ["docker", "ps", "-q", "--filter", "status=running", "--filter", "label=com.docker.compose.project=arena-my_ws", "--filter", "label=com.docker.compose.service=arena"]
    assert commands["probe"] == ["docker", "exec", "{container}", "test", "-f", "/.built"]
    exec_ = commands["exec"]
    assert exec_[:7] == ["docker", "exec", "-i", "{container}", "/entrypoint.sh", "bash", "--norc"]
    assert exec_[7] == "-c" and exec_[8].endswith('exec python3 -m arena_mcp "$@"')
    assert exec_[9:] == ["bash", "serve"]


def test_launcher_without_workspace_uses_the_checkout_layout() -> None:
    commands = printed(launcher("--print-command"))
    ws = LAUNCHER.resolve().parents[2].parent.parent
    expected = f"arena-{ws.name}" if LAUNCHER.resolve().parents[2].parent.name == "src" else "arena-arena_ws"
    assert f"label=com.docker.compose.project={expected}" in commands["lookup"]


def test_launcher_fork_over_ssh_quotes_the_remote_command() -> None:
    commands = printed(launcher("--fork", "mcp", "--host", "box", "--print-command", "ls", "system", "--json", ws="/w/arena_ws"))
    for label in ("lookup", "probe", "exec"):
        assert commands[label][:5] == ["ssh", "-T", "-o", "BatchMode=yes", "box"]
        assert len(commands[label]) == 6
    assert "label=com.docker.compose.project=arena-arena_ws-mcp" in shlex.split(commands["lookup"][5])
    remote = shlex.split(commands["exec"][5])
    assert remote[:4] == ["docker", "exec", "-i", "{container}"] and remote[-4:] == ["bash", "ls", "system", "--json"]


def test_launcher_with_an_explicit_container_skips_the_lookup() -> None:
    commands = printed(launcher("--container", "arena-x-arena-1", "--print-command", "serve", "ros", "jobs"))
    assert "lookup" not in commands
    assert commands["exec"][3] == "arena-x-arena-1" and commands["exec"][-3:] == ["serve", "ros", "jobs"]


def test_launcher_help_and_bad_arguments_go_to_stderr() -> None:
    shown = launcher("--help")
    assert shown.returncode == 0 and shown.stdout == ""
    assert "usage: arena-mcp" in shown.stderr and "source arena" in shown.stderr
    for args, message in ((["--bogus"], "unknown option '--bogus', options: --fork, --host, --container, --print-command, --help"), (["--fork"], "--fork needs a value"), (["--fork", "Bad", "ls"], "fork name 'Bad' must be")):
        result = launcher(*args)
        assert result.returncode == 2 and result.stdout == ""
        assert message in result.stderr


def test_launcher_is_executable() -> None:
    assert os.access(LAUNCHER, os.X_OK)
