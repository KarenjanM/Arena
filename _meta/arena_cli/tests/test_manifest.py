"""Tests for the installed-only completion values in arena_cli.manifest."""

from pathlib import Path

from arena_cli import manifest

ROBOT_GITMODULES = """\
[submodule "arena_robots/robots/mpo700/meshes"]
\tpath = arena_robots/robots/mpo700/meshes
\turl = https://example.invalid/mpo700.git
\trobot = mpo700
[submodule "arena_robots/robots/jackal/meshes"]
\tpath = arena_robots/robots/jackal/meshes
\turl = https://example.invalid/jackal.git
\trobot = jackal
"""

PLANNER_GITMODULES = """\
[submodule "planners/drlvo"]
\tpath = planners/drlvo
\turl = https://example.invalid/drlvo.git
\tplanner = drlvo
[submodule "planners/navid"]
\tpath = planners/navid
\turl = https://example.invalid/navid.git
\tplanner = navid
"""


def _arena(tmp_path: Path) -> Path:
    for sdk, body in (("arena_robots", ROBOT_GITMODULES), ("arena_planners", PLANNER_GITMODULES)):
        (tmp_path / sdk).mkdir()
        (tmp_path / sdk / ".gitmodules").write_text(body)
    for path in ("arena_robots/arena_robots/robots/jackal/meshes", "arena_planners/planners/drlvo"):
        (tmp_path / path).mkdir(parents=True)
        (tmp_path / path / ".git").write_text("gitdir: elsewhere\n")
    return tmp_path


def test_uninstalled_robots_are_the_ones_not_checked_out(tmp_path):
    assert manifest._uninstalled_robots(_arena(tmp_path)) == {"mpo700"}


def test_installed_planners_are_only_the_checked_out_ones(tmp_path):
    assert manifest._installed_planners(_arena(tmp_path)) == ["drlvo"]
