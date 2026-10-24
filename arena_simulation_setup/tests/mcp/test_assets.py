from __future__ import annotations

import os

import pytest

pytest.importorskip("arena_mcp")

from arena_mcp.core import ToolError

from arena_simulation_setup import ASS_DIR
from arena_simulation_setup.mcp import assets


sourced = pytest.mark.skipif(not os.environ.get("ARENA_WS_DIR"), reason="needs a sourced Arena workspace")


@sourced
def test_kinds_come_from_the_cli_table():
    kinds = assets.assets_kinds()
    assert {"object", "material", "world", "suite", "contest", "manifest"} <= set(kinds)
    assert kinds["world"] == "worlds"


@sourced
def test_ls_lists_installed_worlds_and_filters_by_glob():
    listed = assets.assets_ls("world", pattern="map_*")
    assert {"name": "map_empty"} in listed["assets"]
    assert all(row["name"].startswith("map_") for row in listed["assets"])
    assert listed["count"] == len(listed["assets"])


@sourced
def test_find_reports_the_winning_resolver_of_a_world():
    found = assets.assets_find("world", "sbend_corridor")
    assert found["resolves"]
    hit = next(verdict for verdict in found["verdicts"] if verdict["verdict"] == "hit")
    assert hit["path"] == str(ASS_DIR / "worlds" / "sbend_corridor")


@sourced
def test_find_of_an_unknown_kind_fails_with_the_cli_message():
    with pytest.raises(ToolError, match="unknown kind"):
        assets.assets_find("spaceship", "x")


@sourced
def test_pull_of_a_world_absent_from_the_bucket_fails():
    with pytest.raises(ToolError, match="was not pulled"):
        assets.assets_pull("world", "mcp_no_such_world_anywhere")


def test_domain_serves_only_the_arena_asset_verbs():
    assert sorted(spec.name for spec in assets.domain.tools) == ["assets_find", "assets_kinds", "assets_ls", "assets_pull", "assets_push"]
