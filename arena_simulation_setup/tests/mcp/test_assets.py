from __future__ import annotations

import importlib.util

import pytest

pytest.importorskip("arena_mcp")

from arena_mcp.core import ToolError

from arena_simulation_setup import ASS_DIR
from arena_simulation_setup.mcp import assets


def _has_importers() -> bool:
    try:
        return importlib.util.find_spec("arena_assets.importers") is not None
    except ModuleNotFoundError:
        return False


def test_kinds_come_from_the_cli_table():
    kinds = assets.assets_kinds()
    assert {"object", "material", "world", "suite", "contest", "manifest"} <= set(kinds)
    assert kinds["world"] == "worlds"


def test_ls_lists_installed_worlds_and_filters_by_glob():
    listed = assets.assets_ls("world", pattern="map_*")
    assert {"name": "map_empty"} in listed["assets"]
    assert all(row["name"].startswith("map_") for row in listed["assets"])
    assert listed["count"] == len(listed["assets"])


def test_find_reports_the_winning_resolver_of_a_world():
    found = assets.assets_find("world", "sbend_corridor")
    assert found["resolves"]
    hit = next(verdict for verdict in found["verdicts"] if verdict["verdict"] == "hit")
    assert hit["path"] == str(ASS_DIR / "worlds" / "sbend_corridor")


def test_find_of_an_unknown_kind_fails_with_the_cli_message():
    with pytest.raises(ToolError, match="unknown kind"):
        assets.assets_find("spaceship", "x")


def test_pull_of_a_world_absent_from_the_bucket_fails():
    with pytest.raises(ToolError, match="was not pulled"):
        assets.assets_pull("world", "mcp_no_such_world_anywhere")


@pytest.mark.skipif(_has_importers(), reason="arena_assets.importers is installed")
def test_import_tools_fail_with_a_hint_without_the_registry():
    with pytest.raises(ToolError, match="not installed") as raised:
        assets.assets_import_sources()
    assert raised.value.hint
    with pytest.raises(ToolError, match="not installed"):
        assets.assets_import_items("world", "barn")


@pytest.mark.skipif(not _has_importers(), reason="arena_assets.importers is not installed")
def test_import_sources_follow_the_registry():
    from arena_assets.importers import IMPORTERS

    rows = assets.assets_import_sources()
    assert {(row["kind"], row["source"]) for row in rows} == {(meta.kind, meta.name) for meta in IMPORTERS}
    assert all(row["kind"] == "world" for row in assets.assets_import_sources(kind="world"))
    with pytest.raises(ToolError, match="no importer kind"):
        assets.assets_import_items("spaceship", "x")
