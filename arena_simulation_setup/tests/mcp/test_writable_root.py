from __future__ import annotations

import os

import yaml

from arena_simulation_setup import ASS_DIR
from arena_simulation_setup.tree.World import FORMAT, World, WorldIdentifier, parse_world

BOX = {'format': FORMAT, 'levels': [{'id': '0', 'zones': [{'name': 'box', 'polygon': [[0, 0], [4, 0], [4, 3], [0, 3]], 'walls': 'outline'}]}]}


def _box() -> World:
    return parse_world(yaml.safe_load(yaml.safe_dump(BOX)))


def test_world_written_through_resolve_write_lands_in_local_root_and_resolves_back(local_root):
    view = WorldIdentifier('mcp_written_box').resolve_write_sync()
    assert view.path == local_root / 'worlds' / 'mcp_written_box'
    view.save(_box())
    assert (view.path / 'world.yaml').is_file()
    assert (view.path / 'map' / '0.png').is_file()
    resolved = WorldIdentifier('mcp_written_box').resolve_sync()
    assert resolved.path == view.path
    assert resolved.load().level_ids == ['0']
    assert 'mcp_written_box' in {identifier.name for identifier in WorldIdentifier.listall()}


def test_installed_worlds_keep_resolving_from_the_canonical_tree(local_root):
    assert WorldIdentifier('map_empty').resolve_sync().path == ASS_DIR / 'worlds' / 'map_empty'
    assert WorldIdentifier('map_empty').resolve_write_sync().path == local_root / 'worlds' / 'map_empty'


def test_local_root_is_searched_before_the_installed_worlds(local_root):
    view = WorldIdentifier('map_empty').resolve_write_sync()
    view.save(_box())
    assert WorldIdentifier('map_empty').resolve_sync().path == local_root / 'worlds' / 'map_empty'


def test_local_root_follows_the_variable_at_every_resolve(local_root, tmp_path_factory):
    WorldIdentifier('mcp_moving').resolve_write_sync().save(_box())
    assert WorldIdentifier('mcp_moving').resolve_sync().path == local_root / 'worlds' / 'mcp_moving'
    other = tmp_path_factory.mktemp('other_root')
    os.environ['ARENA_ASSETS_DIR_LOCAL'] = str(other)
    assert WorldIdentifier('mcp_moving').resolve_write_sync().path == other / 'worlds' / 'mcp_moving'
