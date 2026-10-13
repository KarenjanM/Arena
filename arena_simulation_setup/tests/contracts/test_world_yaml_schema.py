from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from arena_simulation_setup.tree.World import Level, WorldView, parse_world, write_world
from arena_simulation_setup.tree.World.write import dump_level

_WORLD = Path(__file__).resolve().parents[2] / 'worlds' / 'three_storied_residential'
_LEVEL = '1'


@pytest.fixture(scope='module')
def _raw_level() -> dict:
    if not _WORLD.is_dir():
        pytest.skip(f'fixture world not found: {_WORLD}')
    data, _, _ = WorldView(_WORLD).read()
    return next(level for level in data['levels'] if str(level['id']) == _LEVEL)


@pytest.fixture(scope='module')
def _level() -> Level:
    return WorldView(_WORLD).load().levels[_LEVEL]


def test_existing_world_loads_unchanged(_raw_level, _level):
    """Zones stay unannotated, doors and elevators carry whatever `semantics:` the fixture declares."""
    for zone in _level.zones:
        assert zone.semantics == []
    for kind, items in (('doors', _level.doors), ('elevators', _level.elevators)):
        assert [raw['name'] for raw in _raw_level.get(kind, [])] == [item.name for item in items]
        for raw, item in zip(_raw_level.get(kind, []), items, strict=True):
            assert (item.semantics != []) == ('semantics' in raw)


def test_existing_world_semantics_omitted_on_write(_raw_level, _level):
    """Empty `semantics:` is omitted for doors and elevators, present when the fixture declares it."""
    written = dump_level(_level, {})
    assert all('semantics' not in zone for zone in written['zones'])
    for kind in ('doors', 'elevators'):
        for raw, item in zip(_raw_level.get(kind, []), written.get(kind, []), strict=True):
            assert ('semantics' in item) == ('semantics' in raw)
            if 'semantics' in raw:
                assert item['semantics']


def test_existing_world_rewrite_is_stable():
    world = WorldView(_WORLD).load()
    reparsed = parse_world(yaml.safe_load(write_world(world)))
    for level_id, level in world.levels.items():
        again = reparsed.levels[level_id]
        assert [zone.name for zone in again.zones] == [zone.name for zone in level.zones]
        assert [door.name for door in again.doors] == [door.name for door in level.doors]
        assert [elevator.name for elevator in again.elevators] == [elevator.name for elevator in level.elevators]
        assert [door.semantics for door in again.doors] == [door.semantics for door in level.doors]
        assert [elevator.semantics for elevator in again.elevators] == [elevator.semantics for elevator in level.elevators]
        assert all(zone.semantics == [] for zone in again.zones)
