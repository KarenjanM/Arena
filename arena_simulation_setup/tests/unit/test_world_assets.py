from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml
from arena_simulation_setup.tree import DynamicPaths
from arena_simulation_setup.tree.assets.Human import HumanIdentifier
from arena_simulation_setup.tree.assets.Object import ObjectIdentifier
from arena_simulation_setup.tree.assets.Sound import SoundIdentifier
from arena_simulation_setup.tree.World import WorldView, world_assets
from arena_simulation_setup.tree.World.view import point_world_assets


def _world(directory: Path, *levels: dict) -> WorldView:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'world.yaml').write_text(yaml.safe_dump({'format': 'arena-world/2', 'levels': list(levels)}), encoding='utf-8')
    return WorldView(directory)


def _floor(level_id: str) -> dict:
    return {'id': level_id, 'zones': [{'name': f'room_{level_id}', 'polygon': [[0.0, 0.0], [6.0, 0.0], [6.0, 4.0], [0.0, 4.0]], 'walls': 'outline'}]}


def _model(world: Path, name: str) -> Path:
    model = world / 'assets' / 'Common' / 'Object' / name
    model.mkdir(parents=True)
    (model / 'annotation.yaml').write_text(yaml.safe_dump({'name': name, 'path': f'Object/{name}'}), encoding='utf-8')
    return model


def _resolve(name: str) -> Path:
    return ObjectIdentifier.parse(f'Common/{name}').resolve_sync().path


@pytest.fixture
def restore_paths():
    previous = DynamicPaths.WORLD.path, list(DynamicPaths.LINKED.paths)
    yield previous
    DynamicPaths.WORLD.path, DynamicPaths.LINKED.paths = previous


def test_world_and_linked_assets_resolve_inside_and_paths_restore_after(tmp_path: Path, restore_paths: tuple[Path, list[Path]]) -> None:
    floor = _world(tmp_path / 'house' / 'floors' / 'ground', _floor('0'))
    house = _world(tmp_path / 'house', {'from': './floors/ground'})
    own = _model(house.path, 'World_Assets_Own_Desk')
    linked = _model(floor.path, 'World_Assets_Linked_Shelf')
    with world_assets(house):
        assert DynamicPaths.WORLD.path == house.path
        assert DynamicPaths.LINKED.paths == [floor.path.resolve()]
        assert _resolve('World_Assets_Own_Desk') == own
        assert _resolve('World_Assets_Linked_Shelf') == linked
    assert (DynamicPaths.WORLD.path, DynamicPaths.LINKED.paths) == restore_paths


def test_a_loaded_world_supplies_the_linked_sources(tmp_path: Path, restore_paths: tuple[Path, list[Path]]) -> None:
    floor = _world(tmp_path / 'house' / 'floors' / 'ground', _floor('0'))
    house = _world(tmp_path / 'house', {'from': './floors/ground'})
    world = house.load(validate=False, level_filter=set())
    with world_assets(house, world):
        assert DynamicPaths.LINKED.paths == [floor.path.resolve()]
    assert (DynamicPaths.WORLD.path, DynamicPaths.LINKED.paths) == restore_paths


def test_nested_use_resolves_the_inner_world_and_falls_back_to_the_outer(tmp_path: Path, restore_paths: tuple[Path, list[Path]]) -> None:
    outer = _world(tmp_path / 'outer', _floor('0'))
    inner = _world(tmp_path / 'inner', _floor('0'))
    outer_desk = _model(outer.path, 'World_Assets_Nested_Desk')
    inner_desk = _model(inner.path, 'World_Assets_Nested_Desk')
    with world_assets(outer):
        assert _resolve('World_Assets_Nested_Desk') == outer_desk
        with world_assets(inner):
            assert DynamicPaths.WORLD.path == inner.path
            assert _resolve('World_Assets_Nested_Desk') == inner_desk
        assert DynamicPaths.WORLD.path == outer.path
        assert _resolve('World_Assets_Nested_Desk') == outer_desk
    assert (DynamicPaths.WORLD.path, DynamicPaths.LINKED.paths) == restore_paths


def test_an_exception_in_the_block_restores_the_paths(tmp_path: Path, restore_paths: tuple[Path, list[Path]]) -> None:
    floor = _world(tmp_path / 'house' / 'floors' / 'ground', _floor('0'))
    house = _world(tmp_path / 'house', {'from': './floors/ground'})
    with pytest.raises(RuntimeError, match='inside the block'), world_assets(house):
        assert DynamicPaths.LINKED.paths == [floor.path.resolve()]
        raise RuntimeError('inside the block')
    assert (DynamicPaths.WORLD.path, DynamicPaths.LINKED.paths) == restore_paths


def test_a_world_that_fails_to_load_leaves_the_paths_untouched(tmp_path: Path, restore_paths: tuple[Path, list[Path]]) -> None:
    broken = _world(tmp_path / 'broken', {'from': '../missing_floor_world'})
    with pytest.raises(Exception, match='missing_floor_world'), world_assets(broken):
        pass
    assert (DynamicPaths.WORLD.path, DynamicPaths.LINKED.paths) == restore_paths


@pytest.mark.parametrize('kind', [SoundIdentifier, HumanIdentifier, ObjectIdentifier])
def test_every_asset_kind_resolves_the_world_it_is_pointed_at_after_a_switch(tmp_path: Path, restore_paths: tuple[Path, list[Path]], kind: type) -> None:
    label = kind.label()
    first = _world(tmp_path / 'first', _floor('0'))
    second = _world(tmp_path / 'second', _floor('0'))
    for world in (first, second):
        (world.path / 'assets' / 'Common' / label / f'World_Assets_Switch_{label}').mkdir(parents=True)
    ident = kind.parse(f'Common/World_Assets_Switch_{label}')
    with world_assets(first):
        assert asyncio.run(ident.resolve_path()) == first.path / 'assets' / 'Common' / label / f'World_Assets_Switch_{label}'
    with world_assets(second):
        assert asyncio.run(ident.resolve_path()) == second.path / 'assets' / 'Common' / label / f'World_Assets_Switch_{label}'


def test_pointing_at_no_world_points_at_dev_null(tmp_path: Path, restore_paths: tuple[Path, list[Path]]) -> None:
    house = _world(tmp_path / 'house', _floor('0'))
    point_world_assets(house.path, [tmp_path])
    assert (DynamicPaths.WORLD.path, DynamicPaths.LINKED.paths) == (house.path, [tmp_path])
    point_world_assets(None, [])
    assert (DynamicPaths.WORLD.path, DynamicPaths.LINKED.paths) == (Path('/dev/null'), [])
