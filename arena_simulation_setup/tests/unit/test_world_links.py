from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import pytest
import yaml
from arena_simulation_setup.tree.World import WorldIdentifier, WorldView, parse_world
from arena_simulation_setup.tree.World.parse import WorldFormatError
from arena_simulation_setup.tree import SimplePathResolver
from arena_simulation_setup.tree.World.Scenario import ScenarioView
from arena_simulation_setup.tree.World.view import ComposedResolver, composite_parts


def _floor(level_id: str, elevator: str, destination: str, zone: str) -> dict:
    return {
        'id': level_id,
        'zones': [{'name': zone, 'polygon': [[0.0, 0.0], [10.0, 0.0], [10.0, 8.0], [0.0, 8.0]], 'walls': 'outline'}],
        'elevators': [{'name': elevator, 'position': [5.0, 0.0], 'door_side': '+y', 'destinations': [destination]}],
        'objects': [{'name': f'table_{level_id}', 'pose': [3.0, 3.0], 'model': 'table'}],
    }


def _write(directory: Path, *levels: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'world.yaml').write_text(yaml.safe_dump({'format': 'arena-world/2', 'levels': list(levels)}), encoding='utf-8')
    return directory


@pytest.fixture
def house(tmp_path: Path) -> Path:
    root = tmp_path / 'house'
    _write(root / 'floors' / 'ground', _floor('0', 'lift_0', 'lift_1', 'hall'))
    _write(root / 'floors' / 'upper', _floor('1', 'lift_1', 'lift_0', 'attic'))
    return _write(root, {'from': './floors/ground'}, {'from': './floors/upper'})


def test_subfolder_floors_load_as_the_levels_of_one_world(house: Path) -> None:
    world = WorldView(house).load()
    assert world.level_ids == ['0', '1']
    assert [zone.name for zone in world.levels['1'].zones] == ['attic']
    assert world.levels['0'].elevators[0].destinations == ['lift_1']
    assert world.levels['0'].source == './floors/ground'
    assert world.sources == [(house / 'floors' / 'ground').resolve(), (house / 'floors' / 'upper').resolve()]


def test_linked_levels_get_a_derived_kg(house: Path) -> None:
    from arena_simulation_setup.tree.World import contents

    world = WorldView(house).load()
    assert world.kg is not None
    assert sorted(contents(world.kg, 'attic')) == ['elevator:lift_1', 'obj:table_1']


def test_a_link_can_rename_the_level(tmp_path: Path) -> None:
    _write(tmp_path / 'other', _floor('7', 'lift_7', 'lift_7', 'loft'))
    world = WorldView(_write(tmp_path / 'main', {'id': '3', 'from': '../other:7'})).load()
    assert world.level_ids == ['3']
    assert world.levels['3'].zones[0].name == 'loft'


def test_a_named_world_level_resolves_through_the_world_tree_and_drops_foreign_destinations(tmp_path: Path) -> None:
    world = WorldView(_write(tmp_path / 'main', {'id': '0', 'from': 'three_storied_residential:2'})).load()
    assert world.level_ids == ['0']
    assert [(elevator.name, elevator.destinations) for elevator in world.levels['0'].elevators] == [('2_elevator', [])]


def test_save_writes_the_link_and_keeps_the_floor_folders(house: Path) -> None:
    view = WorldView(house)
    view.save(view.load())
    assert yaml.safe_load((house / 'world.yaml').read_text(encoding='utf-8'))['levels'] == [{'id': '0', 'from': './floors/ground'}, {'id': '1', 'from': './floors/upper'}]
    assert (house / 'floors' / 'ground' / 'world.yaml').exists()
    assert WorldView(house).load().level_ids == ['0', '1']


def test_a_multi_level_world_needs_the_level_named(tmp_path: Path) -> None:
    _write(tmp_path / 'tower', _floor('0', 'a', 'b', 'low'), _floor('1', 'b', 'a', 'high'))
    with pytest.raises(WorldFormatError, match=r"levels\[0\]\.from: '\.\./tower' has levels \['0', '1'\], name one as \.\./tower:<level id>"):
        WorldView(_write(tmp_path / 'main', {'from': '../tower'})).load()


def test_an_unknown_level_is_named_in_the_error(tmp_path: Path) -> None:
    _write(tmp_path / 'tower', _floor('0', 'a', 'a', 'low'))
    with pytest.raises(WorldFormatError, match=r"has no level '5', only \['0'\]"):
        WorldView(_write(tmp_path / 'main', {'from': '../tower:5'})).load()


def test_links_cannot_cycle(tmp_path: Path) -> None:
    _write(tmp_path / 'a', {'from': '../b'})
    _write(tmp_path / 'b', {'from': '../a'})
    with pytest.raises(WorldFormatError, match='cannot link in a cycle'):
        WorldView(tmp_path / 'a').load()


def test_a_link_takes_no_level_content(tmp_path: Path) -> None:
    with pytest.raises(WorldFormatError, match=r"levels\[0\]: unknown keys \['zones'\]"):
        WorldView(_write(tmp_path / 'main', {'from': '../other', 'zones': []})).load()


def test_parse_world_alone_cannot_follow_a_link() -> None:
    with pytest.raises(WorldFormatError, match=r'levels\[0\]\.from: a linked level is read through WorldView'):
        parse_world({'format': 'arena-world/2', 'levels': [{'from': 'map_empty'}]})


def test_assets_of_a_linked_world_resolve(tmp_path: Path) -> None:
    from arena_simulation_setup.tree import DynamicPaths
    from arena_simulation_setup.tree.assets.Object import ObjectIdentifier

    model = tmp_path / 'other' / 'assets' / 'Common' / 'Object' / 'Linked_Only_Shelf'
    model.mkdir(parents=True)
    (model / 'annotation.yaml').write_text(yaml.safe_dump({'name': 'Linked_Only_Shelf', 'path': 'Object/Linked_Only_Shelf'}))
    previous = DynamicPaths.LINKED.paths
    DynamicPaths.LINKED.paths = [tmp_path / 'other']
    try:
        assert ObjectIdentifier.parse('Common/Linked_Only_Shelf').resolve_sync().path == model
    finally:
        DynamicPaths.LINKED.paths = previous


def _kg_data(directory: Path, uses: list[str], entities: list | None = None, facts: list | None = None, constraints: list | None = None) -> None:
    (directory / 'kg.yaml').write_text(yaml.safe_dump({'format': 'arena-kg/1', 'uses': uses, 'entities': entities or [], 'facts': facts or [], 'constraints': constraints or []}), encoding='utf-8')


@pytest.fixture
def annotated_house(house: Path) -> Path:
    _kg_data(
        house / 'floors' / 'ground',
        ['core@2', 'social@1', 'access@1'],
        entities=[{'id': 'room:hall', 'props': {'type': 'Lobby', 'description': 'the ground hall'}}, {'id': 'role:staff'}],
        facts=[{'subject': 'room:hall', 'verb': 'restricted_to', 'object': 'role:staff'}],
        constraints=[{'id': 'con:staff_only', 'kind': 'no_entry', 'scope': ['role:staff', 'room:hall']}],
    )
    _kg_data(house, ['core@2'], entities=[{'id': 'room:hall', 'props': {'description': 'the main hall'}}])
    return house


def test_a_linked_floor_brings_its_kg_and_the_root_wins(annotated_house: Path) -> None:
    from arena_simulation_setup.tree.World import describe

    scene = WorldView(annotated_house).load().kg
    assert scene is not None
    assert scene.uses == ['core@2', 'social@1', 'access@1']
    assert scene.get_entity('room:hall').props == {'description': 'the main hall', 'type': 'Lobby'}
    assert describe(scene, 'hall') == 'the main hall'
    assert scene.get_fact('room:hall', 'restricted_to', 'role:staff') is not None
    assert scene.get_constraint('con:staff_only') is not None


def test_saving_keeps_linked_kg_in_their_own_files(annotated_house: Path) -> None:
    view = WorldView(annotated_house)
    view.save(view.load())
    root = yaml.safe_load((annotated_house / 'kg.yaml').read_text(encoding='utf-8'))
    assert [entity for entity in root['entities'] if entity['id'] == 'room:hall'] == [{'id': 'room:hall', 'props': {'description': 'the main hall'}}]
    assert [fact for fact in root['facts'] if fact.get('source') != 'derived'] == []
    assert 'constraints' not in root or root['constraints'] == []
    reloaded = WorldView(annotated_house).load().kg
    assert reloaded is not None
    assert reloaded.get_entity('room:hall').props == {'description': 'the main hall', 'type': 'Lobby'}
    assert reloaded.get_constraint('con:staff_only') is not None


def test_linked_kg_follows_the_level_rename_and_drops_what_was_not_linked(tmp_path: Path) -> None:
    tower = _write(tmp_path / 'tower', _floor('0', 'a', 'b', 'low'), _floor('1', 'b', 'a', 'high'))
    _kg_data(
        tower,
        ['core@2', 'social@1', 'access@1'],
        entities=[{'id': 'room:high', 'props': {'type': 'Loft'}}, {'id': 'room:low', 'props': {'type': 'Cellar'}}],
        facts=[
            {'subject': 'floor:1', 'verb': 'contains', 'object': 'room:high'},
            {'subject': 'room:low', 'verb': 'restricted_to', 'object': 'role:staff'},
        ],
    )
    scene = WorldView(_write(tmp_path / 'main', {'id': '5', 'from': '../tower:1'})).load().kg
    assert scene is not None
    stated = scene.get_fact('floor:5', 'contains', 'room:high')
    assert stated is not None and stated.source == 'generated'
    assert scene.get_entity('floor:1') is None
    assert scene.get_entity('room:low') is None
    assert scene.get_fact('room:low', 'restricted_to', 'role:staff') is None
    assert scene.get_entity('room:high').props == {'type': 'Loft'}


def test_a_prefix_lets_one_floor_repeat(tmp_path: Path) -> None:
    floor = _write(tmp_path / 'house' / 'floor', _floor('0', 'lift', 'lift', 'flat'))
    _kg_data(floor, ['core@2'], entities=[{'id': 'room:flat', 'props': {'type': 'Apartment'}}], facts=[{'subject': 'floor:0', 'verb': 'contains', 'object': 'room:flat'}])
    root = _write(tmp_path / 'house', *({'id': str(n), 'from': './floor', 'prefix': f'f{n}_'} for n in range(3)))
    world = WorldView(root).load()
    assert [zone.name for level in world.all_levels for zone in level.zones] == ['f0_flat', 'f1_flat', 'f2_flat']
    assert [(elevator.name, elevator.destinations) for level in world.all_levels for elevator in level.elevators] == [('f0_lift', ['f0_lift']), ('f1_lift', ['f1_lift']), ('f2_lift', ['f2_lift'])]
    assert world.kg is not None
    for n in range(3):
        assert world.kg.get_entity(f'room:f{n}_flat').props == {'type': 'Apartment'}
        assert world.kg.get_fact(f'floor:{n}', 'contains', f'room:f{n}_flat') is not None
    assert world.kg.get_entity('room:flat') is None


def test_save_keeps_the_prefix(tmp_path: Path) -> None:
    _write(tmp_path / 'house' / 'floor', _floor('0', 'lift', 'lift', 'flat'))
    view = WorldView(_write(tmp_path / 'house', {'id': '4', 'from': './floor', 'prefix': 'top_'}))
    view.save(view.load())
    assert yaml.safe_load((tmp_path / 'house' / 'world.yaml').read_text(encoding='utf-8'))['levels'] == [{'id': '4', 'from': './floor', 'prefix': 'top_'}]


_ALARM = {'name': 'fire_alarm', 'semantics': [{'preset': 'schedule', 'params': {'windows': [], 'regime': 'alarm'}}]}


def _with_schedules(directory: Path, *schedules: dict) -> None:
    data = yaml.safe_load((directory / 'world.yaml').read_text(encoding='utf-8'))
    data['schedules'] = list(schedules)
    (directory / 'world.yaml').write_text(yaml.safe_dump(data), encoding='utf-8')


def test_a_linked_world_brings_its_schedules_once_and_save_leaves_them_there(tmp_path: Path) -> None:
    floor = _write(tmp_path / 'house' / 'floor', _floor('0', 'lift', 'lift', 'flat'))
    _with_schedules(floor, _ALARM)
    view = WorldView(_write(tmp_path / 'house', {'id': '0', 'from': './floor', 'prefix': 'a_'}, {'id': '1', 'from': './floor', 'prefix': 'b_'}))
    world = view.load()
    assert [schedule.name for schedule in world.schedules] == ['fire_alarm']
    view.save(world)
    assert 'schedules' not in yaml.safe_load((tmp_path / 'house' / 'world.yaml').read_text(encoding='utf-8'))
    assert [schedule.name for schedule in WorldView(tmp_path / 'house').load().schedules] == ['fire_alarm']


def test_a_schedule_that_differs_from_the_root_one_is_an_error(tmp_path: Path) -> None:
    floor = _write(tmp_path / 'house' / 'floor', _floor('0', 'lift', 'lift', 'flat'))
    _with_schedules(floor, _ALARM)
    root = _write(tmp_path / 'house', {'from': './floor'})
    _with_schedules(root, {'name': 'fire_alarm', 'semantics': [{'preset': 'schedule', 'params': {'windows': [], 'regime': 'drill'}}]})
    with pytest.raises(ValueError, match="schedule 'fire_alarm' of .* differs from the one this world already has"):
        WorldView(root).load()


def test_editing_a_linked_floor_marks_the_world_changed_and_reloads_the_edit(house: Path) -> None:
    view = WorldView(house)
    world = view.load()
    loaded = view.mtime(world.sources)
    floor = house / 'floors' / 'upper' / 'world.yaml'
    edited = yaml.safe_load(floor.read_text(encoding='utf-8'))
    edited['levels'][0]['zones'][0]['name'] = 'loft'
    floor.write_text(yaml.safe_dump(edited), encoding='utf-8')
    os.utime(floor, (loaded + 5.0, loaded + 5.0))
    assert view.mtime() == view.mtime(())
    assert view.mtime() < loaded + 5.0
    assert view.mtime(world.sources) == loaded + 5.0
    assert [zone.name for zone in view.load().levels['1'].zones] == ['loft']


def test_a_linked_floor_kg_edit_marks_the_world_changed(annotated_house: Path) -> None:
    view = WorldView(annotated_house)
    world = view.load()
    loaded = view.mtime(world.sources)
    kg = annotated_house / 'floors' / 'ground' / 'kg.yaml'
    os.utime(kg, (loaded + 5.0, loaded + 5.0))
    assert view.mtime(world.sources) == loaded + 5.0


def _composed(root: Path, name: str) -> WorldView:
    path = asyncio.run(ComposedResolver(root).resolve(WorldIdentifier(name)))
    assert path is not None
    return WorldView(path)


def test_a_comma_list_splits_outside_the_level_filters() -> None:
    assert composite_parts('hospital_2[0,1], office_1') == ['hospital_2[0,1]', 'office_1']
    assert composite_parts('hospital_2[0,1]') == ['hospital_2[0,1]']
    assert WorldIdentifier.parse('hospital_2[1],office_1') == ('hospital_2[1],office_1', None)
    assert WorldIdentifier.parse('hospital_2[0,1]') == ('hospital_2', {'0', '1'})


def test_a_single_world_is_not_composed(tmp_path: Path) -> None:
    assert asyncio.run(ComposedResolver(tmp_path).resolve(WorldIdentifier('hospital_1'))) is None
    assert asyncio.run(ComposedResolver(tmp_path).resolve(WorldIdentifier('hospital_2[1]'))) is None


def test_a_comma_list_stacks_the_worlds_as_levels_of_one_world(tmp_path: Path) -> None:
    view = _composed(tmp_path, 'hospital_1,office_1')
    assert view.path == tmp_path / 'hospital_1,office_1'
    assert yaml.safe_load(view.world_path.read_text(encoding='utf-8'))['levels'] == [
        {'id': '0', 'from': 'hospital_1:0', 'prefix': 'hospital_1_'},
        {'id': '1', 'from': 'office_1:0', 'prefix': 'office_1_'},
    ]
    world = view.load()
    assert world.level_ids == ['0', '1']
    assert all(zone.name.startswith('hospital_1_') for zone in world.levels['0'].zones)
    assert all(zone.name.startswith('office_1_') for zone in world.levels['1'].zones)
    assert [elevator for level in world.all_levels for elevator in level.elevators] == []


def test_a_world_listed_twice_gets_a_second_prefix(tmp_path: Path) -> None:
    world = _composed(tmp_path, 'map_empty,map_empty').load()
    assert sorted(elevator.name for elevator in world.levels['1'].elevators) == ['map_empty_2_elevator_ground', 'map_empty_2_elevator_top']
    assert world.levels['1'].elevators[0].destinations == ['map_empty_2_elevator_top']


def test_level_filters_pick_levels_and_free_elevators_ride_between_them(tmp_path: Path) -> None:
    world = _composed(tmp_path, 'hospital_2[0],three_storied_residential[1]').load()
    assert world.level_ids == ['0', '1']
    cabins = {elevator.name: elevator.destinations for level in world.all_levels for elevator in level.elevators}
    assert cabins == {'hospital_2_elevator_ground': ['three_storied_residential_1_elevator'], 'three_storied_residential_1_elevator': ['hospital_2_elevator_ground']}


def test_a_whole_multi_level_world_keeps_its_own_elevators(tmp_path: Path) -> None:
    world = _composed(tmp_path, 'hospital_2,office_1').load()
    assert world.level_ids == ['0', '1', '2']
    cabins = {elevator.name: elevator.destinations for level in world.all_levels for elevator in level.elevators}
    assert cabins == {'hospital_2_elevator_ground': ['hospital_2_elevator_top'], 'hospital_2_elevator_top': ['hospital_2_elevator_ground']}


def test_an_unknown_level_of_a_listed_world_is_named(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"world 'hospital_2' has no level 7, only 0, 1"):
        _composed(tmp_path, 'hospital_2[7],office_1')


def test_composing_again_keeps_the_written_world(tmp_path: Path) -> None:
    view = _composed(tmp_path, 'hospital_1,office_1')
    stamp = view.world_path.stat().st_mtime_ns
    assert _composed(tmp_path, 'hospital_1,office_1').world_path.stat().st_mtime_ns == stamp


def test_free_elevators_of_linked_levels_ride_in_a_ring(tmp_path: Path) -> None:
    for i, name in enumerate('abc'):
        _write(tmp_path / name, _floor(str(i), f'lift_{name}', 'gone', f'hall_{name}'))
    world = WorldView(_write(tmp_path / 'main', *({'id': str(i), 'from': f'../{name}'} for i, name in enumerate('abc')))).load()
    cabins = {elevator.name: elevator.destinations for level in world.all_levels for elevator in level.elevators}
    assert cabins == {'lift_a': ['lift_b'], 'lift_b': ['lift_c'], 'lift_c': ['lift_a']}


class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def test_a_level_with_two_free_elevators_stays_out_of_the_ring(tmp_path: Path) -> None:
    _write(tmp_path / 'a', _floor('0', 'lift_a', 'gone', 'hall_a'))
    _write(tmp_path / 'b', _floor('1', 'lift_b', 'gone', 'hall_b'))
    twin = _floor('2', 'lift_c', 'gone', 'hall_c')
    twin['elevators'].append({'name': 'lift_d', 'position': [8.0, 0.0], 'door_side': '+y', 'destinations': ['gone']})
    _write(tmp_path / 'c', twin)
    logger = logging.getLogger('arena_simulation_setup.tree.World.view')
    collect = _Collect()
    logger.addHandler(collect)
    try:
        world = WorldView(_write(tmp_path / 'main', *({'id': str(i), 'from': f'../{name}'} for i, name in enumerate('abc')))).load()
    finally:
        logger.removeHandler(collect)
    cabins = {elevator.name: elevator.destinations for level in world.all_levels for elevator in level.elevators}
    assert cabins == {'lift_a': ['lift_b'], 'lift_b': ['lift_a'], 'lift_c': [], 'lift_d': []}
    assert any("elevators ['lift_c', 'lift_d'] have no destinations" in message for message in collect.messages)


def test_a_composite_follows_a_level_added_to_a_listed_world(tmp_path: Path) -> None:
    worlds = tmp_path / 'worlds'
    _write(worlds / 'stack_a', _floor('0', 'lift_a', 'lift_a', 'hall_a'))
    _write(worlds / 'stack_b', _floor('0', 'lift_b', 'lift_b', 'hall_b'))
    resolver = SimplePathResolver(WorldIdentifier, worlds)
    WorldIdentifier.use(resolver)
    try:
        composer = ComposedResolver(tmp_path / 'composed')
        identifier = WorldIdentifier('stack_a,stack_b')
        first = asyncio.run(composer.resolve(identifier))
        assert first is not None
        assert WorldView(first).load().level_ids == ['0', '1']
        stamp = (first / 'world.yaml').stat().st_mtime_ns
        assert asyncio.run(composer.resolve(identifier)) == first
        assert (first / 'world.yaml').stat().st_mtime_ns == stamp
        grown = _write(worlds / 'stack_b', _floor('0', 'lift_b', 'lift_b', 'hall_b'), _floor('1', 'lift_c', 'lift_c', 'hall_c'))
        later = (grown / 'world.yaml').stat().st_mtime + 5.0
        os.utime(grown / 'world.yaml', (later, later))
        assert asyncio.run(composer.resolve(identifier)) == first
        assert WorldView(first).load().level_ids == ['0', '1', '2']
    finally:
        WorldIdentifier._resolvers.remove(resolver)


def _custom_world(tmp_path: Path, start: str) -> Path:
    root = _write(tmp_path / 'custom_world', {'id': '0', 'from': 'hospital_1', 'prefix': 'hospital_1_'}, {'id': '1', 'from': 'office_1:0', 'prefix': 'office_1_'})
    scenario = root / 'scenarios' / 'cross'
    scenario.mkdir(parents=True)
    robot = {'start': start, 'goal_level': '1', 'phases': [{'goto': 'office_1_cafeteria'}]}
    (scenario / 'scenario.yaml').write_text(yaml.safe_dump({'format': 'arena-scenario/2', 'robots': [robot]}), encoding='utf-8')
    return root


def test_a_scenario_on_a_linked_world_names_the_prefixed_zones(tmp_path: Path) -> None:
    import numpy as np
    import shapely

    root = _custom_world(tmp_path, 'hospital_1_reception')
    view = WorldView(root)
    world = view.load()
    assert world.level_ids == ['0', '1']
    compacted = world.compact_world(origins=world.level_origins())
    scenario = view.scenario('cross').resolve_sync().load(converter=compacted.zone_converter(np.random.default_rng(0)))
    robot = scenario.robots[0]
    reception = next(zone for zone in compacted.zones if zone.name == 'hospital_1_reception')
    assert reception.shape().covers(shapely.Point(robot.start.position.x, robot.start.position.y))
    assert robot.phases[0].target == 'office_1_cafeteria'
    assert [identifier.name for identifier in view.identifiers(strict=True) if isinstance(identifier, WorldIdentifier)] == ['hospital_1', 'office_1']


def test_an_unprefixed_zone_in_a_linked_world_scenario_is_an_error(tmp_path: Path) -> None:
    import numpy as np

    root = _custom_world(tmp_path, 'reception')
    world = WorldView(root).load()
    compacted = world.compact_world(origins=world.level_origins())
    with pytest.raises(RuntimeError, match=r"robots\[0\]: could not parse .*'start': 'reception'"):
        ScenarioView(root / 'scenarios' / 'cross').load(converter=compacted.zone_converter(np.random.default_rng(0)))
