from __future__ import annotations

from pathlib import Path

import attrs
import yaml
from arena_simulation_setup.tree.World import WorldView, save_snapshot


def _world(directory: Path, *levels: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'world.yaml').write_text(yaml.safe_dump({'format': 'arena-world/2', 'levels': list(levels)}), encoding='utf-8')
    return directory


def _level(level_id: str, zone: str) -> dict:
    return {
        'id': level_id,
        'zones': [{'name': zone, 'polygon': [[0.0, 0.0], [6.0, 0.0], [6.0, 5.0], [0.0, 5.0]], 'walls': 'outline', 'ceiling': {'height': 2.7, 'material': 'Concrete_Smooth'}}],
        'objects': [{'name': f'desk_{level_id}', 'pose': [2.0, 2.0], 'model': 'table'}],
    }


def test_snapshot_writes_linked_levels_out_with_their_lighting_and_every_fact(tmp_path: Path) -> None:
    _world(tmp_path / 'house' / 'floors' / 'ground', _level('0', 'hall'))
    root = _world(tmp_path / 'house', {'from': './floors/ground'}, {'id': '1', 'from': './floors/ground', 'prefix': 'up_'})
    world = WorldView(root).load()
    target = save_snapshot(world, tmp_path / 'run' / 'worlds' / 'house', lighting='auto')
    assert sorted(path.name for path in target.iterdir()) == ['kg.yaml', 'world.yaml']
    data = yaml.safe_load((target / 'world.yaml').read_text(encoding='utf-8'))
    assert all('from' not in level and 'prefix' not in level for level in data['levels'])
    assert [zone['name'] for level in data['levels'] for zone in level['zones']] == ['hall', 'up_hall']
    assert all('lights' in zone['ceiling'] for level in data['levels'] for zone in level['zones'])
    kg = yaml.safe_load((target / 'kg.yaml').read_text(encoding='utf-8'))
    assert {'subject': 'room:up_hall', 'verb': 'contains', 'object': 'obj:up_desk_0', 'source': 'derived'} in kg['facts']


def test_a_snapshot_loads_back_as_the_same_world(tmp_path: Path) -> None:
    _world(tmp_path / 'house' / 'floors' / 'ground', _level('0', 'hall'))
    world = WorldView(_world(tmp_path / 'house', {'from': './floors/ground'})).load()
    again = WorldView(save_snapshot(world, tmp_path / 'snapshot')).load()
    assert again.levels == {level_id: attrs.evolve(level, source='') for level_id, level in world.levels.items()}
    assert again.kg is not None and world.kg is not None
    assert {fact.key for fact in again.kg.facts} == {fact.key for fact in world.kg.facts}


def test_authored_lighting_leaves_the_ceilings_as_written(tmp_path: Path) -> None:
    world = WorldView(_world(tmp_path / 'flat', _level('0', 'hall'))).load()
    data = yaml.safe_load((save_snapshot(world, tmp_path / 'snapshot') / 'world.yaml').read_text(encoding='utf-8'))
    assert 'lights' not in data['levels'][0]['zones'][0]['ceiling']


def test_a_snapshot_carries_the_schedules_of_linked_worlds(tmp_path: Path) -> None:
    floor = _world(tmp_path / 'house' / 'floor', _level('0', 'hall'))
    data = yaml.safe_load((floor / 'world.yaml').read_text(encoding='utf-8'))
    data['schedules'] = [{'name': 'fire_alarm', 'semantics': [{'preset': 'schedule', 'params': {'windows': [], 'regime': 'alarm'}}]}]
    (floor / 'world.yaml').write_text(yaml.safe_dump(data), encoding='utf-8')
    world = WorldView(_world(tmp_path / 'house', {'from': './floor'})).load()
    written = yaml.safe_load((save_snapshot(world, tmp_path / 'snapshot') / 'world.yaml').read_text(encoding='utf-8'))
    assert [schedule['name'] for schedule in written['schedules']] == ['fire_alarm']
