"""Read format 1 worlds (per-level zone files) as format 2 mappings plus a KG."""

from __future__ import annotations

import copy
import itertools
import logging
import math
from collections.abc import Mapping

from arena_simulation_setup.shared.entities import HUMANSIM_OBJECT_KEYS, WAYPOINT_MODES

from .model import FORMAT

logger = logging.getLogger(__name__)

_ZONE_ALIASES = {'mat': 'material', 'ceiling_mat': 'ceiling_material', 'wall_mat': 'wall_material'}
_DOOR_KEYS = frozenset({'name', 'start', 'end', 'kind', 'width', 'height', 'material', 'activation_distance', 'transition_time', 'hold_time', 'semantics'})
_ELEVATOR_KEYS = frozenset({'name', 'position', 'size', 'door_side', 'material', 'destination', 'activation_distance', 'transition_time', 'hold_time', 'travel_time', 'accept_outside_calls', 'recall_on', 'semantics'})
_OBJECT_KEYS = frozenset({'name', 'model', 'pose', 'scale', 'light'})
_HUMAN_KEYS = frozenset({'name', 'model', 'pose', 'waypoints', 'waypoint_mode', 'velocity', 'agent'})
_ZONE_KEYS = frozenset({'name', 'description', 'material', 'corners', 'walls', 'doors', 'elevators', 'schedules', 'signals', 'sounds', 'lights', 'entities', 'ceiling', 'ceiling_height', 'ceiling_cast_shadows', 'ceiling_material', 'ceiling_lights', 'wall_material', 'semantics'})
_SOUND_KEYS = frozenset({'name', 'asset_id', 'position', 'attach', 'frame', 'offset', 'level', 'loop', 'reference_distance_m', 'semantics'})
_LIGHT_RENAMES = {'entity_ref': 'attach', 'pos': 'pose'}


def _aliased(raw: Mapping) -> dict:
    """A format 1 zone with its short keys renamed, the long key winning where both are given."""
    return {**{_ZONE_ALIASES[key]: value for key, value in raw.items() if key in _ZONE_ALIASES}, **{key: value for key, value in raw.items() if key not in _ZONE_ALIASES}}


def is_legacy(data: object) -> bool:
    """Whether a world.yaml mapping is format 1."""
    return not (isinstance(data, Mapping) and 'format' in data)


def _material(value: object) -> object:
    """A format 1 material in a form MaterialIdentifier parses: name, domain/name or [name, modifiers]."""
    if isinstance(value, Mapping):
        name = str(value.get('name', ''))
        domain = value.get('domain')
        modifiers = value.get('_modifiers', value.get('modifiers'))
        full = f'{domain}/{name}' if domain and name else name
        return [full, dict(modifiers)] if modifiers else full
    if isinstance(value, (list, tuple)) and len(value) == 2 and not value[1]:
        return value[0]
    return value


def _empty_material(value: object) -> bool:
    material = _material(value)
    return material == '' or (isinstance(material, list) and material[0] == '')


def _point(value: object) -> object:
    if isinstance(value, Mapping) and set(value) <= {'x', 'y', 'z'}:
        return [value.get('x', 0.0), value.get('y', 0.0), *([value['z']] if value.get('z') else [])]
    return value


def _same_point(a: object, b: object) -> bool:
    pa, pb = _point(a), _point(b)
    if not isinstance(pa, list) or not isinstance(pb, list):
        return pa == pb
    pa = [*pa, 0.0][:3] if len(pa) == 2 else pa
    pb = [*pb, 0.0][:3] if len(pb) == 2 else pb
    return all(math.isclose(float(x), float(y), abs_tol=1e-9) for x, y in zip(pa, pb, strict=True))


class _Upgrade:
    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.descriptions: dict[str, str] = {}
        self.contains: list[tuple[str, str, dict]] = []

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        logger.warning(message)

    def _drop(self, where: str, item: Mapping, allowed: frozenset[str]) -> dict:
        extra = sorted(set(item) - allowed)
        if extra:
            self.warn(f'{where}: dropped keys {extra} that format 2 does not know')
        return {key: value for key, value in item.items() if key in allowed}

    def zone(self, where: str, raw: Mapping) -> dict:
        raw = self._drop(where, _aliased(raw), _ZONE_KEYS)
        name = str(raw['name'])
        zone: dict = {'name': name, 'polygon': [_point(point) for point in raw.get('corners') or []]}
        if 'material' in raw:
            zone['floor'] = False if _empty_material(raw['material']) else _material(raw['material'])
        if raw.get('ceiling', True) is False or ('ceiling_material' in raw and _empty_material(raw['ceiling_material'])):
            zone['ceiling'] = False
        else:
            ceiling: dict = {}
            if raw.get('ceiling_height') is not None:
                ceiling['height'] = raw['ceiling_height']
            if 'ceiling_material' in raw:
                ceiling['material'] = _material(raw['ceiling_material'])
            if raw.get('ceiling_cast_shadows'):
                ceiling['cast_shadows'] = True
            if raw.get('ceiling_lights') is not None:
                ceiling['lights'] = raw['ceiling_lights']
            if ceiling:
                zone['ceiling'] = ceiling
        if raw.get('semantics'):
            zone['semantics'] = raw['semantics']
        if raw.get('description'):
            self.descriptions[name] = str(raw['description'])
        return zone

    def wall(self, raw: Mapping) -> dict | None:
        if 'material' in raw and raw['material'] is not None and _empty_material(raw['material']):
            return None
        wall: dict = {'path': [_point(raw['start']), _point(raw['end'])]}
        if raw.get('kind'):
            wall['kind'] = raw['kind']
        elif raw.get('material') is not None:
            wall['material'] = _material(raw['material'])
        return wall

    def door(self, where: str, raw: Mapping) -> dict:
        raw = self._drop(where, raw, _DOOR_KEYS)
        door = {key: value for key, value in raw.items() if key not in ('start', 'end')}
        door = {'name': door.pop('name'), 'span': [_point(raw['start']), _point(raw['end'])], **door}
        if 'material' in door:
            door['material'] = _material(door['material'])
        return door

    def elevator(self, where: str, raw: Mapping) -> dict:
        raw = self._drop(where, raw, _ELEVATOR_KEYS)
        elevator = dict(raw)
        destination = str(elevator.pop('destination', '') or '')
        names = [part.strip().split('.', 1)[-1] for part in destination.split(',') if part.strip()]
        if names:
            elevator['destinations'] = names
        if elevator.get('door_side', '+x') == '+x':
            elevator.pop('door_side', None)
        if 'material' in elevator:
            elevator['material'] = _material(elevator['material'])
        return elevator

    def obj(self, where: str, raw: Mapping) -> dict:
        raw = dict(raw)
        if 'pos' in raw:
            raw['pose'] = raw.pop('pos')
        raw.pop('level_id', None)
        humansim = {key: raw.pop(key) for key in list(raw) if key in HUMANSIM_OBJECT_KEYS}
        raw = self._drop(where, raw, _OBJECT_KEYS)
        if humansim:
            raw['humansim'] = humansim
        return raw

    def human(self, raw: Mapping, *, keep_level: bool) -> dict:
        raw = dict(raw)
        if 'pos' in raw:
            raw['pose'] = raw.pop('pos')
        level = raw.pop('level_id', None)
        if keep_level and level:
            raw['level'] = str(level)
        if 'waypoints' in raw:
            raw['waypoints'] = [_waypoint(wp) for wp in raw['waypoints'] or []]
        if 'waypoint_mode' in raw and str(raw['waypoint_mode']).lower() not in WAYPOINT_MODES:
            self.warn(f"human {raw.get('name')!r}: waypoint_mode {raw['waypoint_mode']!r} is not one of {list(WAYPOINT_MODES)}, it replays as repeat")
            raw.pop('waypoint_mode')
        elif 'waypoint_mode' in raw:
            raw['waypoint_mode'] = str(raw['waypoint_mode']).lower()
        hunav = {key: raw.pop(key) for key in list(raw) if key not in _HUMAN_KEYS | {'level'}}
        if hunav:
            raw['hunav'] = hunav
        return raw

    def light(self, raw: Mapping) -> dict:
        return {_LIGHT_RENAMES.get(key, key): value for key, value in raw.items()}

    def sound(self, where: str, raw: Mapping) -> dict:
        raw = {_LIGHT_RENAMES.get(key, key): value for key, value in raw.items()}
        return self._drop(where, raw, _SOUND_KEYS)


def _waypoint(value: object) -> object:
    if isinstance(value, Mapping):
        if 'position' in value:
            point = _point(value['position'])
            level = value.get('level_id')
            if level:
                x, y, *z = point if isinstance(point, list) else [point['x'], point['y']]
                return {'x': x, 'y': y, **({'z': z[0]} if z else {}), 'level': str(level)}
            return point
        level = value.get('level_id')
        flat = {key: value[key] for key in ('x', 'y', 'z') if key in value}
        if level:
            flat['level'] = str(level)
        return flat if level else _point(flat)
    return value


def _chain(walls: list[dict]) -> list[dict]:
    """Join consecutive two-point walls of one style that meet end to start into polylines."""
    result: list[dict] = []
    for wall in walls:
        last = result[-1] if result else None
        style = {key: value for key, value in wall.items() if key != 'path'}
        if last is not None and {key: value for key, value in last.items() if key != 'path'} == style and _same_point(last['path'][-1], wall['path'][0]):
            last['path'].append(wall['path'][1])
            continue
        result.append({'path': list(wall['path']), **style})
    for wall in result:
        if len(wall['path']) > 3 and _same_point(wall['path'][0], wall['path'][-1]):
            wall['path'].pop()
            wall['closed'] = True
    return result


def _dedupe(items: list[dict], kind: str, up: _Upgrade, level_id: str, taken: set[str]) -> list[dict]:
    """Drop repeats of an identical item, rename in place a different item whose name the world already uses."""
    seen: dict[str, dict] = {}
    result: list[dict] = []
    for item in items:
        name = str(item.get('name'))
        if name in seen and seen[name] == item:
            continue
        if name in seen or name in taken:
            renamed = f'{name}_{level_id}'
            for suffix in itertools.count(2):
                if renamed not in seen and renamed not in taken:
                    break
                renamed = f'{name}_{level_id}_{suffix}'
            up.warn(f'level {level_id}: a {kind} named {name!r} reuses a name of the world, renamed to {renamed!r}')
            item['name'] = renamed
        seen[str(item['name'])] = item
        result.append(item)
    taken.update(seen)
    return result


def upgrade(levels: Mapping[str, Mapping], graph: Mapping | None = None) -> tuple[dict, dict | None, list[str]]:
    """Format 2 world mapping, KG mapping (None when empty) and warnings for format 1 levels by id."""
    up = _Upgrade()
    world: dict = {'format': FORMAT}
    schedules: list[dict] = []
    signals: list[dict] = []
    out_levels: list[dict] = []
    taken: set[str] = set()
    for level_id, raw in levels.items():
        raw = copy.deepcopy(dict(raw or {}))
        level: dict = {'id': str(level_id)}
        zones, walls, doors, elevators, objects, humans, lights, sounds = [], [], [], [], [], [], [], []
        pedestrian = not raw.get('zones')
        for zone_index, zone_raw in enumerate(raw.get('zones') or []):
            where = f'{level_id}/world.yaml zones[{zone_index}]'
            zone_raw = _aliased(zone_raw)
            name = str(zone_raw['name'])
            zones.append(up.zone(where, zone_raw))
            if 'wall_material' not in zone_raw or not _empty_material(zone_raw['wall_material']):
                pedestrian = True
            walls.extend(wall for wall in (up.wall(item) for item in zone_raw.get('walls') or []) if wall is not None)
            for i, item in enumerate(zone_raw.get('doors') or []):
                doors.append(up.door(f'{where}.doors[{i}]', item))
            for i, item in enumerate(zone_raw.get('elevators') or []):
                elevators.append(up.elevator(f'{where}.elevators[{i}]', item))
                up.contains.append((name, 'elevator', elevators[-1]))
            schedules.extend(zone_raw.get('schedules') or [])
            signals.extend(zone_raw.get('signals') or [])
            for i, item in enumerate(zone_raw.get('sounds') or []):
                sounds.append(up.sound(f'{where}.sounds[{i}]', item))
                up.contains.append((name, 'sound', sounds[-1]))
            for item in zone_raw.get('lights') or []:
                lights.append(up.light(item))
                up.contains.append((name, 'light', lights[-1]))
            entities = zone_raw.get('entities') or {}
            for i, item in enumerate(entities.get('static') or []):
                objects.append(up.obj(f'{where}.entities.static[{i}]', item))
                up.contains.append((name, 'obj', objects[-1]))
            humans.extend(up.human(item, keep_level=False) for item in entities.get('dynamic') or [])
        level_entities = raw.get('entities') or {}
        objects.extend(up.obj(f'{level_id}/world.yaml entities.static[{i}]', item) for i, item in enumerate(level_entities.get('static') or []))
        humans.extend(up.human(item, keep_level=False) for item in level_entities.get('dynamic') or [])
        lights[:0] = [up.light(item) for item in raw.get('lights') or []]
        level['zones'] = _dedupe(zones, 'zone', up, str(level_id), taken)
        sections = {
            'walls': _chain(walls),
            'doors': _dedupe(doors, 'door', up, str(level_id), taken),
            'elevators': _dedupe(elevators, 'elevator', up, str(level_id), taken),
            'objects': _dedupe(objects, 'object', up, str(level_id), taken),
            'humans': _dedupe(humans, 'human', up, str(level_id), taken),
            'lights': _dedupe(lights, 'light', up, str(level_id), taken),
            'sounds': _dedupe(sounds, 'sound', up, str(level_id), taken),
            'microphones': raw.get('microphones') or [],
        }
        level.update({key: value for key, value in sections.items() if value})
        if not pedestrian:
            level['pedestrian_walls'] = False
        unknown = sorted(set(raw) - {'zones', 'entities', 'lights', 'microphones'})
        if unknown:
            up.warn(f'{level_id}/world.yaml: dropped top-level keys {unknown} that format 2 does not know')
        out_levels.append(level)
    if schedules:
        world['schedules'] = _dedupe(schedules, 'schedule', up, 'world', taken)
    if signals:
        world['signals'] = _dedupe(signals, 'signal', up, 'world', taken)
    world['levels'] = out_levels
    return world, _kg_data(up, graph, {str(door['name']) for level in out_levels for door in level.get('doors', [])}), up.warnings


def _kg_data(up: _Upgrade, graph: Mapping | None, doors: set[str]) -> dict | None:
    from arena_simulation_setup.kg import dump, lift_graph
    from arena_simulation_setup.kg.scene import Scene

    scene = lift_graph(graph, doors) if graph else Scene()
    for name, description in up.descriptions.items():
        entity = scene.get_entity(f'room:{name}')
        if entity is None or 'description' not in entity.props:
            scene.add_entity(f'room:{name}', {'description': description})
    for room, kind, item in up.contains:
        name = str(item['name'])
        scene.add_entity(f'room:{room}')
        scene.add_entity(f'{kind}:{name}')
        scene.add_fact(f'room:{room}', 'contains', f'{kind}:{name}')
    if not scene.entities and not scene.facts:
        return None
    return dump(scene)
