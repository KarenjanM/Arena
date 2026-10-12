import logging
import os
import typing
import warnings
from collections.abc import Iterable, Iterator
from pathlib import Path

import attrs
import numpy as np
import yaml
from typing_extensions import Self

from arena_simulation_setup import ASS_DIR
from arena_simulation_setup.tree import (
    WORLD_PROVIDERS,
    FallbackResolver,
    Identifier,
    NetResolver,
    PathView,
    References,
    SimplePathResolver,
)

from .legacy import is_legacy, upgrade
from .Map import Map
from .model import Level, World
from .parse import parse_world
from .Scenario import ScenarioView
from .write import to_yaml, write_world

if typing.TYPE_CHECKING:
    from arena_simulation_setup.kg import Scene

logger = logging.getLogger(__name__)

KG_FILE = 'kg.yaml'
LEGACY_GRAPH_FILE = 'graph.yaml'
_NOT_LEVELS = frozenset({'scenarios', 'assets', 'map'})


class WorldView(PathView):
    @property
    def scenario(self) -> type[Identifier[ScenarioView]]:
        class ScenarioIdentifier(Identifier[ScenarioView]):
            @classmethod
            def listall(cls, **kwargs: object) -> Iterator[Self]:
                scenarios_dir = self.path / 'scenarios'
                if not scenarios_dir.is_dir():
                    yield from ()
                    return
                yield from (cls(entry.name) for entry in os.scandir(scenarios_dir) if entry.is_dir())

            @classmethod
            async def listall_async(cls, **kwargs: object) -> list[Self]:
                return list(cls.listall(**kwargs))

            def load(self, path: Path, /, **kwargs: object) -> ScenarioView:
                del kwargs
                return ScenarioView(path)

        ScenarioIdentifier.use(SimplePathResolver(ScenarioIdentifier, self.path / 'scenarios'))
        ScenarioIdentifier.use(FallbackResolver(ScenarioIdentifier, self.path / 'scenarios'))
        return ScenarioIdentifier

    @property
    def world_path(self) -> Path:
        return self.path / 'world.yaml'

    @property
    def kg_path(self) -> Path:
        return self.path / KG_FILE

    def mtime(self, sources: Iterable[Path] = ()) -> float:
        """The newest modification time of this world's files and those of the linked worlds in `sources`, 0.0 without any."""
        files = [self.world_path, self.kg_path, *(source / name for source in sources for name in ('world.yaml', KG_FILE))]
        return max((path.stat().st_mtime for path in files if path.exists()), default=0.0)

    def map_yaml(self, level_id: str) -> Path:
        """The preview map of a level, `map/<level id>.yaml` beside its `<level id>.png`, or `<level id>/map.yaml` in a format 1 world."""
        current = self.path / 'map' / f'{level_id}.yaml'
        legacy = self.path / level_id / 'map.yaml'
        return legacy if not current.exists() and legacy.exists() else current

    def _legacy_level_dirs(self) -> list[Path]:
        if not self.path.is_dir():
            return []
        return sorted(entry for entry in self.path.iterdir() if entry.is_dir() and entry.name not in _NOT_LEVELS and (entry / 'world.yaml').exists() and is_legacy(yaml.safe_load((entry / 'world.yaml').read_text(encoding='utf-8'))))

    def _drop_legacy_levels(self, level_dirs: list[Path]) -> None:
        for entry in level_dirs:
            for name in ('world.yaml', 'map.png', 'map.yaml'):
                if (entry / name).exists():
                    (entry / name).unlink()
            if not any(entry.iterdir()):
                entry.rmdir()

    def _legacy_levels(self) -> dict[str, dict] | None:
        level_dirs = self._legacy_level_dirs()
        if level_dirs:
            return {entry.name: yaml.safe_load((entry / 'world.yaml').read_text(encoding='utf-8')) or {} for entry in level_dirs}
        if not self.world_path.exists():
            return None
        data = yaml.safe_load(self.world_path.read_text(encoding='utf-8')) or {}
        if not is_legacy(data):
            return None
        if isinstance(data, dict) and 'levels' in data and 'zones' not in data:
            return {str(key): value for key, value in (data['levels'] or {}).items()}
        return {'0': data}

    def read(self) -> tuple[dict, dict | None, list[str]]:
        """The format 2 world mapping, the KG mapping and the legacy upgrade warnings."""
        if self.world_path.exists():
            data = yaml.safe_load(self.world_path.read_text(encoding='utf-8'))
            if not is_legacy(data):
                kg = yaml.safe_load(self.kg_path.read_text(encoding='utf-8')) if self.kg_path.exists() else None
                return data, kg, []
        levels = self._legacy_levels()
        if levels is None:
            raise FileNotFoundError(f'could not find world.yaml in {self.path}')
        graph_path = self.path / LEGACY_GRAPH_FILE
        graph = yaml.safe_load(graph_path.read_text(encoding='utf-8')) if graph_path.exists() else None
        world, kg, notes = upgrade(levels, graph)
        warnings.warn(f'world {self.path} is in format 1, migrate it with touch_world --migrate', DeprecationWarning, stacklevel=2)
        return world, kg, notes

    def _linked_level(self, source: str, chain: tuple[Path, ...], sources: list[Path], links: dict[str, tuple[Path, World, str]]) -> Level:
        name, _, level_id = source.partition(':')
        view = WorldView(self.path / name) if name.startswith(('./', '../')) else WorldIdentifier(name).resolve_sync()
        where = view.path.resolve()
        if where in chain:
            raise ValueError(f'{source!r} links back into {view.path}, levels cannot link in a cycle')
        world = view._parse(chain + (where,), sources)
        if not level_id:
            if len(world.levels) != 1:
                raise ValueError(f'{name!r} has levels {sorted(world.levels)}, name one as {name}:<level id>')
            level_id = next(iter(world.levels))
        if level_id not in world.levels:
            raise ValueError(f'{name!r} has no level {level_id!r}, only {sorted(world.levels)}')
        if where not in sources:
            sources.append(where)
        links[source] = (where, world, level_id)
        return world.levels[level_id]

    def _parse(self, chain: tuple[Path, ...], sources: list[Path], data: dict | None = None, kg: dict | None = None) -> World:
        from arena_simulation_setup.kg import load as load_kg

        from .membership import merge_linked

        if data is None:
            data, kg, _ = self.read()
        links: dict[str, tuple[Path, World, str]] = {}
        world = parse_world(data, link=lambda source: self._linked_level(source, chain, sources, links))
        world.kg = load_kg(kg) if kg is not None else None
        groups: dict[tuple[Path, str], tuple[World, dict[str, str]]] = {}
        for level in world.all_levels:
            if level.source:
                where, linked, source_id = links[level.source]
                renames = groups.setdefault((where, level.prefix), (linked, {}))[1]
                renames[f'floor:{source_id}'] = f'floor:{level.id}'
                for kind, names in linked.levels[source_id].refs().items():
                    renames.update({f'{kind}:{name}': f'{kind}:{level.prefix}{name}' for name in names})
        for (where, _), (linked, _) in groups.items():
            for kind, own, theirs in (('schedule', world.schedules, linked.schedules), ('signal', world.signals, linked.signals)):
                names = {item.name: item for item in own}
                for item in theirs:
                    if item.name not in names:
                        own.append(item)
                        names[item.name] = item
                        world.linked_scope.append(item)
                    elif names[item.name] != item:
                        raise ValueError(f'{kind} {item.name!r} of {where} differs from the one this world already has')
        for linked, renames in groups.values():
            if linked.kg is not None:
                world.kg = merge_linked(world.kg, linked.kg, world, renames)
                world.linked_kg = merge_linked(world.linked_kg, linked.kg, world, renames)
        return world

    def load(self, validate: bool = True, level_filter: set[str] | None = None, derive: str = 'auto') -> World:
        """The World on disk, format 2 or format 1 through the legacy loader, its `from:` levels read from the worlds they name, restricted to `level_filter` when given, its KG derived under `derive`."""
        from .membership import derive as derive_kg

        data, kg, _ = self.read()
        sources: list[Path] = []
        world = self._parse((self.path.resolve(),), sources, data, kg)
        world.sources = sources
        _drop_foreign_destinations(world)
        all_ids = set(world.levels)
        if level_filter is not None:
            world.levels = {level_id: level for level_id, level in world.levels.items() if level_id in level_filter}
        world.apply_elevator_door_sides()
        if validate:
            world.validate(loaded_level_ids=set(world.levels) if world.levels.keys() != all_ids else None)
            if world.kg is not None:
                check_kg(world, world.kg)
        world.kg = derive_kg(world, derive)
        return world

    def identifiers(self, scenarios: bool = True, strict: bool = False) -> Iterator[Identifier]:
        """Every asset this world and its scenarios reference, deduplicated.

        A scenario that will not load is skipped, or raises when *strict*. Callers
        validating a world before publishing it want strict, so an unreadable
        scenario cannot pass as having no references.
        """
        seen: set[tuple[str, str]] = set()
        description = self.load(validate=False)
        for level in description.all_levels:
            name = level.source.partition(':')[0]
            if name and not name.startswith(('./', '../')) and ('WorldIdentifier', name) not in seen:
                seen.add(('WorldIdentifier', name))
                yield WorldIdentifier(name)

        def _fresh(identifier: Identifier) -> bool:
            key = (type(identifier).__name__, identifier.shortname)
            if key in seen:
                return False
            seen.add(key)
            return True

        def _expand(identifier: Identifier) -> Iterator[Identifier]:
            """Yield an identifier, then anything the asset it names references in turn."""
            if not identifier.name or not _fresh(identifier):
                return
            yield identifier
            if not type(identifier).NESTS:
                return
            try:
                loaded = identifier.resolve_sync()
            except Exception:
                return
            if not isinstance(loaded, References):
                return
            try:
                nested = list(loaded.identifiers())
            except Exception:
                if strict:
                    raise
                logging.warning('skipping references of %s', identifier.shortname, exc_info=True)
                return
            for entry in nested:
                yield from _expand(entry)

        for identifier in description.identifiers():
            yield from _expand(identifier)

        if not scenarios:
            return

        compacted = description.compact_world(dict.fromkeys(description.level_ids, (0.0, 0.0)))
        zone_converter = compacted.zone_converter(np.random.default_rng(0))

        for scenario in self.scenario.listall():
            view = scenario.resolve_sync()
            try:
                entries = list(view.identifiers(converter=zone_converter))
            except Exception:
                if strict:
                    raise
                logging.warning('skipping scenario %s', view.path, exc_info=True)
                continue
            for entry in entries:
                yield from _expand(entry)

    def save(self, world: World, map_only: bool = False, validate: bool = True, resolution: float = 0.05, **render: object) -> Path:
        """Write the world as canonical format 2 plus its KG, and a preview map per level under `map/`."""
        from arena_simulation_setup.kg import dump as dump_kg

        from .membership import own_kg

        if validate:
            world.validate()
        self.path.mkdir(parents=True, exist_ok=True)
        if not map_only:
            legacy_dirs = self._legacy_level_dirs()
            own = attrs.evolve(world, schedules=[item for item in world.schedules if not any(item is linked for linked in world.linked_scope)], signals=[item for item in world.signals if not any(item is linked for linked in world.linked_scope)])
            self.world_path.write_text(write_world(own), encoding='utf-8')
            self._drop_legacy_levels(legacy_dirs)
            if world.kg is not None:
                self.kg_path.write_text(to_yaml(dump_kg(own_kg(world.kg, world.linked_kg))), encoding='utf-8')
        map_dir = self.path / 'map'
        map_dir.mkdir(exist_ok=True)
        for level_id, level in world.levels.items():
            if not level.zones and not level.walls:
                continue
            png, origin = level.render(resolution=resolution, **typing.cast(dict[str, typing.Any], render))
            (map_dir / f'{level_id}.png').write_bytes(png)
            self.map_yaml(level_id).write_text(Map.generate_map_yaml(resolution=resolution, filename=f'{level_id}.png', origin=origin), encoding='utf-8')
        return self.path

    def migrate(self) -> list[str]:
        """Rewrite a format 1 world in place as format 2, removing the per-level files. Returns the upgrade warnings."""
        from arena_simulation_setup.kg import load as load_kg

        from .membership import prune

        data, kg, notes = self.read()
        world = self._parse((self.path.resolve(),), [], data)
        world.apply_elevator_door_sides()
        if kg is not None:
            world.kg = load_kg(kg)
            world.kg = prune(world)
        self.save(world)
        graph = self.path / LEGACY_GRAPH_FILE
        if graph.exists() and world.kg is not None:
            graph.unlink()
        for scenario in self.scenario.listall():
            notes.extend(f'scenario {scenario.name}: {note}' for note in scenario.resolve_sync().migrate())
        return notes

    def level_origins(self) -> dict[str, tuple[float, float]] | None:
        """Per-level offsets that flatten all levels into a single map frame, None for a single-level world."""
        world = self.load(validate=False)
        if len(world.levels) <= 1:
            return None
        return world.level_origins()


def save_snapshot(world: World, directory: Path, lighting: str = 'authored') -> Path:
    """Write `world` as it runs into `directory`: format 2 with its levels under `lighting` and linked levels written out, plus every semantic fact, stated and derived."""
    from arena_simulation_setup.kg import dump as dump_kg

    realized = attrs.evolve(world, levels={level_id: attrs.evolve(level.with_lighting(lighting), source='', prefix='') for level_id, level in world.levels.items()}, sources=[], linked_kg=None, linked_scope=[])
    directory.mkdir(parents=True, exist_ok=True)
    files = {'world.yaml': write_world(realized)}
    if realized.kg is not None:
        files[KG_FILE] = to_yaml(dump_kg(realized.kg))
    for name, text in files.items():
        partial = directory / f'.{name}.partial'
        partial.write_text(text, encoding='utf-8')
        partial.replace(directory / name)
    return directory


def _drop_foreign_destinations(world: World) -> None:
    cabins = {elevator.name for level in world.all_levels for elevator in level.elevators}
    for level in world.all_levels:
        if not level.source:
            continue
        for elevator in level.elevators:
            foreign = [name for name in elevator.destinations if name not in cabins]
            if foreign:
                logger.warning('level %s from %s: elevator %s loses destinations %s, they are not in this world', level.id, level.source, elevator.name, foreign)
                elevator.destinations = [name for name in elevator.destinations if name in cabins]


def check_kg(world: World, scene: 'Scene') -> None:
    """Raise when the KG names something the world does not have or carries geometry the world owns."""
    from arena_simulation_setup.kg import check_against_world

    names = world.names()
    problems = check_against_world(
        scene,
        {
            'room': names['zone'],
            'door': names['door'],
            'floor': world.level_ids,
            'elevator': names['elevator'],
            'obj': names['obj'],
            'light': names['light'],
            'sound': names['sound'],
            'microphone': [],
        },
    )
    problems.extend(scene.validate())
    if problems:
        raise ValueError(f'{KG_FILE}: ' + '; '.join(problems))


class WorldIdentifier(Identifier[WorldView]):
    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, WorldIdentifier) and self.name == other.name

    def load(self, path: Path, /, **kwargs: object) -> WorldView:
        del kwargs
        return WorldView(path)

    @classmethod
    def parse(cls, raw: str) -> 'tuple[str, set[str] | None]':
        """Parse a world identifier with optional level filter.

        ``'myworld'`` returns ``('myworld', None)``.
        ``'myworld[1,3]'`` returns ``('myworld', {'1', '3'})``.
        """
        if '[' not in raw:
            return raw, None
        name, _, rest = raw.partition('[')
        ids = {s.strip() for s in rest.rstrip(']').split(',') if s.strip()}
        return name, ids or None


def _world_search_roots() -> list[Path]:
    """Extra world roots from ARENA_WORLD_PATH (colon-separated), searched before the canonical tree."""
    return [Path(p) for p in os.environ.get('ARENA_WORLD_PATH', '').split(':') if p]


WorldIdentifier.use(*(SimplePathResolver(WorldIdentifier, root) for root in _world_search_roots()))
WorldIdentifier.use(SimplePathResolver(WorldIdentifier, ASS_DIR / 'worlds'))
WorldIdentifier.use(*NetResolver.all(WorldIdentifier, providers=WORLD_PROVIDERS, formats=(), annotated=False))
WorldIdentifier.use(FallbackResolver(WorldIdentifier, ASS_DIR / 'worlds'))
