from __future__ import annotations

import fnmatch
import re
import tempfile
import typing
import warnings
from pathlib import Path
from typing import Annotated

import yaml
from mcp.server.mcpserver import Image
from pydantic import Field

from arena_mcp.core import Domain, ToolError, media

if typing.TYPE_CHECKING:
    from arena_simulation_setup.kg import Scene
    from arena_simulation_setup.tree.World import Level, World, WorldView

domain = Domain(
    "worlds",
    """Worlds, scenarios, knowledge graphs and the task grammar of arena_simulation_setup, offline on the files.
    A world name resolves like the runtime: ARENA_WORLD_PATH roots, the writable local root $ARENA_ASSETS_DIR_LOCAL/worlds, the installed worlds, then the world buckets.
    A name may carry a level filter, name[0,1], or compose worlds as one, a,b[1]. worlds_write and worlds_scenario_write only write into the writable local root.""",
)

NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
PREVIEW_PIXELS = 1500
MIN_RESOLUTION = 0.05

WorldName = Annotated[str, Field(description="world name as the runtime resolves it, optionally with a level filter name[0,1], or a composition a,b[1]")]
LevelId = Annotated[str | None, Field(description="one level id, all levels side by side when omitted")]


def _resolve(name: str) -> tuple[WorldView, set[str] | None]:
    from arena_simulation_setup.metrics.nav_at_r import resolve_world

    try:
        return resolve_world(name)
    except (FileNotFoundError, ValueError) as exc:
        raise ToolError(f"world {name!r} does not resolve: {str(exc).splitlines()[0]}", hint="worlds_list shows the resolvable worlds, worlds_find every resolver's verdict") from exc


def _format(view: WorldView) -> str:
    from arena_simulation_setup.tree.World import FORMAT
    from arena_simulation_setup.tree.World.legacy import is_legacy

    if view.world_path.exists() and not is_legacy(yaml.safe_load(view.world_path.read_text(encoding="utf-8"))):
        return FORMAT
    return "arena-world/1"


def _check(world: World, filtered: bool) -> dict[str, object]:
    from arena_simulation_setup.tree.World.view import check_kg

    try:
        notes = world.validate(loaded_level_ids=set(world.levels) if filtered else None)
        if world.kg is not None:
            check_kg(world, world.kg)
    except Exception as exc:
        return {"valid": False, "reason": str(exc)}
    return {"valid": True, "warnings": notes}


def _load(view: WorldView, levels: set[str] | None, derive: str = "auto") -> tuple[World, dict[str, object]]:
    """The world as the runtime loads it, with the strict verdict taken before the KG is derived."""
    from arena_simulation_setup.tree.World import derive as derive_kg

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        world = view.load(validate=False, level_filter=levels, derive="off")
    if levels is not None:
        missing = sorted(levels - set(world.levels))
        if missing:
            raise ToolError(f"world {view.name!r} has no level {', '.join(missing)}", hint="worlds_inspect without a level filter lists the levels")
    verdict = _check(world, levels is not None)
    try:
        world.kg = derive_kg(world, derive)
    except ValueError as exc:
        raise ToolError(str(exc), hint="derive is auto, all or off") from exc
    return world, {"format": _format(view), **verdict}


def _loaded(name: str, derive: str = "auto") -> tuple[WorldView, World, dict[str, object]]:
    from arena_simulation_setup.tree.World import WorldFormatError

    view, levels = _resolve(name)
    try:
        world, verdict = _load(view, levels, derive)
    except (WorldFormatError, FileNotFoundError, ValueError) as exc:
        raise ToolError(f"world {name!r} does not load: {exc}", hint="worlds_validate shows the strict verdict") from exc
    return view, world, verdict


def _extent(level: Level) -> dict[str, list[float]] | None:
    points = [point for zone in level.zones for point in zone.polygon] + [point for path in level.walls for point in path.path]
    if not points:
        return None
    low = [min(point.x for point in points), min(point.y for point in points)]
    high = [max(point.x for point in points), max(point.y for point in points)]
    return {"min": [round(v, 3) for v in low], "max": [round(v, 3) for v in high], "size": [round(high[0] - low[0], 3), round(high[1] - low[1], 3)]}


def _render(level: Level) -> tuple[Image, dict[str, object]] | None:
    """The level drawn by the renderer touch_world and WorldView.save use, at a resolution that keeps it within PREVIEW_PIXELS."""
    extent = _extent(level)
    if extent is None:
        return None
    resolution = max(MIN_RESOLUTION, max(extent["size"]) / PREVIEW_PIXELS)
    png, origin = level.render(resolution=resolution)
    return media.png(png), {"level": level.id, "resolution": resolution, "origin": [float(origin[0]), float(origin[1])]}


def _kg_summary(scene: Scene | None, view: WorldView) -> dict[str, object] | None:
    from arena_simulation_setup.kg import DERIVED
    from arena_simulation_setup.tree.World import connections

    if scene is None:
        return None
    return {
        "file": str(view.kg_path) if view.kg_path.exists() else None,
        "uses": scene.uses,
        "entities": len(scene.entities),
        "stated_facts": len(scene.stated()),
        "derived_facts": sum(1 for fact in scene.facts if fact.source == DERIVED),
        "constraints": len(scene.constraints),
        "connections": [list(connection) for connection in connections(scene)],
    }


def _level_summary(level: Level) -> dict[str, object]:
    names = {
        "zones": [zone.name for zone in level.zones],
        "doors": [door.name for door in level.doors],
        "elevators": [elevator.name for elevator in level.elevators],
        "objects": [entity.name for entity in level.objects],
        "humans": [human.name for human in level.humans],
        "lights": [light.name for light in level.lights],
        "sounds": [sound.name for sound in level.sounds],
        "microphones": [microphone.listener_id for microphone in level.microphones],
    }
    return {
        "id": level.id,
        "linked_from": level.source or None,
        "prefix": level.prefix or None,
        "extent": _extent(level),
        "wall_paths": len(level.walls),
        "counts": {kind: len(items) for kind, items in names.items()},
        **names,
    }


def _summary(view: WorldView, world: World, verdict: dict[str, object]) -> dict[str, object]:
    names = world.names()
    return {
        "world": view.name,
        "path": str(view.path),
        **verdict,
        "linked_worlds": [str(source) for source in world.sources],
        "levels": [_level_summary(level) for level in world.levels.values()],
        "schedules": names["schedule"],
        "signals": names["signal"],
        "kg": _kg_summary(world.kg, view),
        "scenarios": sorted(scenario.name for scenario in view.scenario.listall()),
    }


def _nouns(world: World) -> dict[str, str]:
    from arena_simulation_setup.tree.World import describe

    if world.kg is None:
        return {}
    return {zone.name: text for level in world.levels.values() for zone in level.zones if (text := describe(world.kg, zone.name))}


def _component(value: str, what: str) -> None:
    if not NAME.fullmatch(value):
        raise ToolError(f"{what} {value!r} is not one path component of letters, digits, '_', '.' and '-'", hint=f"pick a plain {what}")


def _writable(view: WorldView) -> None:
    from arena_simulation_setup.tree import local_root

    root = (local_root() / "worlds").resolve()
    if not view.path.resolve().is_relative_to(root):
        raise ToolError(f"world {view.name!r} lives at {view.path}, outside the writable root {root}", hint="write a copy into the writable root with worlds_write first")


@domain.tool(read_only=True, open_world=True)
def worlds_list(
    network: Annotated[bool, Field(description="also list the world buckets, a remote listing cached for ARENA_MODELS_TTL")] = False,
    pattern: Annotated[str | None, Field(description="glob over world names, e.g. hospital*")] = None,
) -> list[dict[str, object]]:
    """Every resolvable world with the resolver it resolves from and the later ones it shadows."""
    from arena_simulation_setup.tree import PathResolverBase
    from arena_simulation_setup.tree.World import WorldIdentifier
    from arena_simulation_setup.tree.World.view import ComposedResolver

    rows: dict[str, dict[str, object]] = {}
    shadows: dict[str, list[str]] = {}
    for resolver in WorldIdentifier._resolvers:
        if isinstance(resolver, ComposedResolver):
            names = sorted(entry.name for entry in resolver.root.iterdir() if entry.is_dir()) if resolver.root.is_dir() else []
        else:
            names = [identifier.name for identifier in resolver.listall(network=network)]
        for name in names:
            if pattern is not None and not fnmatch.fnmatchcase(name, pattern):
                continue
            if name in rows:
                shadows[name].append(repr(resolver))
                continue
            path = resolver.root / name if isinstance(resolver, ComposedResolver) else resolver.path / name if isinstance(resolver, PathResolverBase) else None
            shadows[name] = []
            rows[name] = {"name": name, "resolver": repr(resolver), "path": str(path) if path is not None and path.exists() else None, "shadows": shadows[name]}
    return [rows[name] for name in sorted(rows)]


@domain.tool(read_only=True, open_world=True)
def worlds_find(name: WorldName) -> dict[str, object]:
    """Every world resolver's verdict for one name (hit, miss, shadowed, stale), without downloading, like `arena asset find world`."""
    from arena_simulation_setup.tree import Verdict
    from arena_simulation_setup.tree.World import WorldIdentifier

    base, levels = WorldIdentifier.split_filter(name)
    try:
        verdicts = WorldIdentifier(base).probe_sync()
    except (FileNotFoundError, ValueError) as exc:
        raise ToolError(f"world {name!r} cannot be probed: {exc}", hint="a composition names worlds that resolve, worlds_find each part") from exc
    return {
        "world": base,
        "levels": sorted(levels) if levels is not None else None,
        "resolves": any(verdict.verdict in (Verdict.HIT, Verdict.STALE) for verdict in verdicts),
        "verdicts": [{"resolver": repr(verdict.resolver), "verdict": verdict.verdict.value, "path": str(verdict.path) if verdict.path is not None else None} for verdict in verdicts],
    }


@domain.tool(read_only=True)
def worlds_inspect(name: WorldName) -> list:
    """Levels with extent and the names of their zones, doors, elevators, objects, humans, lights, sounds and microphones, schedules, signals, KG summary, scenarios, strict verdict, then a preview image per level."""
    view, world, verdict = _loaded(name)
    summary = _summary(view, world, verdict)
    images: list[Image] = []
    previews: list[dict[str, object]] = []
    for level in world.levels.values():
        rendered = _render(level)
        if rendered is not None:
            images.append(rendered[0])
            previews.append(rendered[1])
    return [{**summary, "previews": previews}, *images]


@domain.tool(read_only=True)
def worlds_preview(name: WorldName, level: LevelId = None) -> list:
    """The rendered map of one level, or of every level laid side by side the way touch_world draws map.png."""
    view, world, _ = _loaded(name)
    if level is not None:
        if level not in world.levels:
            raise ToolError(f"world {view.name!r} has no level {level!r}, only {', '.join(world.levels)}", hint="pass one of those ids, or omit level")
        target = world.levels[level]
    else:
        origins = world.level_origins() if len(world.levels) > 1 else dict.fromkeys(world.level_ids, (0.0, 0.0))
        target = world.compact_world(origins)
        target.id = ",".join(world.level_ids)
    rendered = _render(target)
    if rendered is None:
        raise ToolError(f"world {view.name!r} has no zones or walls to draw", hint="worlds_inspect shows what the levels hold")
    image, info = rendered
    return [{"world": view.name, **info}, image]


@domain.tool(read_only=True)
def worlds_validate(
    name: Annotated[str | None, Field(description="world name as the runtime resolves it, optionally with a level filter")] = None,
    path: Annotated[str | None, Field(description="a world directory or its world.yaml")] = None,
    yaml_text: Annotated[str | None, Field(description="the text of a world.yaml, validated in isolation, so ./ links do not resolve")] = None,
) -> dict[str, object]:
    """The strict verdict on one world given by name, directory or text: format 2 parse and validation errors, or the legacy loader's verdict for format 1."""
    from arena_simulation_setup.tree.World import WorldView

    if sum(value is not None for value in (name, path, yaml_text)) != 1:
        raise ToolError("pass exactly one of name, path and yaml_text")
    if name is not None:
        view, levels = _resolve(name)
        return _validate(view, levels)
    if path is not None:
        target = Path(path).expanduser()
        if target.is_file():
            target = target.parent
        if not target.is_dir():
            raise ToolError(f"{path} is not a world directory or world.yaml", hint="pass the directory holding world.yaml")
        return _validate(WorldView(target.resolve()), None)
    with tempfile.TemporaryDirectory() as staging:
        view = WorldView(Path(staging) / "world")
        view.path.mkdir()
        view.world_path.write_text(typing.cast(str, yaml_text), encoding="utf-8")
        return {**_validate(view, None), "path": None}


def _validate(view: WorldView, levels: set[str] | None) -> dict[str, object]:
    from arena_simulation_setup.tree.World import WorldFormatError

    try:
        world, verdict = _load(view, levels, "off")
    except (WorldFormatError, FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        return {"world": view.name, "path": str(view.path), "valid": False, "reason": str(exc)}
    return {"world": view.name, "path": str(view.path), **verdict, "levels": world.level_ids}


@domain.tool(destructive=True)
def worlds_migrate(name: Annotated[str, Field(description="a world name, rewritten in place where it resolves")]) -> dict[str, object]:
    """Rewrite a format 1 world in place as format 2, deleting the per-level files, and return the upgrade warnings."""
    from arena_simulation_setup.tree.World import FORMAT

    view, _ = _resolve(name)
    if _format(view) == FORMAT:
        return {"world": view.name, "path": str(view.path), "migrated": False, "reason": "already format 2"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            notes = view.migrate()
        except (FileNotFoundError, ValueError) as exc:
            raise ToolError(f"world {name!r} does not migrate: {exc}", hint="worlds_validate shows the legacy loader's verdict") from exc
    return {"world": view.name, "path": str(view.path), "migrated": True, "warnings": notes}


@domain.tool(destructive=True)
def worlds_write(
    name: Annotated[str, Field(description="new world name, one path component")],
    yaml_text: Annotated[str, Field(description="format 2 world.yaml text")],
    overwrite: Annotated[bool, Field(description="replace an existing world.yaml of that name in the writable root")] = False,
) -> dict[str, object]:
    """Validate a format 2 world and save it with its preview maps into the writable root $ARENA_ASSETS_DIR_LOCAL/worlds/<name>."""
    from arena_simulation_setup.tree.World import FORMAT, WorldIdentifier
    from arena_simulation_setup.tree.World.legacy import is_legacy

    _component(name, "world name")
    try:
        data = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise ToolError(f"yaml_text is not YAML: {exc}") from exc
    if is_legacy(data):
        raise ToolError("yaml_text is format 1", hint=f"write format: {FORMAT}, worlds_validate checks a draft")
    view = WorldIdentifier(name).resolve_write_sync()
    previous = view.world_path.read_text(encoding="utf-8") if view.world_path.exists() else None
    if previous is not None and not overwrite:
        raise ToolError(f"world {name!r} already exists at {view.path}", hint="pass overwrite=true to replace it, or pick another name")
    created = not view.path.exists()
    view.path.mkdir(parents=True, exist_ok=True)
    view.world_path.write_text(yaml_text, encoding="utf-8")
    try:
        world, verdict = _load(view, None)
    except Exception as exc:
        verdict = {"valid": False, "reason": str(exc)}
    if not verdict["valid"]:
        if previous is not None:
            view.world_path.write_text(previous, encoding="utf-8")
        else:
            view.world_path.unlink()
            if created and not any(view.path.iterdir()):
                view.path.rmdir()
        raise ToolError(f"world {name!r} is invalid, nothing was written: {verdict['reason']}", hint="fix the reported entry, worlds_validate checks a draft")
    maps = view.path / "map"
    if maps.is_dir():
        for file in maps.iterdir():
            if file.suffix in (".png", ".yaml") and file.stem not in {str(level) for level in world.level_ids}:
                file.unlink()
    view.save(world, map_only=True)
    resolved = WorldIdentifier(name).resolve_sync().path
    return {"world": name, "path": str(view.path), "levels": world.level_ids, "replaced": previous is not None, "resolves_to": str(resolved), "shadowed": resolved.resolve() != view.path.resolve(), "warnings": verdict["warnings"]}


@domain.tool(read_only=True, open_world=True)
def worlds_nav_at_r(
    name: WorldName,
    robot: Annotated[str | None, Field(description="robot name in arena_robots or a robot directory, its caps/mobile.yaml radius is r")] = None,
    radius: Annotated[float | None, Field(description="footprint radius r in meters, the Jackal radius 0.267 when neither robot nor radius is given")] = None,
    walls_only: Annotated[bool, Field(description="ignore static entities, to tell layout faults from furniture blockages")] = False,
    start: Annotated[list[float] | None, Field(description="[x, y] in meters in each level's frame, adds the single-start share")] = None,
) -> list:
    """NAV@r per level (share of the traversable space a random spawn reaches, unreachable zones), with the component map per level."""
    import numpy as np
    from PIL import Image as PilImage

    from arena_simulation_setup.metrics.nav_at_r import DEFAULT_RADIUS, nav_at_r, robot_radius

    if robot is not None and radius is not None:
        raise ToolError("pass robot or radius, not both")
    if radius is not None and not radius > 0:
        raise ToolError(f"radius is a positive length in meters, got {radius}")
    if start is not None and len(start) != 2:
        raise ToolError(f"start is [x, y], got {start}")
    try:
        r = robot_radius(robot) if robot is not None else radius if radius is not None else DEFAULT_RADIUS
    except LookupError as exc:
        raise ToolError(str(exc), hint="pass radius instead") from exc
    view, levels = _resolve(name)
    with tempfile.TemporaryDirectory() as scratch:
        image = Path(scratch) / "nav.png"
        results = nav_at_r(view, radius=r, walls_only=walls_only, start=None if start is None else (start[0], start[1]), levels=levels, image=image)
        if not results:
            raise ToolError(f"world {name!r} has no level matching the filter", hint="worlds_inspect lists the levels")
        files = [image] if len(results) == 1 else [image.with_name(f"{image.stem}_{level_id}{image.suffix}") for level_id in results]
        images = [media.array_png(np.asarray(PilImage.open(file).convert("RGB"))) for file in files]
    legend = "dominant component green, stranded components in other colors, free cells too tight for r light grey, entities brown, walls black"
    return [{"world": view.name, "radius": r, "walls_only": walls_only, "levels": [result.asdict() for result in results.values()], "images": legend}, *images]


@domain.tool(read_only=True)
def worlds_kg(
    name: WorldName,
    derive: Annotated[str, Field(description="auto adds the contains and connects facts geometry implies where no stated fact covers them, all also replaces the stated ones geometry can derive, off keeps only the stated facts")] = "auto",
) -> dict[str, object]:
    """The world's knowledge graph as the runtime sees it: entities, stated and derived facts, constraints, plus door connections and zone contents."""
    from arena_simulation_setup.kg import dump
    from arena_simulation_setup.tree.World import contents, describe

    view, world, _ = _loaded(name, derive)
    scene = typing.cast("Scene", world.kg)
    zones = [zone.name for level in world.levels.values() for zone in level.zones]
    return {
        "world": view.name,
        "summary": _kg_summary(scene, view),
        "zones": {zone: {"description": describe(scene, zone), "contains": contents(scene, zone)} for zone in zones},
        "kg": dump(scene),
    }


@domain.tool(read_only=True)
def worlds_kg_check(name: WorldName) -> dict[str, object]:
    """Problems of the world's stated kg.yaml against the names world.yaml declares and against its vocabulary."""
    from arena_simulation_setup.tree.World.view import KG_FILE, check_kg

    view, world, _ = _loaded(name, "off")
    problems: list[str] = []
    if world.kg is not None:
        try:
            check_kg(world, world.kg)
        except ValueError as exc:
            problems = str(exc).removeprefix(f"{KG_FILE}: ").split("; ")
    return {"world": view.name, "kg": str(view.kg_path) if view.kg_path.exists() else None, "problems": problems}


@domain.tool(read_only=True)
def worlds_vocabularies(uses: Annotated[list[str] | None, Field(description="packages to combine, e.g. ['core@2', 'social'], for their union and the kg.yaml JSON schema over it")] = None) -> dict[str, object]:
    """The KG vocabulary packages (node kinds, verbs, constraint kinds), or the union of the given ones with its kg.yaml JSON schema."""
    from arena_simulation_setup.kg import Vocabulary, schema_for
    from arena_simulation_setup.kg.vocab import available

    if uses is None:
        return {"packages": [{"package": package.key, "requires": list(package.requires), "nodes": dict(package.nodes), "verbs": dict(package.verbs), "constraints": dict(package.constraints)} for package in available().values()]}
    try:
        vocab = Vocabulary(uses)
    except ValueError as exc:
        raise ToolError(str(exc), hint="worlds_vocabularies without uses lists the packages") from exc
    return {"uses": vocab.uses, "nodes": vocab.nodes, "verbs": vocab.verbs, "constraints": vocab.constraints, "schema": schema_for(vocab)}


@domain.tool(read_only=True)
def worlds_scenarios(name: WorldName) -> list[dict[str, object]]:
    """The scenarios of a world with their file and format."""
    view, _ = _resolve(name)
    rows = []
    for identifier in sorted(view.scenario.listall(), key=lambda identifier: identifier.name):
        scenario = identifier.resolve_sync()
        try:
            raw = yaml.safe_load(Path(scenario.scenario_path).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            rows.append({"scenario": identifier.name, "path": scenario.scenario_path, "format": None, "reason": str(exc)})
            continue
        rows.append({"scenario": identifier.name, "path": scenario.scenario_path, "format": raw.get("format", "arena-scenario/1") if isinstance(raw, dict) else None})
    return rows


def _scenario_verdict(raw: object, world: World | None) -> dict[str, object]:
    """Parse a format 2 scenario mapping, its zone references sampled from the world's zones when one is given."""
    import numpy as np

    from arena_simulation_setup.shared.render import render_phases
    from arena_simulation_setup.tree.World.Scenario import parse_scenario
    from arena_simulation_setup.utils.cattrs import converter

    convert = converter if world is None else world.compact_world(dict.fromkeys(world.level_ids, (0.0, 0.0))).zone_converter(np.random.default_rng(0))
    try:
        scenario = parse_scenario(raw, convert)
    except Exception as exc:
        reason = str(exc)
        if world is None:
            reason += " (zone references only resolve when the world is given)"
        return {"valid": False, "reason": reason}
    problems = []
    if world is not None:
        for index, robot in enumerate(scenario.robots):
            problems.extend(f"robots[{index}].{key}: no level {value!r}, only {', '.join(world.level_ids)}" for key, value in (("level", robot.level), ("goal_level", robot.goal_level)) if value and value not in world.levels)
    if problems:
        return {"valid": False, "reason": "; ".join(problems)}
    nouns = _nouns(world) if world is not None else {}
    return {
        "valid": True,
        "counts": {"objects": len(scenario.objects), "humans": len(scenario.humans), "sounds": len(scenario.sounds), "lights": len(scenario.lights), "robots": len(scenario.robots), "regions": len(scenario.regions), "timeline": len(scenario.timeline), "conditions": len(scenario.conditions)},
        "instructions": [render_phases(robot.phases, nouns, robot.conditions) for robot in scenario.robots],
    }


@domain.tool(read_only=True)
def worlds_scenario(name: WorldName, scenario: Annotated[str, Field(description="scenario name, as worlds_scenarios lists it")]) -> dict[str, object]:
    """One scenario: its format 2 mapping (format 1 upgraded), the verdict against its world and each robot's task as text."""
    view, world, _ = _loaded(name)
    try:
        found = view.scenario(scenario).resolve_sync()
    except FileNotFoundError as exc:
        raise ToolError(f"world {view.name!r} has no scenario {scenario!r}", hint="worlds_scenarios lists them") from exc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        raw = found.read()
    return {"world": view.name, "scenario": scenario, "path": found.scenario_path, **_scenario_verdict(raw, world), "content": raw}


@domain.tool(read_only=True)
def worlds_scenario_validate(
    yaml_text: Annotated[str, Field(description="scenario.yaml text, format 2 or format 1 (then upgraded)")],
    world: Annotated[str | None, Field(description="world whose zones, doors and elevators the scenario's zone references must name")] = None,
) -> dict[str, object]:
    """The verdict on a scenario text, its zone references checked against the world when given, with the upgrade warnings of a format 1 text."""
    from arena_simulation_setup.tree.World.Scenario import upgrade_scenario

    try:
        raw = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        return {"valid": False, "reason": f"not YAML: {exc}"}
    legacy = isinstance(raw, dict) and "format" not in raw
    notes: list[str] = []
    if legacy:
        raw, notes = upgrade_scenario(raw)
    loaded = _loaded(world)[1] if world is not None else None
    return {**_scenario_verdict(raw, loaded), "upgrade_warnings": notes, "upgraded": raw if legacy else None}


@domain.tool(destructive=True)
def worlds_scenario_write(
    world: Annotated[str, Field(description="a world in the writable root")],
    scenario: Annotated[str, Field(description="scenario name, one path component")],
    yaml_text: Annotated[str, Field(description="format 2 scenario.yaml text")],
    overwrite: Annotated[bool, Field(description="replace an existing scenario of that name")] = False,
) -> dict[str, object]:
    """Validate a format 2 scenario against its world and save it as scenarios/<scenario>/scenario.yaml of a world in the writable root."""
    from arena_simulation_setup.tree.World.Scenario import SCENARIO_FORMAT

    _component(scenario, "scenario name")
    view, loaded, _ = _loaded(world)
    _writable(view)
    try:
        raw = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise ToolError(f"yaml_text is not YAML: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("format") != SCENARIO_FORMAT:
        raise ToolError(f"yaml_text is not a {SCENARIO_FORMAT} mapping", hint="worlds_scenario_validate upgrades a format 1 text and shows the result")
    verdict = _scenario_verdict(raw, loaded)
    if not verdict["valid"]:
        raise ToolError(f"scenario {scenario!r} is invalid, nothing was written: {verdict['reason']}", hint="worlds_scenario_validate checks a draft")
    target = view.scenario(scenario).resolve_write_sync()
    existing = [name for name in ("scenario.yaml", "scenario.json") if (target.path / name).exists()]
    if existing and not overwrite:
        raise ToolError(f"scenario {scenario!r} already exists at {target.path}", hint="pass overwrite=true to replace it")
    target.path.mkdir(parents=True, exist_ok=True)
    for name in existing:
        (target.path / name).unlink()
    (target.path / "scenario.yaml").write_text(yaml_text, encoding="utf-8")
    return {"world": view.name, "scenario": scenario, "path": str(target.path / "scenario.yaml"), "replaced": bool(existing), **verdict}


@domain.tool(read_only=True)
def worlds_task_parse(task_yaml: Annotated[str, Field(description="a TaskRequest as YAML: {phases: [...], conditions: [...]} or a bare list of phases")]) -> dict[str, object]:
    """Parse a robot task (phases with until, conditions and text), report the first error, or return the canonical form and its instruction text."""
    from arena_simulation_setup.shared.conditions import parse_until, render_until
    from arena_simulation_setup.shared.render import render_condition, render_phase, render_phases
    from arena_simulation_setup.shared.task import TaskRequest

    try:
        request = TaskRequest.parse(yaml.safe_load(task_yaml) or [])
    except Exception as exc:
        return {"valid": False, "reason": str(exc)}
    return {
        "valid": True,
        "kind": request.kind,
        "request": request.serialize(),
        "phases": [{"type": type(phase).__name__, "until": render_until(parse_until(phase.until)) if phase.until is not None else None, "text": render_phase(phase)} for phase in request.phases],
        "conditions": [render_condition(condition) for condition in request.conditions],
        "text": render_phases(request.phases, None, request.conditions),
    }


def _atom(atom: object) -> dict[str, object]:
    import attrs

    from arena_simulation_setup.shared.conditions import NotAtom

    if isinstance(atom, NotAtom):
        return {"form": "not", "atom": _atom(atom.atom)}
    return {"form": type(atom).__name__.removesuffix("Atom").lower(), **attrs.asdict(atom)}


@domain.tool(read_only=True)
def worlds_condition_parse(text: Annotated[str, Field(description="an until expression or atom, e.g. 'robot in kitchen and robot sees alice within 3 for 2', or an episode condition as YAML {op, p, q}")]) -> dict[str, object]:
    """Parse an until expression or an episode condition, report the first error, or return its structure, canonical form and sentence."""
    from arena_simulation_setup.shared.conditions import EpisodeCondition, parse_until, render_until
    from arena_simulation_setup.shared.render import render_condition, render_until_text

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        data = None
    try:
        if isinstance(data, dict):
            condition = EpisodeCondition.parse(data)
            return {"valid": True, "kind": "condition", "condition": condition.serialize(), "text": render_condition(condition)}
        until = parse_until(text)
    except Exception as exc:
        return {"valid": False, "reason": str(exc)}
    return {
        "valid": True,
        "kind": "until",
        "clauses": [{"atoms": [_atom(atom) for atom in clause.atoms], "dwell": clause.dwell} for clause in until.clauses],
        "canonical": render_until(until),
        "text": render_until_text(until),
    }


@domain.resource("arena://worlds/{name}", mime_type="application/json", description="One world's inspect summary without images")
def world_resource(name: str) -> dict[str, object]:
    view, world, verdict = _loaded(name)
    return _summary(view, world, verdict)
