from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast

import pydantic
import yaml

from arena_mcp.core import ROS, Domain, ToolError, media, tg

if TYPE_CHECKING:
    from arena_simulation_setup.tree.World import World
    from arena_simulation_setup.utils.generative import WorldGenerator, WorldGeneratorType

domain = Domain(
    "worldgen",
    """Arena's own world generators (empty, hallway, barn, sketch, letter, ...), run in process.
    Preview a generator with a config and seed, save the result as a world under the writable world root, and queue it on an env with its episode binding.
    worldgen_alphabet and worldgen_normalize describe the sketch grammar, the generative README in arena_simulation_setup has all of it.""",
    requires=("arena_simulation_setup", "shapely"),
)

BINDING_KEYS = frozenset({"tm_robots", "tm_obstacles", "robots_params", "obstacles_params"})

GeneratorName = Annotated[str, pydantic.Field(description="generator name, worldgen_generators lists them")]
Config = Annotated[dict[str, object] | None, pydantic.Field(description="generator configuration, fields from worldgen_generators, unset fields keep their defaults")]
Sketch = Annotated[str | None, pydantic.Field(description="sketch text for generators with a sketch field, overrides config['sketch']")]
Seed = Annotated[int, pydantic.Field(description="random seed, negative for a nondeterministic run")]
WorldName = Annotated[str, pydantic.Field(description="world name, one path component")]


def world_name(name: str) -> str:
    """The name when it is one safe path component, else a ToolError."""
    if not name or name in (".", "..") or name.startswith(".") or "/" in name or "\\" in name or "\0" in name:
        raise ToolError(f"world name {name!r} must be one path component, not empty and not starting with a dot", hint="use letters, digits, '_' or '-'")
    return name


def _generator(name: str) -> WorldGeneratorType:
    from arena_simulation_setup.utils.generative import WorldGenerator, WorldGeneratorType

    try:
        generator = WorldGeneratorType(name)
    except ValueError:
        generator = None
    if generator is None or generator not in WorldGenerator.available():
        raise ToolError(f"no generator {name!r}", hint=f"generators: {', '.join(sorted(item.value for item in WorldGenerator.available()))}")
    return generator


def _configuration(generator: WorldGeneratorType, config: dict[str, object] | None, sketch: str | None) -> dict[str, object]:
    from arena_simulation_setup.utils.generative import WorldGenerator

    model = WorldGenerator.config_model(generator)
    merged = dict(config or {})
    if sketch is not None:
        if "sketch" not in model.model_fields:
            raise ToolError(f"generator {generator.value!r} takes no sketch", hint="generators with a sketch field are marked in worldgen_generators")
        merged["sketch"] = sketch
    unknown = sorted(set(merged) - set(model.model_fields))
    if unknown:
        raise ToolError(f"generator {generator.value!r} has no field {', '.join(unknown)}", hint=f"fields: {', '.join(model.model_fields)}")
    try:
        model.model_validate(merged)
    except pydantic.ValidationError as exc:
        raise ToolError(f"invalid {generator.value} configuration: {exc}", hint="worldgen_generators shows each field's schema") from exc
    return merged


def _compute(generator: str, config: dict[str, object] | None, sketch: str | None, seed: int) -> tuple[WorldGenerator, World, dict[str, object]]:
    from arena_simulation_setup.utils.generative import WorldGenerator
    from arena_simulation_setup.utils.generative.layout import diagnostics_of

    kind = _generator(generator)
    configuration = _configuration(kind, config, sketch)
    started = time.perf_counter()
    try:
        active = WorldGenerator(kind, configuration, seed)
        world = active.compute()
    except (ValueError, yaml.YAMLError) as exc:
        raise ToolError(f"{kind.value} could not build a world: {exc}") from exc
    elapsed = (time.perf_counter() - started) * 1e3
    level = world.all_levels[0]
    diagnostics = diagnostics_of(level)
    report: dict[str, object] = {
        "generator": kind.value,
        "seed": seed,
        "diagnostics": {
            "components": diagnostics.components,
            "islands": diagnostics.islands,
            "zones": diagnostics.zones,
            "extent": [round(float(value), 3) for value in diagnostics.extent],
            "compile_ms": round(elapsed, 1),
        },
        "warnings": [{"row": note.row, "col": note.col, "text": note.text} for note in active.warnings],
        "episode_binding": active.params(),
    }
    if "sketch" in WorldGenerator.config_model(kind).model_fields:
        report["normalized"] = active.normalize(str(configuration.get("sketch", "")))
    frame = active.frame()
    if frame is not None:
        report["grid"] = {"origin": [float(frame.origin[0]), float(frame.origin[1])], "pitch": float(frame.pitch), "rows": frame.rows, "cols": frame.cols}
    return active, world, report


@domain.tool(read_only=True)
def worldgen_generators() -> list[dict[str, object]]:
    """Every registered generator with its configuration JSON schema and the episode binding its defaults pin."""
    from arena_simulation_setup.utils.generative import PedestrianConfiguration, WorldGenerator

    out: list[dict[str, object]] = []
    for kind in WorldGenerator.available():
        model = WorldGenerator.config_model(kind)
        out.append(
            {
                "name": kind.value,
                "config_schema": model.model_json_schema(),
                "takes_sketch": "sketch" in model.model_fields,
                "episode_binding": WorldGenerator(kind, {}, 0).params(),
                "binding_follows_pedestrians": issubclass(model, PedestrianConfiguration),
            }
        )
    return out


@domain.tool(read_only=True)
def worldgen_preview(generator: GeneratorName, config: Config = None, sketch: Sketch = None, seed: Seed = 0) -> list:
    """Run a generator in process: diagnostics, warnings by sketch row/col, normalized sketch, episode binding and the rendered map."""
    _, world, report = _compute(generator, config, sketch, seed)
    level = world.all_levels[0]
    diagnostics = cast("dict[str, object]", report["diagnostics"])
    extent = cast("list[float]", diagnostics["extent"])
    resolution = max(0.05, max(extent) / media.MAX_SIDE)
    png, origin = level.render(resolution=resolution)
    report["map"] = {"resolution": resolution, "origin": [float(origin[0]), float(origin[1])]}
    return [report, media.png(png)]


@domain.tool(destructive=True)
def worldgen_save(name: WorldName, generator: GeneratorName, config: Config = None, sketch: Sketch = None, seed: Seed = 0, overwrite: bool = False) -> dict[str, object]:
    """Generate and save a world under the writable world root. The result carries the path, diagnostics and the episode binding for worldgen_queue."""
    from arena_simulation_setup.tree.World import WorldIdentifier
    from arena_simulation_setup.utils.generative.world_generator import save_generated

    view = WorldIdentifier(world_name(name)).resolve_write_sync()
    target = Path(view.path)
    if target.exists() and any(target.iterdir()) and not overwrite:
        raise ToolError(f"world {name!r} already exists at {target}", hint="pass overwrite=True to replace it, or pick another name")
    active, world, report = _compute(generator, config, sketch, seed)
    if target.exists() and overwrite:
        shutil.rmtree(target)
    try:
        path = save_generated(view, world, active.files())
    except ValueError as exc:
        raise ToolError(f"could not save {name!r}: {exc}") from exc
    return {"world": name, "path": str(path), **report}


@domain.tool(read_only=True)
def worldgen_alphabet() -> dict[str, object]:
    """The sketch glyph table as arm weights per direction, the ASCII and fill aliases and the void characters."""
    from arena_simulation_setup.utils.generative import alphabet
    from arena_simulation_setup.utils.generative.sketch import WEIGHTS

    names = {alphabet.NONE: "none", **{value: key for key, value in WEIGHTS.items()}}

    def entry(glyph: str, arms: tuple[int, ...]) -> dict[str, object]:
        return {"glyph": glyph, "arms": {direction: names[weight] for direction, weight in zip(alphabet.DIRECTIONS, arms, strict=True) if weight != alphabet.NONE}}

    return {
        "directions": list(alphabet.DIRECTIONS),
        "weights": [names[weight] for weight in sorted(names)],
        "glyphs": [entry(glyph, arms) for glyph, arms in alphabet.entries()],
        "aliases": [{**entry(char, alphabet.arms_of(char) or alphabet.VOID), "canonical": alphabet.normalize(char)} for char in (*alphabet.ASCII_ALIASES, *alphabet.FILL_ALIASES)],
        "void_chars": "".join(alphabet.VOID_CHARS),
    }


@domain.tool(read_only=True)
def worldgen_normalize(sketch: str) -> dict[str, object]:
    """A sketch rewritten into canonical glyphs, plus the grid cells holding characters that are neither glyphs nor legend symbols."""
    from arena_simulation_setup.utils.generative import alphabet
    from arena_simulation_setup.utils.generative.sketch import normalize, split_sketch

    try:
        directives, rows = split_sketch(sketch)
    except (ValueError, yaml.YAMLError) as exc:
        raise ToolError(f"cannot read the sketch directives: {exc}") from exc
    unknown = [{"row": row, "col": col, "char": char} for row, line in enumerate(rows) for col, char in enumerate(line) if char not in directives.legend and alphabet.arms_of(char) is None]
    return {"sketch": normalize(sketch), "unknown": unknown}


@domain.tool(destructive=True)
async def worldgen_queue(name: WorldName, env_id: int | None = None, binding: Annotated[dict[str, object] | None, pydantic.Field(description="episode binding from worldgen_save: tm_robots, tm_obstacles, robots_params, obstacles_params")] = None) -> dict[str, object]:
    """Queue a world on an env for its next episode reset with the generator's episode binding, as the world generator panel does."""
    from rcl_interfaces.msg import Parameter
    from task_generator_msgs.srv import QueueEpisode

    from arena_mcp.core.ros import python_to_param

    binding = dict(binding or {})
    unknown = sorted(set(binding) - BINDING_KEYS)
    if unknown:
        raise ToolError(f"episode binding has no key {', '.join(unknown)}", hint=f"keys: {', '.join(sorted(BINDING_KEYS))}")

    def leaves(key: str) -> list[Parameter]:
        value = binding.get(key) or {}
        if not isinstance(value, dict):
            raise ToolError(f"binding {key} must map parameter leaves to values, got {type(value).__name__}")
        return [Parameter(name=str(leaf), value=python_to_param(item, 0)) for leaf, item in value.items()]

    request = QueueEpisode.Request(
        action=QueueEpisode.Request.MERGE,
        keep_modules=True,
        world=world_name(name),
        tm_robots=str(binding.get("tm_robots") or ""),
        tm_obstacles=str(binding.get("tm_obstacles") or ""),
        robots_params=leaves("robots_params"),
        obstacles_params=leaves("obstacles_params"),
    )
    service = f"{await tg(env_id)}/config/queue_episode"
    response = await ROS.call(QueueEpisode, service, request, hint="is the env's task generator up?")
    if not response.success:
        raise ToolError(f"queue_episode refused {name!r}: {response.error_msg}")
    return {"queued": name, "service": service, "episode_binding": binding, "applies": "at the next episode reset"}
