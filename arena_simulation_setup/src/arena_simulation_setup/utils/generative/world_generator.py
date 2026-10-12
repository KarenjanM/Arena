import sys
from collections.abc import Mapping
from pathlib import Path

from arena_simulation_setup.tree.World import World, WorldIdentifier, WorldView

from . import WorldGenerator, WorldGeneratorType

__all__ = ['WorldGenerator', 'WorldGeneratorType', 'save_generated']


def save_generated(view: WorldView, world: World, files: Mapping[str, bytes]) -> Path:
    """Save a generated world in format 2 plus the generator's auxiliary files by relative path."""
    root = view.save(world).resolve()
    for name, content in files.items():
        target = (root / name).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f'generated file {name!r} lies outside the world directory')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return view.path


def test_generate(out: str, name: str, config: dict) -> Path:
    gen = WorldGenerator(WorldGeneratorType(name), config)
    return save_generated(WorldIdentifier(out).resolve_write_sync(), gen.compute(), gen.files())


def main(argv: list[str] = sys.argv) -> None:
    import json
    import os

    if len(argv) == 3:
        result = test_generate(argv[1], argv[2], {})
        print(f'Generated world saved to {result}')
    elif len(argv) == 4:
        result = test_generate(argv[1], argv[2], json.loads(argv[3]))
        print(f'Generated world saved to {result}')
    else:
        print(f'usage: {os.path.basename(__file__)} <world_name> <generator> [<config>]')
        sys.exit(1)


if __name__ == '__main__':
    main()
