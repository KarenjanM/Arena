from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'preload_world'


def _model(world: Path, name: str) -> Path:
    model = world / 'assets' / 'Common' / 'Object' / name
    model.mkdir(parents=True)
    (model / 'annotation.yaml').write_text(yaml.safe_dump({'name': name, 'path': f'Object/{name}'}), encoding='utf-8')
    return model


def _world(directory: Path, *levels: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'world.yaml').write_text(yaml.safe_dump({'format': 'arena-world/2', 'levels': list(levels)}), encoding='utf-8')
    return directory


def test_world_local_and_linked_assets_resolve_from_their_worlds(tmp_path: Path) -> None:
    floor = _world(tmp_path / 'preload_house' / 'floors' / 'ground', {'id': '0', 'zones': [{'name': 'hall', 'polygon': [[0.0, 0.0], [6.0, 0.0], [6.0, 4.0], [0.0, 4.0]]}], 'objects': [{'name': 'shelf', 'model': 'Preload_Linked_Shelf', 'pose': [1.0, 1.0]}]})
    house = _world(tmp_path / 'preload_house', {'from': './floors/ground'}, {'id': '1', 'zones': [{'name': 'loft', 'polygon': [[0.0, 0.0], [6.0, 0.0], [6.0, 4.0], [0.0, 4.0]]}], 'objects': [{'name': 'desk', 'model': 'Preload_Own_Desk', 'pose': [2.0, 2.0]}]})
    own = _model(house, 'Preload_Own_Desk')
    linked = _model(floor, 'Preload_Linked_Shelf')
    result = subprocess.run(
        [sys.executable, str(SCRIPT), 'preload_house', '--no-scenarios'],
        env={**os.environ, 'ARENA_WORLD_PATH': str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f'-> {own}' in result.stdout
    assert f'-> {linked}' in result.stdout
