"""World reference closure, offline: no wall kinds, so nothing reaches a network resolver."""

from __future__ import annotations

import pytest

from arena_simulation_setup.tree.World import WorldView

_WORLD = """
format: arena-world/2
levels:
- id: '0'
  zones:
  - name: z
    polygon: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    floor: false
  walls:
  - path: [[0.0, 0.0], [1.0, 0.0]]
"""

_LEGACY_WORLD = """
zones:
  - description: "test zone"
    name: "z"
    walls:
      - start: {x: 0.0, y: 0.0}
        end: {x: 1.0, y: 0.0}
    corners:
      - [0.0, 0.0]
      - [1.0, 0.0]
      - [1.0, 1.0]
      - [0.0, 1.0]
    mat: ""
    entities:
      static: []
"""


@pytest.fixture(params=[_WORLD, _LEGACY_WORLD], ids=['format2', 'format1'])
def _world(tmp_path, request):
    (tmp_path / 'world.yaml').write_text(request.param)
    return WorldView(tmp_path)


def test_closure_skips_unset_material(_world):
    """`floor: false` (format 1 `mat: ""`) is an unset field, not a reference to an asset called ''."""
    identifiers = list(_world.identifiers(scenarios=False))
    assert [i for i in identifiers if not i.name] == []


def test_closure_yields_the_materials_that_are_set(_world):
    """The ceiling and wall materials fall back to named defaults, and those are references."""
    identifiers = list(_world.identifiers(scenarios=False))
    assert {i.shortname for i in identifiers} == {'Common/Material/Concrete_Smooth', 'Common/Material/Marble'}


def test_closure_yields_the_default_ceiling_material(_world):
    identifiers = list(_world.identifiers(scenarios=False))
    assert 'Common/Material/Concrete_Smooth' in {i.shortname for i in identifiers}
