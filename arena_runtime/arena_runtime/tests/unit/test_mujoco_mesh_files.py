"""MuJoCo obstacles take every mesh visual of their SDF, glow visuals included."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("arena_runtime.sim.mujoco_simulator")

from arena_runtime.sim.mujoco_simulator import _mesh_files  # noqa: E402
from arena_simulation_setup.utils.models import Model, ModelType  # noqa: E402

_LAMP_SDF = """<sdf version="1.9"><model name="lamp"><link name="link">
<visual name="visual"><geometry><mesh><uri>lamp.dae</uri></mesh></geometry></visual>
<visual name="glow_Shade"><geometry><mesh><uri>file://lamp.glow_0.dae</uri></mesh></geometry><material><diffuse>0.85 0.8 0.7 1</diffuse></material></visual>
<visual name="marker"><geometry><box><size>1 1 1</size></box></geometry></visual>
<collision name="collision"><geometry><mesh><uri>lamp.dae</uri></mesh></geometry></collision>
</link></model></sdf>"""


def test_sdf_lamp_gives_its_body_and_glow_meshes_and_skips_primitives(tmp_path: Path):
    model = Model(type=ModelType.SDF, name="lamp", description=_LAMP_SDF, path=tmp_path / "lamp.sdf")
    assert _mesh_files(model) == [("visual", str(tmp_path / "lamp.dae")), ("glow_Shade", str(tmp_path / "lamp.glow_0.dae"))]


def test_obj_is_its_own_single_visual_and_a_pathless_model_has_none(tmp_path: Path):
    assert _mesh_files(Model(type=ModelType.OBJ, name="crate", description="", path=tmp_path / "crate.obj")) == [("visual", str(tmp_path / "crate.obj"))]
    assert _mesh_files(Model(type=ModelType.SDF, name="lamp", description=_LAMP_SDF, path=None)) == []
