from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from arena_simulation_setup.tree.configs.parametrized import ParametrizedConfig, ParametrizedIdentifier
from arena_simulation_setup.tree.configs.environment import EnvironmentDescription, EnvironmentIdentifier


# ---------------------------------------------------------------------------
# ParametrizedIdentifier.load
# ---------------------------------------------------------------------------


def _write_xml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "test_config.xml"
    p.write_text(content)
    return p


def test_parametrized_load_valid_xml(tmp_path):
    xml = """<random>
  <static>
    <obstacle min="1" max="3" model="box" type="static"/>
  </static>
  <interactive/>
  <dynamic/>
</random>"""
    path = _write_xml(tmp_path, xml)
    ident = ParametrizedIdentifier("test_config")
    cfg = ident.load(path)
    assert isinstance(cfg, ParametrizedConfig)
    assert len(cfg.STATIC) == 1
    assert cfg.STATIC[0].min == 1
    assert cfg.STATIC[0].max == 3


def test_parametrized_load_interactive_and_dynamic_zones(tmp_path):
    xml = """<random>
  <static/>
  <interactive>
    <obstacle min="2" max="4" model="chair" type="interactive"/>
  </interactive>
  <dynamic>
    <obstacle min="1" max="5" model="adult" type="dynamic" zone="lobby"/>
  </dynamic>
</random>"""
    path = _write_xml(tmp_path, xml)
    ident = ParametrizedIdentifier("test_config")
    cfg = ident.load(path)
    assert cfg.STATIC == []
    assert len(cfg.INTERACTIVE) == 1
    assert cfg.INTERACTIVE[0].min == 2
    assert cfg.INTERACTIVE[0].max == 4
    assert cfg.INTERACTIVE[0].model == "chair"
    assert cfg.INTERACTIVE[0].zone == ""
    assert len(cfg.DYNAMIC) == 1
    assert cfg.DYNAMIC[0].model == "adult"
    assert cfg.DYNAMIC[0].zone == "lobby"


def test_parametrized_load_wrong_root_tag(tmp_path):
    xml = "<world><static/></world>"
    path = _write_xml(tmp_path, xml)
    ident = ParametrizedIdentifier("test_config")
    with pytest.raises(ValueError, match="random"):
        ident.load(path)


def test_parametrized_load_missing_model_attr(tmp_path):
    xml = """<random>
  <static>
    <obstacle min="1" max="3"/>
  </static>
</random>"""
    path = _write_xml(tmp_path, xml)
    ident = ParametrizedIdentifier("test_config")
    with pytest.raises((ValueError, KeyError)):
        ident.load(path)


def test_parametrized_load_empty_sections(tmp_path):
    xml = "<random><static/><interactive/><dynamic/></random>"
    path = _write_xml(tmp_path, xml)
    ident = ParametrizedIdentifier("test_config")
    cfg = ident.load(path)
    assert cfg.STATIC == []
    assert cfg.INTERACTIVE == []
    assert cfg.DYNAMIC == []


# ---------------------------------------------------------------------------
# EnvironmentIdentifier.load
# ---------------------------------------------------------------------------


def test_environment_load_valid_yaml(tmp_path):
    env_file = tmp_path / "env.yaml"
    env_file.write_text(yaml.dump({"key1": "value1", "key2": 42}))
    ident = EnvironmentIdentifier("env.yaml")
    desc = ident.load(env_file)
    assert isinstance(desc, EnvironmentDescription)
    assert desc["key1"] == "value1"


def test_environment_load_non_dict_raises(tmp_path):
    env_file = tmp_path / "env.yaml"
    env_file.write_text(yaml.dump(["a", "b", "c"]))
    ident = EnvironmentIdentifier("env.yaml")
    with pytest.raises(ValueError, match="mapping"):
        ident.load(env_file)


def test_environment_load_file_not_found(tmp_path):
    ident = EnvironmentIdentifier("nonexistent.yaml")
    with pytest.raises((FileNotFoundError, OSError)):
        ident.load(tmp_path / "nonexistent.yaml")
