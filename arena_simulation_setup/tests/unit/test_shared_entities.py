from __future__ import annotations

import pytest

from arena_simulation_setup.shared.entities import (
    CustomDynamicObstacle,
    DynamicObstacle,
    Entity,
    Named,
    Obstacle,
)
from arena_simulation_setup.tree.assets.Object import ObjectIdentifier
from arena_simulation_setup.tree.World.Scenario import parse_scenario, upgrade_scenario
from arena_simulation_setup.utils.geometry import Pose, Scale


def _model_id():
    return ObjectIdentifier("test_model")


def _make_entity(**kwargs) -> Entity:
    defaults = dict(
        name="test_entity",
        pose=Pose(),
        model=_model_id(),
    )
    defaults.update(kwargs)
    return Entity(**defaults)


def _make_obstacle(**kwargs) -> Obstacle:
    defaults = dict(
        name="test_obstacle",
        pose=Pose(),
        model=_model_id(),
        scale=None,
    )
    defaults.update(kwargs)
    return Obstacle(**defaults)


# ---------------------------------------------------------------------------
# Named
# ---------------------------------------------------------------------------


def test_named_sim_path_default_is_name():
    n = Named(name="foo")
    assert n.sim_path == "foo"


def test_named_sim_path_override():
    n = Named(name="foo", extra={"sim_path": "bar"})
    assert n.sim_path == "bar"


def test_named_sim_path_setter():
    n = Named(name="foo")
    n.sim_path = "custom"
    assert n.sim_path == "custom"
    assert n.extra["sim_path"] == "custom"


def test_named_extra_dict_preserved():
    n = Named(name="test", extra={"key": "value"})
    assert n.extra["key"] == "value"


def test_named_serialize_omits_empty_extra():
    n = Named(name="hello")
    d = n.serialize()
    assert "extra" not in d or not d["extra"]


def test_named_serialize_leaves_out_runtime_extra():
    n = Named(name="hello", extra={"custom_key": 42})
    d = n.serialize()
    assert d == {"name": "hello"}
    assert n.extra == {"custom_key": 42}


def test_named_parse_rejects_the_pos_alias_and_extra():
    with pytest.raises(ValueError, match=r"unknown keys \['pos'\]"):
        Named.parse({"name": "x", "pos": [1.0, 2.0]})
    with pytest.raises(ValueError, match=r"unknown keys \['extra'\]"):
        Named.parse({"name": "x", "extra": {"sim_path": "y"}})


def test_legacy_scenario_maps_pos_to_pose():
    upgraded, _ = upgrade_scenario({"static": [{"name": "x", "model": "box", "pos": [1.0, 2.0]}]})
    (obj,) = parse_scenario(upgraded).objects
    assert (obj.pose.position.x, obj.pose.position.y) == pytest.approx((1.0, 2.0))


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------


def test_entity_pose_converter():
    e = _make_entity(pose=Pose.parse([1.0, 2.0, 0.5]))
    assert e.pose.position.x == pytest.approx(1.0)


def test_entity_model_is_object_identifier():
    e = _make_entity()
    assert isinstance(e.model, ObjectIdentifier)


def test_entity_extra_is_read_directly():
    e = _make_entity()
    e.extra["my_key"] = "my_val"
    assert e.extra.get("my_key") == "my_val"
    assert e.extra.get("bbox") is None


def test_entity_serialize_leaves_out_extra():
    e = _make_entity()
    e.extra["my_key"] = "my_val"
    d = e.serialize()
    assert "my_key" not in d
    assert "extra" not in d


# ---------------------------------------------------------------------------
# Obstacle
# ---------------------------------------------------------------------------


def test_obstacle_scale_set():
    obs = _make_obstacle()
    obs.scale = Scale(2.0, 2.0, 2.0)
    assert obs.scale.x == pytest.approx(2.0)


def test_obstacle_serialize_returns_dict():
    obs = _make_obstacle()
    d = obs.serialize()
    assert isinstance(d, dict)
    assert d["name"] == obs.name


# ---------------------------------------------------------------------------
# DynamicObstacle
# ---------------------------------------------------------------------------


def test_dynamic_obstacle_empty_waypoints():
    from arena_simulation_setup.tree.assets.Human import HumanIdentifier

    do = DynamicObstacle(
        name="dyn",
        pose=Pose(),
        model=HumanIdentifier.converter("some_pedestrian"),
        waypoints=[],
        velocity=1.5,
    )
    assert do.waypoints == []
    assert do.velocity == pytest.approx(1.5)


def test_dynamic_obstacle_velocity_converter_string():
    from arena_simulation_setup.tree.assets.Human import HumanIdentifier

    do = DynamicObstacle(
        name="dyn",
        pose=Pose(),
        model=HumanIdentifier.converter("p"),
        velocity="2.5",
    )
    assert do.velocity == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# CustomDynamicObstacle
# ---------------------------------------------------------------------------


def test_custom_dynamic_obstacle_getattr_hit():
    from arena_simulation_setup.tree.assets.Human import HumanIdentifier

    cdo = CustomDynamicObstacle(
        name="cdo",
        pose=Pose(),
        model=HumanIdentifier.converter("p"),
    )
    cdo.extra["custom_prop"] = 99
    assert cdo.custom_prop == 99


def test_custom_dynamic_obstacle_getattr_miss():
    from arena_simulation_setup.tree.assets.Human import HumanIdentifier

    cdo = CustomDynamicObstacle(
        name="cdo",
        pose=Pose(),
        model=HumanIdentifier.converter("p"),
    )
    with pytest.raises(AttributeError):
        _ = cdo.nonexistent_field


def test_custom_dynamic_obstacle_parse_rejects_unknown_fields():
    data = {
        "name": "cdo",
        "pose": [0.0, 0.0],
        "model": "p",
        "custom_field": "hello",
    }
    with pytest.raises(ValueError, match=r"unknown keys \['custom_field'\]"):
        CustomDynamicObstacle.parse(data)
    cdo = CustomDynamicObstacle(name="cdo", pose=Pose(), model="p", extra={"custom_field": "hello"})
    assert cdo.custom_field == "hello"


@pytest.fixture
def bench_assets(tmp_path):
    from arena_simulation_setup.tree import DynamicPaths

    bench = tmp_path / "Hospital" / "Object" / "TestBench"
    bench.mkdir(parents=True)
    (bench / "annotation.yaml").write_text("bounding_box: [[-0.5, 0.5], [0.0, 0.4], [0.0, 0.8]]\naffordances:\n  sit:\n    - {bbox: [[-0.5, 0.5], [0.0, 0.4], [0.0, 0.45]], yaw: -y}\n")
    previous = DynamicPaths.ARENA.path
    DynamicPaths.ARENA.path = tmp_path
    yield tmp_path
    DynamicPaths.ARENA.path = previous


def test_obstacle_affordance_slots_read_the_asset_annotation(bench_assets):
    import asyncio

    slots = asyncio.run(Obstacle.parse({"name": "bench", "model": "Hospital/TestBench", "pose": [1.0, 2.0, 0.0]}).affordance_slots())
    assert [(slot.kind, slot.index, slot.bbox[2]) for slot in slots] == [("sit", 0, (0.0, 0.45))]


def test_obstacle_affordance_slots_are_empty_for_an_unresolvable_model(bench_assets):
    import asyncio

    assert asyncio.run(Obstacle.parse({"name": "ghost", "model": "Hospital/TestMissingSeat", "pose": [0.0, 0.0, 0.0]}).affordance_slots()) == []
