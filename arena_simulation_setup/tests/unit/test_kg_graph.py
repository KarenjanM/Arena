from __future__ import annotations

import pytest
import yaml

from arena_simulation_setup.kg import dump, lift_graph, load, parse_direction

EXPORTED_GRAPH = """\
prompt: a two-bedroom apartment with living room, kitchen, bathroom
nodes:
- attributes:
  - 'shape: rectangular'
  id: living_room
  room_type: Living Room
  assets:
  - Sofa
  - Coffee Table
  - TV
  - Rug
  description: A bright and inviting living room featuring a comfortable sofa facing
    a modern TV setup. The space is anchored by a warm area rug and a central coffee
    table.
- attributes:
  - 'shape: rectangular'
  id: kitchen
  room_type: Kitchen
  assets:
  - Countertop
  - Oven
  - Refrigerator
  - Dining Table
  description: A functional modern kitchen equipped with stainless steel appliances
    and ample countertop space. A small dining table sits neatly in the corner for
    casual meals.
- attributes:
  - 'shape: square'
  id: bathroom
  room_type: Bathroom
  assets:
  - Bathtub
  - Toilet
  - Sink
  - Mirror
  description: A compact yet stylish bathroom with a pristine bathtub and a sleek
    vanity sink. Crisp white tiles complete the clean, contemporary aesthetic.
- attributes:
  - 'shape: rectangular'
  id: bedroom_1
  room_type: Bedroom
  assets:
  - Double Bed
  - Wardrobe
  - Nightstand
  description: A spacious primary bedroom centered around a plush double bed with
    matching wooden nightstands. A large wardrobe provides generous storage along
    the wall.
- attributes:
  - 'shape: rectangular'
  id: bedroom_2
  room_type: Bedroom
  assets:
  - Single Bed
  - Desk
  - Chair
  - Wardrobe
  description: A cozy secondary bedroom ideal for a guest or child, featuring a single
    bed and a dedicated desk area. Soft lighting creates a peaceful atmosphere.
edges:
- attributes:
  - 'position: right of'
  id: edge_kitchen_living
  source: kitchen
  target: living_room
- attributes:
  - 'position: left of'
  id: edge_bed1_living
  source: bedroom_1
  target: living_room
- attributes:
  - 'position: above'
  id: edge_bath_living
  source: bathroom
  target: living_room
- attributes:
  - 'position: below'
  id: edge_bed2_living
  source: bedroom_2
  target: living_room
"""


@pytest.fixture
def exported(tmp_path) -> dict:
    path = tmp_path / "graph.yaml"
    path.write_text(EXPORTED_GRAPH)
    with path.open() as f:
        return yaml.safe_load(f)


def test_exported_graph_lifts_and_validates(exported: dict):
    scene = lift_graph(exported)
    assert scene.uses == ["core@2"]
    assert scene.validate() == []
    rooms = [e.id for e in scene.entities.values() if e.kind == "room"]
    assert rooms == ["room:living_room", "room:kitchen", "room:bathroom", "room:bedroom_1", "room:bedroom_2"]
    assert len([e for e in scene.entities.values() if e.kind == "obj"]) == 19
    assert [e.id for e in scene.entities.values() if e.kind == "door"] == ["door:door_edge_kitchen_living", "door:door_edge_bed1_living", "door:door_edge_bath_living", "door:door_edge_bed2_living"]


def test_node_becomes_room_with_type_description_and_attr(exported: dict):
    scene = lift_graph(exported)
    kitchen = scene.get_entity("room:kitchen")
    assert kitchen.props["type"] == "Kitchen"
    assert kitchen.props["description"].startswith("A functional modern kitchen")
    assert scene.get_fact("room:kitchen", "attr", "shape: rectangular") is not None


def test_assets_become_unplaced_objects_contained_by_their_room(exported: dict):
    scene = lift_graph(exported)
    table = scene.get_entity("obj:living_room_Coffee_Table_1")
    assert table.props == {"label": "Coffee Table", "placed": False}
    assert scene.get_fact("room:living_room", "contains", "obj:living_room_Coffee_Table_1") is not None
    assert "obj:bedroom_2_Wardrobe_3" in scene.entities
    assert "obj:bedroom_1_Wardrobe_1" in scene.entities


def test_edges_become_door_connections_and_positions(exported: dict):
    scene = lift_graph(exported)
    assert scene.get_fact("room:kitchen", "connects", "room:living_room", {"door": "door:door_edge_kitchen_living"}).params == {"door": "door:door_edge_kitchen_living"}
    assert scene.get_fact("room:kitchen", "right_of", "room:living_room") is not None
    assert scene.get_fact("room:bedroom_1", "left_of", "room:living_room") is not None
    assert scene.get_fact("room:bathroom", "above", "room:living_room") is not None
    assert scene.get_fact("room:bedroom_2", "below", "room:living_room") is not None
    assert all(f.source == "generated" for f in scene.facts)


def test_prompt_becomes_note_on_floor_zero(exported: dict):
    scene = lift_graph(exported)
    assert "floor:0" in scene.entities
    assert scene.get_fact("floor:0", "note", "a two-bedroom apartment with living room, kitchen, bathroom") is not None


def test_lifted_scene_round_trips_through_the_kg(exported: dict):
    scene = lift_graph(exported)
    assert dump(load(dump(scene))) == dump(scene)


def test_graph_without_prompt_or_edges_has_no_floor():
    scene = lift_graph({"nodes": [{"id": "office", "room_type": "Office", "assets": ["Desk"]}], "edges": []})
    assert list(scene.entities) == ["room:office", "obj:office_Desk_0"]
    assert scene.validate() == []


def test_dangling_edge_shows_in_validation():
    scene = lift_graph({"nodes": [{"id": "a", "room_type": "Hall"}], "edges": [{"id": "e", "source": "a", "target": "b"}]})
    assert scene.validate() == ["fact ('room:a', 'connects', 'room:b'): unknown object 'room:b'"]


def test_non_positional_edge_attribute_is_rejected():
    with pytest.raises(ValueError, match=r"edges\[0\].attributes\[0\]: 'wide' is no position"):
        lift_graph({"nodes": [{"id": "a", "room_type": "Hall"}], "edges": [{"id": "e", "source": "a", "target": "a", "attributes": ["wide"]}]})


def test_unknown_node_key_is_rejected():
    with pytest.raises(ValueError, match=r"nodes\[0\]: unknown key 'roomtype'"):
        lift_graph({"nodes": [{"id": "a", "roomtype": "Hall"}]})


def test_colliding_object_ids_are_rejected():
    nodes = [{"id": "a", "room_type": "Hall", "assets": ["b c"]}, {"id": "a_b", "room_type": "Hall", "assets": ["c"]}]
    with pytest.raises(ValueError, match=r"nodes\[1\].assets\[0\]: object id 'obj:a_b_c_0' already taken"):
        lift_graph({"nodes": nodes})


@pytest.mark.parametrize(
    ("attribute", "direction"),
    [("position: left of", "left_of"), ("position:right of", "right_of"), ("position: above", "above"), ("Position: Below", "below"), ("above of", "above"), ("shape: square", None)],
)
def test_parse_direction(attribute: str, direction: str | None):
    assert parse_direction(attribute) == direction


def test_edges_connect_through_the_doors_the_world_has_and_keep_their_positions(exported: dict):
    scene = lift_graph(exported, doors={"door_kitchen__living_room", "door_edge_bed1_living"})
    assert scene.validate() == []
    connects = [(f.subject, f.object, f.params["door"]) for f in scene.facts if f.verb == "connects"]
    assert connects == [("room:kitchen", "room:living_room", "door:door_kitchen__living_room"), ("room:bedroom_1", "room:living_room", "door:door_edge_bed1_living")]
    assert [e.id for e in scene.entities.values() if e.kind == "door"] == ["door:door_kitchen__living_room", "door:door_edge_bed1_living"]
    positions = {"left_of", "right_of", "above", "below"}
    assert len([f for f in scene.facts if f.verb in positions]) == len([f for f in lift_graph(exported).facts if f.verb in positions])
