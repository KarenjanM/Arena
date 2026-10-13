from __future__ import annotations

import pytest
import yaml

from arena_simulation_setup.kg import DERIVED, FORMAT, Scene, dump, load, write

KG_YAML = """\
format: arena-kg/1
uses: [core@2, social@1, access@1]
entities:
- {id: floor:0}
- {id: floor:1}
- {id: room:reception, props: {type: Main Entrance, description: A spacious reception and waiting hall}}
- {id: room:triage, props: {type: Triage and Admissions Area}}
- {id: room:emergency}
- {id: room:pharmacy}
- {id: door:main_entrance}
- {id: obj:desk_1, props: {label: Reception Desk}}
- {id: obj:reception_kiosk_0, props: {label: Information Kiosk, placed: false}}
- {id: elevator:lift_0}
- {id: elevator:lift_1}
- {id: role:staff}
facts:
- {subject: floor:0, verb: contains, object: room:reception}
- {subject: floor:0, verb: contains, object: elevator:lift_0}
- {subject: floor:1, verb: contains, object: elevator:lift_1}
- {subject: elevator:lift_0, verb: connects_floor, object: elevator:lift_1}
- {subject: room:reception, verb: contains, object: obj:desk_1}
- {subject: room:triage, verb: contains, object: obj:desk_1}
- {subject: room:reception, verb: connects, object: room:triage, params: {door: door:main_entrance}, source: derived}
- {subject: room:emergency, verb: left_of, object: room:triage}
- {subject: room:pharmacy, verb: restricted_to, object: role:staff}
- {subject: room:reception, verb: attr, object: 'shape: L-shaped'}
constraints:
- {id: con:exit_clear, kind: keep_clear, scope: [door:main_entrance], params: {min: 1.0}}
- {id: con:quiet, kind: no_entry, scope: [role:staff, room:pharmacy], hard: false}
"""


def _kg_data() -> dict:
    return yaml.safe_load(KG_YAML)


def test_design_example_loads_and_validates():
    scene = load(_kg_data())
    assert scene.uses == ["core@2", "social@1", "access@1"]
    assert scene.validate() == []
    assert scene.get_entity("obj:reception_kiosk_0").props == {"label": "Information Kiosk", "placed": False}
    assert scene.get_fact("room:reception", "connects", "room:triage", {"door": "door:main_entrance"}).source == DERIVED
    assert scene.get_constraint("con:quiet").hard is False


def test_dump_reproduces_the_file_mapping():
    assert dump(load(_kg_data())) == _kg_data()


def test_dump_omits_defaults_and_empty_fields():
    scene = Scene(["core@2", "social@1"])
    scene.add_entity("room:a")
    scene.add_entity("room:b", {"type": "Kitchen"})
    scene.add_fact("room:a", "connects", "room:b")
    scene.add_constraint("con:gap", "min_clearance", ["room:a", "room:b"], {"min": 0.5})
    assert dump(scene) == {
        "format": FORMAT,
        "uses": ["core@2", "social@1"],
        "entities": [{"id": "room:a"}, {"id": "room:b", "props": {"type": "Kitchen"}}],
        "facts": [{"subject": "room:a", "verb": "connects", "object": "room:b"}],
        "constraints": [{"id": "con:gap", "kind": "min_clearance", "scope": ["room:a", "room:b"], "params": {"min": 0.5}}],
    }


def test_write_then_load_from_path(tmp_path):
    path = tmp_path / "kg.yaml"
    write(load(_kg_data()), path)
    assert yaml.safe_load(path.read_text()) == _kg_data()
    assert dump(load(path)) == _kg_data()


def test_misspelled_fact_key_names_the_item(tmp_path):
    data = _kg_data()
    data["facts"][3]["sorce"] = "derived"
    with pytest.raises(ValueError, match=r"^facts\[3\]: unknown key 'sorce'$"):
        load(data)
    path = tmp_path / "kg.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=rf"^{path}: facts\[3\]: unknown key 'sorce'$"):
        load(path)


@pytest.mark.parametrize(
    ("section", "index", "key"),
    [("entities", 2, "kind"), ("constraints", 0, "soft")],
)
def test_unknown_item_key_is_rejected(section: str, index: int, key: str):
    data = _kg_data()
    data[section][index][key] = True
    with pytest.raises(ValueError, match=rf"^{section}\[{index}\]: unknown key '{key}'$"):
        load(data)


def test_unknown_top_level_key_is_rejected():
    data = _kg_data()
    data["sim"] = {}
    with pytest.raises(ValueError, match=r"top level: unknown key 'sim'"):
        load(data)


def test_missing_format_is_rejected():
    data = _kg_data()
    del data["format"]
    with pytest.raises(ValueError, match=r"top level: missing key 'format'"):
        load(data)


def test_wrong_format_is_rejected():
    data = _kg_data()
    data["format"] = "arena-kg/2"
    with pytest.raises(ValueError, match=r"format: expected 'arena-kg/1'"):
        load(data)


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "kg.yaml"
    path.write_text("")
    with pytest.raises(ValueError, match=r"top level: expected a mapping"):
        load(path)


def test_invalid_fact_source_is_rejected():
    data = _kg_data()
    data["facts"][0]["source"] = "measured"
    with pytest.raises(ValueError, match=r"^facts\[0\]: fact source 'measured'"):
        load(data)


def test_duplicate_entity_is_rejected():
    data = _kg_data()
    data["entities"].append({"id": "floor:0"})
    with pytest.raises(ValueError, match=r"entities\[12\]: duplicate entity 'floor:0'"):
        load(data)


def test_entity_id_without_kind_is_rejected():
    data = _kg_data()
    data["entities"][0]["id"] = "reception"
    with pytest.raises(ValueError, match=r"^entities\[0\]: id 'reception' is not of the form <kind>:<name>$"):
        load(data)


def test_nested_prop_value_is_rejected():
    data = _kg_data()
    data["entities"][2]["props"]["type"] = ["a", "b"]
    with pytest.raises(ValueError, match=r"entities\[2\].props.type: expected a string, number or boolean"):
        load(data)


def test_auto_kinds_are_declared_on_first_reference():
    scene = load({"format": FORMAT, "uses": ["core@2", "social@1"], "entities": [{"id": "room:a"}], "constraints": [{"id": "con:c", "kind": "accessible_route", "scope": ["profile:wheelchair", "room:a", "room:a"]}]})
    assert "profile:wheelchair" in scene.entities
    assert scene.validate() == []
    assert dump(scene)["entities"] == [{"id": "room:a"}, {"id": "profile:wheelchair"}]


def test_slash_in_name_is_accepted():
    scene = load({"format": FORMAT, "uses": ["core@2"], "entities": [{"id": "obj:icu_0/bed_1"}]})
    assert scene.get_entity("obj:icu_0/bed_1").name == "icu_0/bed_1"


def test_validate_reports_vocabulary_violations():
    scene = Scene(["core@2", "social@1"])
    scene.add_entity("room:a", {"shade": "red"})
    scene.add_entity("obj:chair")
    scene.add_entity("ghost:x")
    scene.add_fact("obj:chair", "connects", "room:a")
    scene.add_fact("room:a", "contains", "obj:table")
    scene.add_fact("room:a", "hugs", "room:a")
    scene.add_fact("room:a", "connects", "room:a", {"door": "door:missing"})
    scene.add_constraint("con:k", "keep_clear", ["room:a", "obj:chair"], {"max": 1.0})
    scene.add_constraint("con:u", "levitate", ["room:a"])
    assert scene.validate() == [
        "entity room:a: undeclared prop 'shade'",
        "entity ghost:x: unknown kind 'ghost'",
        "fact ('obj:chair', 'connects', 'room:a'): subject kind 'obj' not allowed for 'connects'",
        "fact ('room:a', 'contains', 'obj:table'): unknown object 'obj:table'",
        "fact ('room:a', 'hugs', 'room:a'): unknown verb 'hugs'",
        "fact ('room:a', 'connects', 'room:a'): param door='door:missing' references unknown entity",
        "constraint con:k: expects 1 ids, got 2",
        "constraint con:k: undeclared param 'max'",
        "constraint con:u: unknown kind 'levitate'",
    ]


def test_drop_derived_keeps_stated_facts():
    scene = load(_kg_data())
    stated = scene.stated()
    dropped = scene.drop_derived()
    assert [f.key for f in dropped] == [("room:reception", "connects", "room:triage", "door:main_entrance")]
    assert scene.facts == stated
    assert all(f.source == "generated" for f in scene.facts)


def test_add_fact_merges_params_of_an_existing_key():
    scene = Scene()
    scene.add_entity("room:a")
    scene.add_entity("obj:o")
    scene.add_fact("room:a", "contains", "obj:o")
    scene.add_fact("room:a", "contains", "obj:o", {"note": "corner"})
    assert [f.params for f in scene.facts] == [{"note": "corner"}]


def test_two_doors_between_two_rooms_are_two_connects_facts():
    scene = Scene()
    for ref in ("room:a", "room:b", "door:d1", "door:d2"):
        scene.add_entity(ref)
    scene.add_fact("room:a", "connects", "room:b", {"door": "door:d1"})
    scene.add_fact("room:a", "connects", "room:b", {"door": "door:d2"})
    assert sorted(f.params["door"] for f in scene.facts) == ["door:d1", "door:d2"]
    assert scene.get_fact("room:a", "connects", "room:b", {"door": "door:d2"}) is not None


def test_remove_entity_drops_its_facts_and_constraints():
    scene = load(_kg_data())
    entity, facts, constraints = scene.remove_entity("room:pharmacy")
    assert entity is not None
    assert entity.id == "room:pharmacy"
    assert [f.key for f in facts] == [("room:pharmacy", "restricted_to", "role:staff")]
    assert [c.id for c in constraints] == ["con:quiet"]
    assert scene.validate() == []
    assert scene.remove_fact("floor:0", "contains", "room:reception") is not None
    assert scene.remove_constraint("con:exit_clear") is not None
    assert scene.constraints == {}
