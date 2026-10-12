from __future__ import annotations

import pytest
import yaml

from arena_simulation_setup.kg import FORMAT, Vocabulary, dump, lift_graph, schema_for

jsonschema = pytest.importorskip("jsonschema")

KG_YAML = """\
format: arena-kg/1
uses: [core@2, social@1, access@1]
entities:
- {id: floor:0}
- {id: room:reception, props: {type: Main Entrance, description: A spacious reception and waiting hall}}
- {id: room:pharmacy}
- {id: door:main_entrance}
- {id: obj:kiosk_0, props: {label: Information Kiosk, placed: false}}
- {id: elevator:lift_0}
- {id: elevator:lift_1}
- {id: wall:north}
- {id: role:staff}
- {id: region:queue, props: {kind: source, boundary: '0,0 1,0 1,1'}}
facts:
- {subject: floor:0, verb: contains, object: room:reception}
- {subject: floor:0, verb: contains, object: elevator:lift_0}
- {subject: elevator:lift_0, verb: connects_floor, object: elevator:lift_1}
- {subject: room:reception, verb: connects, object: room:pharmacy, params: {door: door:main_entrance}, source: derived}
- {subject: room:pharmacy, verb: restricted_to, object: role:staff}
- {subject: room:reception, verb: attr, object: 'shape: L-shaped'}
- {subject: region:queue, verb: note, object: 3}
constraints:
- {id: con:exit_clear, kind: keep_clear, scope: [door:main_entrance], params: {min: 1.0}}
- {id: con:pos, kind: placement, scope: [room:reception, room:pharmacy], params: {dir: left_of}, hard: false}
"""

USES = ["core@2", "social@1", "access@1"]


@pytest.fixture
def validator() -> jsonschema.Draft202012Validator:
    schema = schema_for(Vocabulary(USES))
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def _errors(validator: jsonschema.Draft202012Validator, data: dict) -> list[str]:
    return [e.message for e in validator.iter_errors(data)]


def test_schema_header_names_format_and_packages():
    schema = schema_for(Vocabulary(USES))
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["format"] == {"const": FORMAT}
    assert schema["properties"]["uses"]["items"] == {"enum": USES}


def test_design_example_satisfies_schema(validator: jsonschema.Draft202012Validator):
    assert _errors(validator, yaml.safe_load(KG_YAML)) == []


def test_lifted_graph_satisfies_core2_schema():
    scene = lift_graph({"prompt": "a flat", "nodes": [{"id": "a", "room_type": "Hall", "assets": ["Sofa"], "attributes": ["shape: square"]}, {"id": "b", "room_type": "Bath"}], "edges": [{"id": "e", "source": "a", "target": "b", "attributes": ["position: left of"]}]})
    assert list(jsonschema.Draft202012Validator(schema_for(scene.vocab)).iter_errors(dump(scene))) == []


@pytest.mark.parametrize(
    ("section", "item"),
    [
        ("entities", {"id": "room:a", "props": {"shade": "red"}}),
        ("entities", {"id": "obj:a", "props": {"placed": "no"}}),
        ("entities", {"id": "role:staff", "props": {"label": "Staff"}}),
        ("entities", {"id": "ghost:a"}),
        ("entities", {"id": "room:a b"}),
        ("facts", {"subject": "obj:a", "verb": "connects", "object": "room:b"}),
        ("facts", {"subject": "room:a", "verb": "contains", "object": "role:staff"}),
        ("facts", {"subject": "room:a", "verb": "hugs", "object": "room:b"}),
        ("facts", {"subject": "room:a", "verb": "attr", "object": "x", "source": "measured"}),
        ("facts", {"subject": "room:a", "verb": "attr", "object": "x", "sorce": "derived"}),
        ("constraints", {"id": "con:k", "kind": "keep_clear", "scope": ["door:a", "door:b"]}),
        ("constraints", {"id": "con:k", "kind": "keep_clear", "scope": ["door:a"], "params": {"min": "far"}}),
        ("constraints", {"id": "con:k", "kind": "placement", "scope": ["room:a", "room:b"], "params": {"dir": "behind"}}),
        ("constraints", {"id": "k", "kind": "keep_clear", "scope": ["door:a"]}),
    ],
)
def test_schema_rejects_off_vocabulary_items(validator: jsonschema.Draft202012Validator, section: str, item: dict):
    assert _errors(validator, {"format": FORMAT, "uses": USES, section: [item]}) != []


@pytest.mark.parametrize(
    "data",
    [
        {"uses": USES},
        {"format": "arena-kg/2", "uses": USES},
        {"format": FORMAT, "uses": ["robot@1"]},
        {"format": FORMAT, "uses": USES, "sim": {}},
    ],
)
def test_schema_rejects_bad_top_level(validator: jsonschema.Draft202012Validator, data: dict):
    assert _errors(validator, data) != []


def test_schema_follows_loaded_packages():
    core_only = schema_for(Vocabulary(["core@2"]))
    verbs = [v["properties"]["verb"]["const"] for v in core_only["properties"]["facts"]["items"]["oneOf"]]
    assert "restricted_to" not in verbs
    assert "connects_floor" in verbs
    kinds = [v["properties"]["id"]["pattern"] for v in core_only["properties"]["entities"]["items"]["oneOf"]]
    assert "^(microphone):[^ ]+$" in kinds
    assert "^(role):[^ ]+$" not in kinds
