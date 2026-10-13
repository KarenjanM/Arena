from __future__ import annotations

from arena_simulation_setup.kg import Scene, check_against_world

WORLD_NAMES = {
    "floor": ["0", "1"],
    "room": ["reception", "triage"],
    "door": ["main_entrance"],
    "elevator": ["lift_0", "lift_1"],
    "obj": ["desk_1"],
    "light": ["lamp"],
    "sound": ["alarm"],
    "microphone": ["array"],
}


def _scene(*ids: str) -> Scene:
    scene = Scene(["core@2", "social@1"])
    for id in ids:
        scene.add_entity(id)
    return scene


def test_resolving_entities_pass():
    scene = _scene("floor:0", "floor:1", "room:reception", "room:triage", "door:main_entrance", "elevator:lift_0", "obj:desk_1", "light:lamp", "sound:alarm", "microphone:array")
    scene.add_entity("room:reception", {"type": "Main Entrance", "description": "A hall"})
    assert check_against_world(scene, WORLD_NAMES) == []


def test_unresolved_world_kinds_are_reported():
    scene = _scene("room:lobby", "door:back", "floor:2", "elevator:freight", "light:spot", "sound:hum", "microphone:mic")
    assert check_against_world(scene, WORLD_NAMES) == [
        "room:lobby: no room named 'lobby' in world.yaml",
        "door:back: no door named 'back' in world.yaml",
        "floor:2: no floor named '2' in world.yaml",
        "elevator:freight: no elevator named 'freight' in world.yaml",
        "light:spot: no light named 'spot' in world.yaml",
        "sound:hum: no sound named 'hum' in world.yaml",
        "microphone:mic: no microphone named 'mic' in world.yaml",
    ]


def test_unplaced_object_need_not_resolve():
    scene = _scene()
    scene.add_entity("obj:kiosk_0", {"label": "Kiosk", "placed": False})
    scene.add_entity("obj:chair", {"label": "Chair"})
    scene.add_entity("obj:bench", {"placed": True})
    assert check_against_world(scene, WORLD_NAMES) == [
        "obj:chair: no obj named 'chair' in world.yaml",
        "obj:bench: no obj named 'bench' in world.yaml",
    ]


def test_geometry_props_on_resolving_entities_are_reported():
    scene = _scene()
    scene.add_entity("room:reception", {"boundary": "0,0 1,0 1,1", "area": 1.0})
    scene.add_entity("door:main_entrance", {"line": "0,0 1,0", "sim": "x"})
    scene.add_entity("obj:desk_1", {"pose": "0,0,0", "model": "desk", "scale": 1.0})
    assert check_against_world(scene, WORLD_NAMES) == [
        "room:reception: geometry prop 'boundary' on an entity that resolves in world.yaml",
        "room:reception: geometry prop 'area' on an entity that resolves in world.yaml",
        "door:main_entrance: geometry prop 'line' on an entity that resolves in world.yaml",
        "door:main_entrance: geometry prop 'sim' on an entity that resolves in world.yaml",
        "obj:desk_1: geometry prop 'pose' on an entity that resolves in world.yaml",
        "obj:desk_1: geometry prop 'model' on an entity that resolves in world.yaml",
        "obj:desk_1: geometry prop 'scale' on an entity that resolves in world.yaml",
    ]


def test_unplaced_object_keeps_its_geometry():
    scene = _scene()
    scene.add_entity("obj:sketch", {"placed": False, "pose": "1,2,0", "model": "chair"})
    assert check_against_world(scene, WORLD_NAMES) == []


def test_kg_only_kinds_are_never_checked():
    scene = _scene("agent:nurse", "role:staff", "profile:wheelchair")
    scene.add_entity("region:queue", {"kind": "source", "boundary": "0,0 1,0 1,1"})
    scene.add_entity("material:Common/tile", {"label": "tile"})
    assert check_against_world(scene, WORLD_NAMES) == []


def test_kind_absent_from_names_resolves_nothing():
    assert check_against_world(_scene("sound:alarm"), {"room": ["reception"]}) == ["sound:alarm: no sound named 'alarm' in world.yaml"]
