from __future__ import annotations

import math
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from arena_simulation_setup.kg.scene import Scene
from arena_simulation_setup.shared.route import ORDINALS, WORDINGS, instruct, level_nouns, portals, zone_shapes
from arena_simulation_setup.shared.task import GoToPhase
from arena_simulation_setup.tree.World import Level, World, WorldView, contents
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position

WORLDS = Path(__file__).parents[2] / "worlds"


def _world(name: str) -> World:
    return WorldView(WORLDS / name).load()


ZONED = sorted(path.name for path in WORLDS.iterdir() if path.is_dir() and len(_world(path.name).all_levels) == 1 and len(_world(path.name).all_levels[0].zones) > 1)

TEMPLATE_WORDS = frozenset("go to walk through then take the door on your left right into ahead turn around and continue straight about meter meters stop next just inside here".split()) | frozenset(ORDINALS)

ROOM = r"the [a-z0-9() -]+?"
SIDE = r"(left|right)"
CROSS = rf"(take the (({'|'.join(ORDINALS)}) )?door on your {SIDE} into {ROOM}|go through the door ahead into {ROOM}|turn {SIDE} into {ROOM}|continue straight into {ROOM}|turn around and (go through the door|continue) into {ROOM})"
TURN = rf"(turn ({SIDE}|around) and )?"
SENTENCES = [
    re.compile(rf"go to {ROOM}\."),
    re.compile(rf"{TURN}walk through {ROOM}, then {CROSS}\."),
    re.compile(rf"{CROSS}\."),
    re.compile(rf"{TURN}walk about [0-9]+ meters?( into {ROOM})? and stop( next to {ROOM})?\."),
    re.compile(rf"stop( next to {ROOM})? (just inside {ROOM}|here)\."),
]


def _level(world: str) -> tuple[Level, Scene]:
    loaded = _world(world)
    return loaded.all_levels[0], loaded.kg


def _goto(target: str, x: float, y: float) -> GoToPhase:
    return GoToPhase(target=target, pose=Pose(position=Position(x, y), orientation=Orientation.identity()))


def instructions(level: Level, semantics: Scene) -> Iterator[str]:
    """Every instruction between two zone interiors from four headings in both wordings, and the directions to every static entity."""
    spots = {name: shape.representative_point().coords[0] for name, shape in zone_shapes(level).items()}
    found = portals(level, semantics)
    for start in spots.values():
        for target, goal in spots.items():
            for quarter in range(4):
                for wording in WORDINGS:
                    yield instruct(_goto(target, *goal), level, semantics, (*start, quarter * math.pi / 2.0), wording, found=found)["text"]
    first = next(iter(spots.values()))
    objects = {f"obj:{entity.name}": entity for entity in level.objects}
    for zone in level.zones:
        for entity in (objects[ref] for ref in contents(semantics, zone.name) if ref in objects):
            yield instruct(_goto(zone.name, entity.pose.position.x, entity.pose.position.y), level, semantics, (*first, 0.0), "route", found=found)["text"]


def test_some_shipped_world_has_zones():
    assert {"hospital_1", "office_1"} <= set(ZONED)


@pytest.mark.parametrize("world", ZONED)
def test_every_sentence_follows_a_template(world):
    for text in instructions(*_level(world)):
        for sentence in re.split(r"(?<=\.) (?=[A-Z])", text):
            assert sentence[0].isupper(), text
            assert any(pattern.fullmatch(sentence.lower()) for pattern in SENTENCES), sentence


@pytest.mark.parametrize("world", ZONED)
def test_words_are_template_words_nouns_or_numbers(world):
    level, semantics = _level(world)
    nouns = {word for spoken in level_nouns(level, semantics).values() for word in spoken.split()}
    for text in instructions(level, semantics):
        words = set(re.sub(r"[.,]", "", text.lower()).split())
        assert {word for word in words - TEMPLATE_WORDS - nouns if not word.isdigit()} == set(), text


def test_shipped_worlds_exercise_the_whole_template_vocabulary():
    seen = {word for world in ZONED for text in instructions(*_level(world)) for word in re.sub(r"[.,]", "", text.lower()).split()}
    assert TEMPLATE_WORDS - seen <= {"fourth", "fifth"}
