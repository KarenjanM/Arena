from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from arena_simulation_setup.generative.scenario.formats import world_roles
from arena_simulation_setup.generative.scenario.propose import client
from arena_simulation_setup.generative.scenario.propose.zone_roles import (
    Proposal,
    build_prompt,
    classify_tail,
    merge,
    parse,
    unknown_names,
)
from arena_simulation_setup.generative.scenario.zones import Role, stated_roles, zones_by_role
from arena_simulation_setup.kg import FORMAT, load

SHIPPED = Path(__file__).resolve().parents[2] / "worlds"


def zone(name, contents=(), corners=((0, 0), (4, 0), (4, 4), (0, 4))):
    return {
        "name": name,
        "corners": [{"x": x, "y": y} for x, y in corners],
        "entities": {"static": [{"name": f"e{i}", "model": m} for i, m in enumerate(contents)]},
    }


ZONES = [
    zone("main_hallway"),
    zone("kitchen"),
    zone("top_room_0", contents=["Common/Furniture/desk"]),
    zone("3_study"),
]


def test_only_the_tail_is_asked_about() -> None:
    # The substring rules cover 99 of the tree's 116 names for free; asking a model to confirm
    # that `kitchen` is a break zone is the expensive random number generator to avoid.
    assert unknown_names(ZONES) == ["3_study", "top_room_0"]


def test_a_stated_name_is_never_asked_again(kg) -> None:
    _write_kg(kg, [_fact("3_study", "work")])
    assert unknown_names(ZONES, stated_roles(load(kg))) == ["top_room_0"]


def test_the_prompt_carries_the_whole_vocabulary_but_asks_only_the_tail() -> None:
    prompt = build_prompt(ZONES, ["top_room_0"], world="w")
    # The rest of the world is the evidence: `top_room_0` beside `main_hallway` reads
    # differently from `top_room_0` beside `bedroom_1`.
    assert "main_hallway" in prompt
    assert "transit" in prompt
    assert "desk" in prompt
    assert prompt.count("3_study") == 1


def test_a_zones_size_and_contents_reach_the_prompt() -> None:
    prompt = build_prompt(ZONES, ["top_room_0"], world="w")
    assert "size_m" in prompt
    assert "contents" in prompt


GOOD = '{"top_room_0": {"role": "work", "why": "holds a desk"}}'


def test_a_well_formed_answer_parses() -> None:
    proposals, complaints = parse(GOOD, ["top_room_0"], model="m", sha="s")
    assert not complaints
    assert proposals == [Proposal(zone="top_room_0", role=Role.WORK, why="holds a desk", model="m", prompt_sha="s")]


def test_a_fenced_answer_parses() -> None:
    proposals, _ = parse(f"```json\n{GOOD}\n```", ["top_room_0"], model="m", sha="s")
    assert [p.zone for p in proposals] == ["top_room_0"]


def test_an_unasked_zone_is_dropped_not_accepted() -> None:
    proposals, complaints = parse(
        '{"kitchen": {"role": "work"}}',
        ["top_room_0"],
        model="m",
        sha="s",
    )
    assert proposals == []
    assert any("not one of the zones asked" in c for c in complaints)
    assert any("no answer returned" in c for c in complaints)


def test_an_invented_role_is_refused_not_coerced() -> None:
    proposals, complaints = parse(
        '{"top_room_0": {"role": "storage_cupboard"}}',
        ["top_room_0"],
        model="m",
        sha="s",
    )
    assert proposals == []
    assert any("is not a role" in c for c in complaints)


def test_prose_instead_of_json_is_a_complaint_not_a_crash() -> None:
    proposals, complaints = parse("I think it is a work zone!", ["top_room_0"], model="m", sha="s")
    assert proposals == []
    assert any("not JSON" in c for c in complaints)


@pytest.fixture
def kg(tmp_path: Path) -> Path:
    return tmp_path / "kg.yaml"


def _fact(zone: str, role: str, **params: str) -> dict:
    return {"subject": f"room:{zone}", "verb": "has_purpose", "object": f"purpose:{role}", **({"params": params} if params else {})}


def _write_kg(path: Path, facts: list[dict], uses: tuple[str, ...] = ("core@2", "space@1")) -> None:
    rooms = sorted({f["subject"] for f in facts})
    path.write_text(yaml.safe_dump({"format": FORMAT, "uses": list(uses), "entities": [{"id": r} for r in rooms], "facts": facts}))


def _purposes(path: Path) -> list[tuple[str, str, dict]]:
    return [(f.subject, str(f.object), dict(f.params)) for f in load(path).facts if f.verb == "has_purpose"]


def test_a_confident_proposal_is_written_as_a_fact_with_its_provenance(kg) -> None:
    written, _ = merge([Proposal(zone="a", role=Role.WORK, why="w", model="m", prompt_sha="s")], kg)
    assert written == 1
    scene = load(kg)
    assert scene.uses == ["core@2", "space@1"]
    assert scene.validate() == []
    assert _purposes(kg) == [("room:a", "purpose:work", {"by": "llm", "model": "m", "prompt_sha": "s", "why": "w"})]


def test_a_proposal_joins_an_existing_kg_without_touching_the_rest(kg) -> None:
    kg.write_text(yaml.safe_dump({"format": FORMAT, "uses": ["core@2"], "entities": [{"id": "room:a", "props": {"description": "Den"}}, {"id": "obj:desk"}], "facts": [{"subject": "room:a", "verb": "contains", "object": "obj:desk"}]}))
    assert merge([Proposal(zone="a", role=Role.WORK, model="m")], kg)[0] == 1
    scene = load(kg)
    assert scene.uses == ["core@2", "space@1"]
    assert scene.get_entity("room:a").props == {"description": "Den"}
    assert scene.get_fact("room:a", "contains", "obj:desk") is not None
    assert stated_roles(scene) == {"a": Role.WORK}


def test_an_unknown_answer_is_left_for_a_human(kg) -> None:
    written, kept = merge([Proposal(zone="a", role=Role.UNKNOWN, model="m")], kg)
    assert written == 0
    assert any("left for a human" in k for k in kept)
    assert not kg.exists()


@pytest.mark.parametrize("params", [{}, {"by": "human"}])
def test_a_human_fact_is_never_overwritten(kg, params) -> None:
    _write_kg(kg, [_fact("a", "private", **params)])
    written, kept = merge([Proposal(zone="a", role=Role.WORK, model="m")], kg)
    assert written == 0
    assert any("reviewed by a human" in k for k in kept)
    assert stated_roles(load(kg)) == {"a": Role.PRIVATE}


def test_rerunning_with_the_same_answer_writes_nothing(kg) -> None:
    proposals = [Proposal(zone="a", role=Role.WORK, model="m", prompt_sha="s")]
    assert merge(proposals, kg)[0] == 1
    before = kg.read_text()
    assert merge(proposals, kg)[0] == 0
    assert kg.read_text() == before


def test_a_new_model_answer_replaces_the_old_one(kg) -> None:
    merge([Proposal(zone="a", role=Role.WORK, model="m")], kg)
    assert merge([Proposal(zone="a", role=Role.BREAK, model="m2")], kg)[0] == 1
    assert _purposes(kg) == [("room:a", "purpose:break", {"by": "llm", "model": "m2"})]
    assert load(kg).get_entity("purpose:work") is None


def test_no_kg_states_no_roles() -> None:
    assert stated_roles(None) == {}


def test_a_human_fact_wins_over_a_model_fact(kg) -> None:
    _write_kg(kg, [_fact("a", "work", by="llm"), _fact("a", "break")])
    assert stated_roles(load(kg)) == {"a": Role.BREAK}


def test_two_human_roles_for_one_zone_are_refused(kg) -> None:
    _write_kg(kg, [_fact("a", "work"), _fact("a", "break")])
    with pytest.raises(ValueError, match=r"room:a: has_purpose states the roles \['break', 'work'\]"):
        stated_roles(load(kg))


def test_a_purpose_that_is_no_role_is_refused(kg) -> None:
    _write_kg(kg, [_fact("a", "storage_cupboard")])
    with pytest.raises(ValueError, match=r"expected purpose:<work\|break"):
        stated_roles(load(kg))


def test_a_stated_role_reaches_the_role_split(kg) -> None:
    _write_kg(kg, [_fact("top_room_0", "work", by="llm")])
    by_role = zones_by_role(ZONES, stated_roles(load(kg)))
    assert "top_room_0" in by_role[Role.WORK]
    assert "top_room_0" not in by_role[Role.UNKNOWN]


def test_a_world_reads_its_roles_from_its_kg(tmp_path: Path) -> None:
    root = tmp_path / "w"
    root.mkdir()
    polygon = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]
    (root / "world.yaml").write_text(yaml.safe_dump({"format": "arena-world/2", "levels": [{"id": "0", "zones": [{"name": "top_room_0", "polygon": polygon, "floor": False}]}]}))
    assert world_roles(root) == {}
    assert merge([Proposal(zone="top_room_0", role=Role.WORK, model="m")], root / "kg.yaml")[0] == 1
    assert world_roles(root) == {"top_room_0": Role.WORK}


def test_the_shipped_study_is_a_work_zone() -> None:
    assert world_roles(SHIPPED / "three_storied_residential")["3_study"] is Role.WORK


def test_nothing_to_ask_asks_nothing() -> None:
    assert classify_tail([zone("kitchen")], world="w", names=[]) == ([], [])


@pytest.mark.skipif(client.available(), reason="a model is configured here")
def test_no_configured_model_is_a_clear_refusal_not_a_guess() -> None:
    with pytest.raises(client.Unavailable, match="no LLM configured"):
        client.ask("s", "u")


def test_the_prompt_hash_changes_with_the_prompt() -> None:
    assert client.prompt_sha256("a", "b") == client.prompt_sha256("a", "b")
    assert client.prompt_sha256("a", "b") != client.prompt_sha256("ab", "")
