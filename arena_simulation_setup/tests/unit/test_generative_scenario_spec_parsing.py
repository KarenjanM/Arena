from __future__ import annotations

import pytest

from arena_simulation_setup.generative.scenario.promptgen.lexicon import match, score, when_in
from arena_simulation_setup.generative.scenario.promptgen.spec import SchemaError, parse_d, parse_selection


@pytest.mark.parametrize(
    "text",
    [
        "A surgeon crosses the hall.",
        "The boxes are unpacked in the corridor.",
        "Someone mentions a crusher in the basement.",
    ],
)
def test_lexicon_phrases_do_not_match_inside_words(text) -> None:
    assert score(text) == {}
    assert match(text) is None


@pytest.mark.parametrize(
    "text, case",
    [
        ("Everyone is evacuated from the ward.", "evacuation"),
        ("The crowd surges towards the lifts.", "stampede"),
        ("The workers rushed back, shoving.", "stampede"),
    ],
)
def test_lexicon_stems_and_inflections_still_match(text, case) -> None:
    got = match(text)
    assert got is not None and got.case == case


def test_busy_multiplier_words_match_whole_words_outside_time_phrases() -> None:
    assert match("The floor is busy and everyone is here.").k == 2.0
    assert match("The floor gets busy a little later.").k == 2.0
    assert match("The floor gets a little busy.").k == 1.5
    assert match("The floor is very busy.").k == 3.0


def test_a_word_containing_a_case_phrase_does_not_own_its_sentence() -> None:
    assert when_in("A surgeon walks by after ten seconds.", exclude_cases=("stampede",)).as_dict() == {"at": 10.0}


@pytest.mark.parametrize("raw, names", [("ped_1", ("ped_1",)), (["ped_1", "ped_2"], ("ped_1", "ped_2"))])
def test_names_and_reuse_take_a_list_or_one_string(raw, names) -> None:
    assert parse_selection({"names": raw}, where="B.select").names == names
    assert parse_d({"case": "stubborn", "reuse": raw}).reuse == names


@pytest.mark.parametrize("raw", [5, {"ped_1": 1}])
def test_names_and_reuse_reject_a_non_list(raw) -> None:
    with pytest.raises(SchemaError, match=r"B\.select\.names"):
        parse_selection({"names": raw}, where="B.select")
    with pytest.raises(SchemaError, match=r"D\.reuse"):
        parse_d({"case": "stubborn", "reuse": raw})
