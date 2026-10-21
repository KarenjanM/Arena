from __future__ import annotations

import pytest

from arena_simulation_setup.generative.scenario.promptgen.lexicon import match, score, when_in


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
