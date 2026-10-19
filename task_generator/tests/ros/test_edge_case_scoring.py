"""What an episode is scored against: the scope, the arrival window and the progress record."""

from __future__ import annotations

import pytest

pytest.importorskip("arena_humansim")

from task_generator.tasks.obstacles.edge_case.criticality import Sample  # noqa: E402
from task_generator.tasks.obstacles.edge_case.provenance import CaseRecord  # noqa: E402
from task_generator.tasks.obstacles.edge_case.scoring import score_episode, score_scope  # noqa: E402


def _case(**kwargs) -> CaseRecord:
    return CaseRecord(run_seed="a", episode_id=1, world="w", seed=0, **kwargs)


def _drive(peds: dict[str, float], *, seconds: int = 10, speed: float = 1.0) -> list[Sample]:
    """The robot drives along y = 0 at `speed`; each pedestrian stands at (6, y)."""
    return [Sample(t=float(t), robot=(t * speed, 0.0), robot_vel=(speed, 0.0), peds={n: ((6.0, y), (0.0, 0.0), 0.3) for n, y in peds.items()}) for t in range(seconds + 1)]


def test_auto_scope_isolates_the_injected_agent() -> None:
    assert score_scope("auto", _case(injected=1), set(), _drive({"env_0/edge_0": 1.0, "env_0/walker": 3.0})) == ("injected", ["env_0/edge_0"], "")


def test_auto_scope_scores_everyone_when_nothing_was_injected() -> None:
    assert score_scope("auto", _case(injected=0), set(), _drive({"env_0/walker": 3.0})) == ("all", [], "")


def test_auto_scope_scores_the_formation_the_block_owns() -> None:
    samples = _drive({"env_0/d_1": 1.0, "env_0/d_2": 1.5, "env_0/normal_1": 3.0})
    assert score_scope("auto", _case(injected=0), {"d_1", "d_2"}, samples) == ("injected", ["env_0/d_1", "env_0/d_2"], "")


def test_scope_falls_back_loudly_when_the_injected_agent_never_appeared() -> None:
    scope, agents, note = score_scope("injected", _case(injected=2), set(), _drive({"env_0/walker": 3.0}))
    assert (scope, agents) == ("all", []) and "none appeared" in note


def test_explicit_all_scope_never_narrows() -> None:
    assert score_scope("all", _case(injected=1), set(), _drive({"env_0/edge_0": 1.0})) == ("all", [], "")


def test_a_scoped_episode_is_also_scored_ambient() -> None:
    samples = _drive({"edge_0": 0.4, "walker": 2.0})
    result, ambient, _extra = score_episode(samples, encounter=None, agents=["edge_0"], goal=None, route_length=0.0)
    assert result.min_clearance_with == "edge_0" and ambient is not None and ambient.samples == result.samples
    _result, ambient, _extra = score_episode(samples, encounter=None, agents=[], goal=None, route_length=0.0)
    assert ambient is None


def test_the_encounter_point_gives_a_post_encroachment_time() -> None:
    result, _ambient, _extra = score_episode(_drive({"edge_0": 1.0}), encounter=(6.0, 0.5), agents=[], goal=None, route_length=0.0)
    assert result.pet_s is not None


def test_an_episode_with_no_samples_is_not_observed() -> None:
    result, _ambient, extra = score_episode([], encounter=None, agents=[], goal=(10.0, 0.0), route_length=10.0)
    assert not result.observed and extra == {}


def test_the_window_closes_two_seconds_after_the_robot_arrives() -> None:
    # arrives at t = 10 and waits at the goal; a walker brushes it at t = 14, after the window
    samples = _drive({"walker": 3.0}, seconds=10) + [Sample(t=float(t), robot=(10.0, 0.0), peds={"walker": ((10.2 if t >= 14 else 6.0, 0.0 if t >= 14 else 3.0), (0.0, 0.0), 0.3)}) for t in range(11, 20)]
    result, _ambient, extra = score_episode(samples, encounter=None, agents=[], goal=(10.0, 0.0), route_length=10.0)
    assert result.duration_s == pytest.approx(12.0) and not result.collided
    assert extra == {"progress": 1.0, "arrived": True, "ended": "arrived"}


def test_a_robot_cut_short_reports_how_far_it_got() -> None:
    _result, _ambient, extra = score_episode(_drive({"walker": 3.0}, seconds=6), encounter=None, agents=[], goal=(10.0, 0.0), route_length=10.0)
    assert extra == {"progress": 0.6, "arrived": False, "ended": "cut"}
