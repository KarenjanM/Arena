from __future__ import annotations

import math

import pytest
from arena_simulation_setup.generative.scenario.promptgen.encounter import (
    Encounter,
    GeometryError,
    angle_offsets,
    encounter_fractions,
    first_blocked,
    rotate,
    solve_encounter,
)

ROBOT_START = (0.0, 0.0)
ROBOT_GOAL = (20.0, 0.0)


def solve(**kwargs: object) -> Encounter:
    params: dict[str, object] = {"robot_speed": 1.0, "ped_speed": 1.0}
    params.update(kwargs)
    return solve_encounter((ROBOT_START, ROBOT_GOAL), **params)  # type: ignore[arg-type]


def test_both_reach_the_encounter_point_together() -> None:
    enc = solve(encounter_at=0.5, approach_angle=180.0)

    assert enc.t_encounter == pytest.approx(10.0)
    assert enc.encounter == pytest.approx((10.0, 0.0))
    assert math.dist(enc.spawn, enc.encounter) == pytest.approx(enc.ped_speed * enc.t_encounter)


def test_co_arrival_offset_delays_the_pedestrian_by_starting_it_further_out() -> None:
    base = solve(encounter_at=0.5, approach_angle=180.0)
    late = solve(encounter_at=0.5, approach_angle=180.0, co_arrival_offset=2.0)

    extra = math.dist(late.spawn, late.encounter) - math.dist(base.spawn, base.encounter)
    assert extra == pytest.approx(2.0)
    assert late.designed_pet == pytest.approx(2.0)


def test_negative_offset_past_the_encounter_is_refused() -> None:
    with pytest.raises(GeometryError, match="cancels the pedestrian's approach"):
        solve(encounter_at=0.5, co_arrival_offset=-10.0)


def test_head_on_spawns_ahead_of_the_robot_on_its_own_path() -> None:
    enc = solve(encounter_at=0.5, approach_angle=180.0)
    assert enc.spawn[0] > enc.encounter[0]
    assert enc.spawn[1] == pytest.approx(0.0)
    assert enc.goal[0] < enc.encounter[0]


def test_crossing_spawns_off_the_robots_path() -> None:
    enc = solve(encounter_at=0.5, approach_angle=90.0)
    assert enc.encounter == pytest.approx((10.0, 0.0))
    assert enc.spawn[1] < 0.0
    assert enc.goal[1] > 0.0
    assert enc.spawn[0] == pytest.approx(enc.encounter[0])


def test_same_direction_spawns_behind_the_encounter() -> None:
    enc = solve(encounter_at=0.5, approach_angle=0.0, ped_speed=0.5)
    assert enc.spawn[0] < enc.encounter[0]
    assert enc.spawn[1] == pytest.approx(0.0)


def test_rotate_by_180_negates() -> None:
    assert rotate((1.0, 0.0), 180.0) == pytest.approx((-1.0, 0.0))
    assert rotate((0.0, 1.0), 90.0) == pytest.approx((-1.0, 0.0))


def test_goal_is_beyond_the_encounter_never_short_of_it() -> None:
    enc = solve(encounter_at=0.5, approach_angle=180.0, lead_out=4.0)
    assert math.dist(enc.encounter, enc.goal) == pytest.approx(4.0)
    assert math.dist(enc.spawn, enc.goal) > math.dist(enc.spawn, enc.encounter)


def test_encounter_at_places_the_meeting_along_the_leg() -> None:
    early = solve(encounter_at=0.25)
    late = solve(encounter_at=0.75)
    assert early.encounter[0] == pytest.approx(5.0)
    assert late.encounter[0] == pytest.approx(15.0)
    assert early.t_encounter == pytest.approx(5.0)
    assert late.t_encounter == pytest.approx(15.0)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5])
def test_encounter_at_must_be_strictly_inside_the_leg(bad: float) -> None:
    with pytest.raises(GeometryError, match="encounter_at"):
        solve(encounter_at=bad)


def test_coincident_start_and_goal_is_refused() -> None:
    with pytest.raises(GeometryError, match="no route to intercept"):
        solve_encounter(((1.0, 1.0), (1.0, 1.0)), robot_speed=1.0, ped_speed=1.0)


def blocked_north(pt: tuple[float, float]) -> bool:
    return pt[1] <= 1e-9


def test_unobstructed_case_gets_exactly_what_it_asked_for() -> None:
    enc = solve(encounter_at=0.5, approach_angle=180.0, is_valid=lambda _pt: True)
    assert enc.angle_achieved == pytest.approx(180.0)
    assert enc.retries == 0
    assert not enc.deviated


def test_blocked_angle_is_retried_and_the_deviation_is_reported() -> None:
    enc = solve(encounter_at=0.5, approach_angle=90.0, angle_search=180.0, angle_step=15.0, is_valid=blocked_north)

    assert enc.deviated
    assert enc.retries > 0
    assert enc.angle_requested == pytest.approx(90.0)
    assert enc.spawn[1] <= 1e-9
    assert enc.goal[1] <= 1e-9


def test_angle_search_zero_refuses_rather_than_deviating() -> None:
    with pytest.raises(GeometryError, match="no navigable interception"):
        solve(encounter_at=0.5, approach_angle=90.0, angle_search=0.0, is_valid=blocked_north)


def only_the_leg_line(pt: tuple[float, float]) -> bool:
    return abs(pt[1]) < 1e-6 and -0.1 <= pt[0] <= 20.1


def test_abort_reason_names_the_factors_tried() -> None:
    with pytest.raises(GeometryError) as excinfo:
        solve(encounter_at=0.5, approach_angle=90.0, angle_search=0.0, is_valid=only_the_leg_line)
    message = str(excinfo.value)
    assert "no navigable interception" in message
    assert "angle=90" in message
    assert "encounter_at=0.5" in message


def test_spawn_on_top_of_the_robot_is_rejected() -> None:
    with pytest.raises(GeometryError, match="no navigable interception"):
        solve(encounter_at=0.5, approach_angle=0.0, ped_speed=1.0, angle_search=0.0, min_robot_clearance=1.5)


def test_angle_offsets_try_the_request_first_then_alternate_outward() -> None:
    assert list(angle_offsets(30.0, 15.0)) == [0.0, 15.0, -15.0, 30.0, -30.0]
    assert list(angle_offsets(0.0, 15.0)) == [0.0]
    assert list(angle_offsets(30.0, 0.0)) == [0.0]


def test_segment_scan_catches_a_wall_between_free_endpoints() -> None:

    def gap(pt: tuple[float, float]) -> bool:
        return not (4.0 < pt[0] < 6.0)

    assert first_blocked((0.0, 0.0), (3.0, 0.0), gap) is None
    assert first_blocked((0.0, 0.0), (10.0, 0.0), gap) is not None


def test_no_predicate_means_no_filtering() -> None:
    assert first_blocked((0.0, 0.0), (10.0, 0.0), None) is None


def test_designed_ttc_is_the_encounter_time() -> None:
    enc = solve(encounter_at=0.3)
    assert enc.designed_ttc == pytest.approx(enc.t_encounter)


def test_encounter_sweep_tries_the_request_first_then_spreads_both_ways() -> None:
    got = list(encounter_fractions(0.5))
    assert got[0] == 0.5
    assert got[1:5] == [0.45, 0.55, 0.4, 0.6]
    assert min(got) <= 0.15, "must reach approaches short enough for a furnished room"
    assert max(got) >= 0.85
    assert all(0.1 <= v <= 0.9 for v in got)
    assert len(got) == len(set(got))


def test_encounter_sweep_stays_in_range_for_an_off_centre_request() -> None:
    got = list(encounter_fractions(0.85))
    assert got[0] == 0.85
    assert all(0.1 <= v <= 0.9 for v in got)


def test_a_corridor_only_reachable_by_a_short_approach_is_solved() -> None:
    E = (10.0, 0.0)

    def near_encounter(pt: tuple[float, float]) -> bool:
        return math.dist(pt, E) <= 2.0

    enc = solve_encounter(
        (ROBOT_START, ROBOT_GOAL),
        robot_speed=1.0,
        ped_speed=1.0,
        encounter_at=0.5,
        angle_search=0.0,
        lead_out=1.0,
        is_valid=near_encounter,
    )
    assert math.dist(enc.spawn, enc.encounter) == pytest.approx(3.0)
    with pytest.raises(GeometryError):
        solve_encounter(
            (ROBOT_START, ROBOT_GOAL),
            robot_speed=1.0,
            ped_speed=1.0,
            encounter_at=0.5,
            angle_search=0.0,
            lead_out=1.0,
            is_valid=lambda pt: math.dist(pt, E) <= 1.0,
        )

    enc = solve_encounter(
        (ROBOT_START, ROBOT_GOAL),
        robot_speed=1.0,
        ped_speed=1.0,
        encounter_at=0.5,
        angle_search=0.0,
        lead_out=1.0,
        is_valid=lambda pt: pt[0] <= 6.0,
    )
    assert enc.encounter_at_achieved < enc.encounter_at_requested
    assert math.dist(enc.spawn, enc.encounter) < 6.0
    assert enc.deviated, "a moved encounter must be reported, not silently used"


def test_a_leg_through_a_wall_is_diagnosed_not_buried_in_candidates() -> None:

    def far_away(pt: tuple[float, float]) -> bool:
        return math.dist(pt, (50.0, 50.0)) < 2.0

    with pytest.raises(GeometryError, match="straight-line leg .* is itself blocked"):
        solve(encounter_at=0.5, is_valid=far_away)


def test_a_leg_blocked_midway_still_solves_near_its_open_end() -> None:

    def wall_band(pt: tuple[float, float]) -> bool:
        return not (8.0 < pt[0] < 12.0)

    enc = solve(encounter_at=0.5, approach_angle=180.0, is_valid=wall_band)
    assert enc.deviated
    assert not (8.0 < enc.encounter[0] < 12.0)


def test_forbidden_endpoints_do_not_read_as_a_blocked_leg() -> None:

    def leg_interior_only(pt: tuple[float, float]) -> bool:
        return abs(pt[1]) < 1e-6 and 1.0 <= pt[0] <= 19.0

    with pytest.raises(GeometryError) as excinfo:
        solve(encounter_at=0.5, approach_angle=90.0, angle_search=0.0, is_valid=leg_interior_only)
    message = str(excinfo.value)
    assert "straight-line leg" not in message, "forbidden endpoints must not read as a blocked leg"
    assert "no navigable interception" in message


def test_a_lead_in_that_does_not_fit_starts_closer_and_later() -> None:
    route = [(0.0, 0.0), (20.0, 0.0)]

    def is_valid(p: tuple[float, float]) -> bool:
        return p[0] >= -3.0

    enc = solve_encounter(route, robot_speed=0.5, ped_speed=2.0, approach_angle=0.0, encounter_at=0.5, is_valid=is_valid, angle_search=0.0)
    assert enc.spawn[0] >= -3.0
    assert enc.t_encounter == pytest.approx(20.0)


def test_a_spawn_inside_the_goals_reserved_disc_is_moved_closer_and_later() -> None:
    route = ((0.0, 0.0), (13.0, 0.0))

    def spawn_valid(pt):
        return math.dist(pt, (13.0, 0.0)) > 2.0

    enc = solve_encounter(route, robot_speed=1.0, ped_speed=1.2, approach_angle=180.0, encounter_at=0.5, is_valid=None, spawn_valid=spawn_valid, angle_search=0.0)
    assert math.dist(enc.spawn, (13.0, 0.0)) > 2.0, f"spawn {enc.spawn} still inside the reserved disc"
    assert math.dist(enc.spawn, enc.encounter) >= 3.0


def test_the_solver_says_when_every_spawn_is_reserved() -> None:
    route = ((0.0, 0.0), (8.0, 0.0))
    with pytest.raises(GeometryError, match="reserved"):
        solve_encounter(route, robot_speed=1.0, ped_speed=1.0, approach_angle=180.0, encounter_at=0.5, is_valid=None, spawn_valid=lambda pt: False, angle_search=0.0)


def test_encounter_on_a_bent_route_lands_on_the_route() -> None:
    route = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    enc = solve_encounter(route, robot_speed=1.0, ped_speed=1.0, encounter_at=0.75)

    assert enc.encounter == pytest.approx((10.0, 5.0))
    assert enc.t_encounter == pytest.approx(15.0)
    assert math.dist(enc.encounter, (7.5, 7.5)) > 2.0


def test_head_on_follows_the_local_heading_after_a_corner() -> None:
    route = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
    enc = solve_encounter(route, robot_speed=1.0, ped_speed=1.0, encounter_at=0.75, approach_angle=180.0)

    assert enc.spawn[1] > enc.encounter[1]
    assert enc.spawn[0] == pytest.approx(enc.encounter[0])
