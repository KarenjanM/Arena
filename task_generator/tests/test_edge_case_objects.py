"""Object timelines: parsing, reference checking, and resolution against a route."""

from __future__ import annotations

import math

import pytest

from task_generator.tasks.obstacles.edge_case.objects import (
    TimelineError,
    load_timeline,
    parse_timeline,
    resolve,
)

_ROUTE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]

_STRAIGHT = [(0.0, 0.0), (20.0, 0.0)]


def test_a_bare_list_of_events_is_accepted() -> None:
    events = parse_timeline([{"action": "spawn", "entity": "cart", "model": "Hospital/SM_TrashCan", "at": {"t": 5.0}}])
    assert len(events) == 1
    assert events[0].entity == "cart"


def test_spawn_needs_a_model() -> None:
    with pytest.raises(TimelineError, match="a spawn needs a model"):
        parse_timeline([{"action": "spawn", "entity": "cart", "at": {"t": 1.0}}])


def test_despawn_must_not_carry_a_model() -> None:
    with pytest.raises(TimelineError, match="names an entity, not a model"):
        parse_timeline(
            [
                {"action": "spawn", "entity": "cart", "model": "m", "at": {"t": 1.0}},
                {"action": "despawn", "entity": "cart", "model": "m", "at": {"t": 2.0}},
            ]
        )


def test_every_event_needs_an_entity_name() -> None:
    with pytest.raises(TimelineError, match="every event needs a name"):
        parse_timeline([{"action": "spawn", "model": "m", "at": {"t": 1.0}}])


def test_t_and_route_fraction_are_mutually_exclusive() -> None:
    with pytest.raises(TimelineError, match="not both"):
        parse_timeline([{"action": "spawn", "entity": "c", "model": "m", "at": {"t": 1.0, "route_fraction": 0.5}}])


def test_a_route_fraction_outside_the_route_is_rejected() -> None:
    with pytest.raises(TimelineError, match=r"within \[0, 1\]"):
        parse_timeline([{"action": "spawn", "entity": "c", "model": "m", "at": {"route_fraction": 1.4}}])


def test_unknown_keys_are_errors_not_warnings() -> None:
    with pytest.raises(TimelineError, match="unknown key"):
        parse_timeline([{"action": "spawn", "entity": "c", "model": "m", "at": {"t": 1}, "place": {"latteral_offset": 1.0}}])


def test_an_empty_timeline_is_rejected() -> None:
    with pytest.raises(TimelineError, match="no events"):
        parse_timeline({"events": []})


def test_despawning_something_never_spawned_is_caught_at_parse_time() -> None:
    with pytest.raises(TimelineError, match="which no earlier event spawns"):
        parse_timeline([{"action": "despawn", "entity": "ghost", "at": {"t": 3.0}}])


def test_spawning_the_same_entity_twice_is_rejected() -> None:
    with pytest.raises(TimelineError, match="spawned twice"):
        parse_timeline(
            [
                {"action": "spawn", "entity": "cart", "model": "m", "at": {"t": 1.0}},
                {"action": "spawn", "entity": "cart", "model": "m", "at": {"t": 2.0}},
            ]
        )


def test_spawn_then_despawn_is_the_case_this_exists_for() -> None:
    events = parse_timeline(
        [
            {"action": "spawn", "entity": "cart", "model": "m", "at": {"route_fraction": 0.4}},
            {"action": "despawn", "entity": "cart", "at": {"t": 30.0}},
        ]
    )
    assert [e.action for e in events] == ["spawn", "despawn"]


def _spawn(entity="cart", **at):
    return {"action": "spawn", "entity": entity, "model": "m", "at": at}


def test_a_route_fraction_becomes_the_robots_arrival_time() -> None:
    (event,) = resolve(parse_timeline([_spawn(route_fraction=0.5)]), _ROUTE, robot_speed=1.0)
    assert event.t == pytest.approx(10.0)
    assert event.pose == pytest.approx((10.0, 0.0))


def test_an_absolute_time_is_taken_as_given() -> None:
    (event,) = resolve(parse_timeline([_spawn(t=4.0)]), _ROUTE, robot_speed=1.0)
    assert event.t == pytest.approx(4.0)


def test_speed_scales_the_arrival_time() -> None:
    (slow,) = resolve(parse_timeline([_spawn(route_fraction=0.5)]), _ROUTE, robot_speed=0.5)
    assert slow.t == pytest.approx(20.0)


def test_placement_can_lead_the_firing_point() -> None:
    events = parse_timeline([{"action": "spawn", "entity": "cart", "model": "m", "at": {"route_fraction": 1 / 3}, "place": {"route_fraction": 2 / 3}}])
    (event,) = resolve(events, _STRAIGHT, robot_speed=1.0)
    assert event.robot_arc == pytest.approx(20 / 3)
    assert event.pose == pytest.approx((40 / 3, 0.0))
    assert event.reveal_distance_m == pytest.approx(20 / 3)


def test_reveal_distance_is_straight_line_not_arc_length() -> None:
    events = parse_timeline([{"action": "spawn", "entity": "cart", "model": "m", "at": {"route_fraction": 1 / 3}, "place": {"route_fraction": 2 / 3}}])
    (event,) = resolve(events, _ROUTE, robot_speed=1.0)
    assert event.pose == pytest.approx((10.0, 10 / 3))
    assert event.reveal_distance_m == pytest.approx(math.dist((20 / 3, 0.0), (10.0, 10 / 3)))
    assert event.reveal_distance_m < 20 / 3, "the bend makes the straight line shorter than the arc"


def test_reveal_distance_is_reported_even_when_it_is_zero() -> None:
    (event,) = resolve(parse_timeline([_spawn(route_fraction=0.5)]), _ROUTE, robot_speed=1.0)
    assert event.reveal_distance_m == pytest.approx(0.0)


def test_lateral_offset_shifts_left_of_travel() -> None:
    events = parse_timeline([{"action": "spawn", "entity": "cart", "model": "m", "at": {"route_fraction": 0.25}, "place": {"lateral_offset": 1.5}}])
    (event,) = resolve(events, _ROUTE, robot_speed=1.0)
    assert event.pose == pytest.approx((5.0, 1.5))


def test_default_yaw_faces_back_down_the_route() -> None:
    (event,) = resolve(parse_timeline([_spawn(route_fraction=0.25)]), _STRAIGHT, robot_speed=1.0)
    assert (math.cos(event.yaw), math.sin(event.yaw)) == pytest.approx((-1.0, 0.0), abs=1e-9)


def test_an_absolute_pose_bypasses_the_route_entirely() -> None:
    events = parse_timeline([{"action": "spawn", "entity": "cart", "model": "m", "at": {"t": 10.0}, "place": {"pose": [8.0, 14.1, 1.57]}}])
    (event,) = resolve(events, _ROUTE, robot_speed=1.0)
    assert event.pose == pytest.approx((8.0, 14.1))
    assert event.yaw == pytest.approx(1.57)


def test_a_despawn_carries_no_pose() -> None:
    events = parse_timeline(
        [
            {"action": "spawn", "entity": "cart", "model": "m", "at": {"t": 5.0}},
            {"action": "despawn", "entity": "cart", "at": {"t": 15.0}},
        ]
    )
    _, despawn = resolve(events, _ROUTE, robot_speed=1.0)
    assert despawn.pose is None


def test_events_are_ordered_by_fire_time() -> None:
    events = parse_timeline(
        [
            {"action": "spawn", "entity": "late", "model": "m", "at": {"t": 30.0}},
            {"action": "spawn", "entity": "early", "model": "m", "at": {"t": 2.0}},
        ]
    )
    assert [e.entity for e in resolve(events, _ROUTE, robot_speed=1.0)] == ["early", "late"]


def test_a_despawn_at_the_same_time_as_its_spawn_stays_after_it() -> None:
    events = parse_timeline(
        [
            {"action": "spawn", "entity": "cart", "model": "m", "at": {"t": 5.0}},
            {"action": "despawn", "entity": "cart", "at": {"t": 5.0}},
        ]
    )
    assert [e.action for e in resolve(events, _ROUTE, robot_speed=1.0)] == ["spawn", "despawn"]


def test_a_route_relative_timeline_needs_a_route() -> None:
    with pytest.raises(TimelineError, match="at least two points"):
        resolve(parse_timeline([_spawn(route_fraction=0.5)]), [(0.0, 0.0)], robot_speed=1.0)


def test_load_timeline_reads_yaml(tmp_path) -> None:
    path = tmp_path / "objects.yaml"
    path.write_text("events:\n  - action: spawn\n    entity: cart\n    model: m\n    at: {route_fraction: 0.5}\n")
    (event,) = load_timeline(str(path))
    assert event.entity == "cart"


def test_missing_timeline_file_is_reported_as_such(tmp_path) -> None:
    with pytest.raises(TimelineError, match="not found"):
        load_timeline(str(tmp_path / "absent.yaml"))


def test_shipped_timelines_are_discoverable_by_name() -> None:
    from task_generator.tasks.obstacles.edge_case.objects import available_timelines

    assert {"corridor_pinch", "doorway_block_clear", "trap_behind"} <= set(available_timelines())


def test_every_shipped_timeline_parses_and_resolves() -> None:
    from task_generator.tasks.obstacles.edge_case.objects import available_timelines

    for name in available_timelines():
        events = load_timeline(name)
        assert events, name
        resolved = resolve(events, _STRAIGHT, robot_speed=1.1)
        assert all(r.pose is not None for r in resolved if r.action == "spawn"), name


def test_an_unknown_name_lists_what_is_shipped() -> None:
    with pytest.raises(TimelineError, match="shipped timelines"):
        load_timeline("no_such_timeline")


def test_a_search_directory_resolves_a_scenario_relative_timeline(tmp_path) -> None:
    from task_generator.tasks.obstacles.edge_case.objects import resolve_timeline_path

    scenario = tmp_path / "scenarios" / "case"
    scenario.mkdir(parents=True)
    (scenario / "objects.yaml").write_text("events:\n  - action: spawn\n    entity: c\n    model: m\n    at: {t: 1.0}\n")

    assert resolve_timeline_path("./objects.yaml", [str(scenario)]) == str(scenario / "objects.yaml")
    assert load_timeline("objects.yaml", [str(scenario)])


def test_an_explicit_path_wins_over_a_shipped_name(tmp_path) -> None:
    path = tmp_path / "corridor_pinch.yaml"
    path.write_text("events:\n  - action: spawn\n    entity: mine\n    model: m\n    at: {t: 1.0}\n")
    (event,) = load_timeline(str(path))
    assert event.entity == "mine"
