from __future__ import annotations

import math

import pytest

pytest.importorskip("rclpy")


def _phases(traversals: int):
    from task_generator.shared import Pose, Position
    from task_generator.tasks.robots.edge_case.impl import TM_EdgeCase

    return TM_EdgeCase.traversal_phases(Pose(Position(0.0, 0.0)), Pose(Position(10.0, 0.0)), traversals)


def test_single_traversal_is_one_leg() -> None:
    ph = _phases(1)
    assert len(ph) == 1
    assert ph[0].pose.position.x == 10.0


def test_n_traversals_is_2n_minus_1_legs() -> None:
    assert [len(_phases(n)) for n in (1, 2, 3, 5)] == [1, 3, 5, 9]


def test_legs_alternate_and_end_at_the_goal() -> None:
    xs = [p.pose.position.x for p in _phases(3)]
    assert xs == [10.0, 0.0, 10.0, 0.0, 10.0]


def test_zero_traversals_is_rejected() -> None:
    with pytest.raises(ValueError, match="traversals must be >= 1"):
        _phases(0)


def _fresh():
    from task_generator.tasks.robots.edge_case.impl import _Progress

    return _Progress()


def _blocked(progress, distance, now, *, min_improvement=0.25, timeout=15.0):
    from task_generator.tasks.robots.edge_case.impl import TM_EdgeCase

    return TM_EdgeCase.is_blocked(progress, distance, now, min_improvement=min_improvement, timeout=timeout)


def test_steady_progress_never_blocks() -> None:
    p = _fresh()
    for i in range(30):
        assert _blocked(p, distance=20.0 - i, now=float(i)) is False


def test_stalled_robot_blocks_after_the_timeout() -> None:
    p = _fresh()
    assert _blocked(p, 5.0, 0.0) is False
    assert _blocked(p, 5.0, 14.0) is False
    assert _blocked(p, 5.0, 15.0) is True


def test_a_short_pause_does_not_block() -> None:
    p = _fresh()
    _blocked(p, 10.0, 0.0)
    assert _blocked(p, 10.0, 8.0) is False
    assert _blocked(p, 9.0, 9.0) is False
    assert _blocked(p, 9.0, 20.0) is False
    assert _blocked(p, 9.0, 24.1) is True


def test_progress_must_exceed_the_threshold_to_count() -> None:
    p = _fresh()
    _blocked(p, 5.0, 0.0)
    for i in range(1, 16):
        assert _blocked(p, 5.0 - i * 0.01, float(i)) is (i >= 15)


def test_moving_away_from_the_goal_is_not_progress() -> None:
    p = _fresh()
    _blocked(p, 5.0, 0.0)
    assert _blocked(p, 8.0, 10.0) is False
    assert _blocked(p, 8.0, 15.0) is True


def test_best_distance_is_a_high_water_mark() -> None:
    p = _fresh()
    _blocked(p, 10.0, 0.0)
    _blocked(p, 4.0, 1.0)
    assert _blocked(p, 9.0, 5.0) is False
    assert _blocked(p, 4.1, 10.0) is False
    assert _blocked(p, 4.1, 16.1) is True


def test_reset_clears_the_watchdog() -> None:
    p = _fresh()
    _blocked(p, 5.0, 0.0)
    _blocked(p, 5.0, 20.0)
    p.reset(100.0)
    assert p.best_distance == math.inf
    assert _blocked(p, 5.0, 110.0) is False


def test_retries_repeat_each_leg_in_place() -> None:
    from task_generator.shared import Pose, Position
    from task_generator.tasks.robots.edge_case.impl import TM_EdgeCase

    phases = TM_EdgeCase.traversal_phases(Pose(Position(0.0, 0.0)), Pose(Position(10.0, 0.0)), 2, retries=3)
    xs = [p.pose.position.x for p in phases]
    assert xs == [10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0]
    with pytest.raises(ValueError, match="retries"):
        TM_EdgeCase.traversal_phases(Pose(Position(0.0, 0.0)), Pose(Position(10.0, 0.0)), 1, retries=0)


def _plan(phases, hold=0.0, within=0.6, tolerance=0.3):
    from task_generator.shared import Pose, Position
    from task_generator.tasks.robots.edge_case.impl import plan_legs

    return plan_legs(Pose(Position(0.0, 0.0)), phases, traversals=1, retries=1, hold=hold, accept_within=within, goal_tolerance=tolerance)


def test_the_scenario_route_becomes_legs_with_the_wider_tolerance() -> None:
    from task_generator.shared import Pose, Position
    from task_generator.tasks.robots.request import GoToPhase

    goal = Pose(Position(10.0, 0.0))
    (leg,) = _plan([GoToPhase(pose=Pose(Position(5.0, 0.0))), GoToPhase(pose=goal)])
    assert leg.pose == goal and leg.tolerance_radius == 0.6
    (leg,) = _plan([GoToPhase(pose=goal)], within=0.0)
    assert leg.tolerance_radius is None
    assert _plan([]) == []


def test_a_hold_is_a_pose_less_first_phase() -> None:
    from task_generator.shared import Pose, Position
    from task_generator.tasks.robots.request import GoToPhase

    hold, leg = _plan([GoToPhase(pose=Pose(Position(10.0, 0.0)))], hold=15.0)
    assert hold.pose is None and hold.hold and hold.hold_time == 15.0 and leg.pose.position.x == 10.0


def test_a_robot_holding_inside_its_tolerance_is_never_blocked() -> None:
    from task_generator.shared import Pose, Position
    from task_generator.tasks.robots.edge_case.impl import _Progress, leg_blocked
    from task_generator.tasks.robots.request import GoToPhase

    here = Pose(Position(1.1, 1.0))
    holding = GoToPhase(pose=Pose(Position(1.0, 1.0)), tolerance_radius=0.5, hold_time=20.0)
    progress = _Progress()
    assert all(leg_blocked(progress, here, holding, t, min_improvement=0.25, timeout=15.0) is None for t in (0.0, 10.0, 30.0, 60.0))
    tight = GoToPhase(pose=Pose(Position(1.0, 1.0)), tolerance_radius=0.0)
    assert leg_blocked(progress, here, tight, 90.0, min_improvement=0.25, timeout=15.0) is None
    assert leg_blocked(progress, here, tight, 120.0, min_improvement=0.25, timeout=15.0) == pytest.approx(0.1)
