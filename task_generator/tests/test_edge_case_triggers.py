"""Firing a case on where the robot is rather than on a clock."""

from __future__ import annotations

from task_generator.tasks.obstacles.edge_case.triggers import Trigger, centroid, distance_to


def test_a_within_trigger_fires_on_approach() -> None:
    assert Trigger(robot_within=3.0).satisfied(2.9) is True
    assert Trigger(robot_within=3.0).satisfied(3.1) is False


def test_a_beyond_trigger_is_the_mirror() -> None:
    assert Trigger(robot_beyond=3.0).satisfied(4.0) is True
    assert Trigger(robot_beyond=3.0).satisfied(1.0) is False


def test_an_unknown_distance_never_fires() -> None:
    assert Trigger(robot_within=3.0).satisfied(None) is False
    assert Trigger(robot_within=3.0).satisfied(float("nan")) is False


def test_the_description_names_the_side_and_the_point() -> None:
    described = Trigger(robot_within=3.0).describe((1.0, 2.0))
    assert "within 3m" in described and "1.00,2.00" in described


def test_the_centroid_is_the_default_reference() -> None:
    assert centroid([(0.0, 0.0), (2.0, 0.0), (1.0, 3.0)]) == (1.0, 1.0)


def test_an_empty_crowd_has_no_centroid() -> None:
    assert centroid([]) is None


def test_distance_is_unknown_when_either_end_is() -> None:
    assert distance_to(None, (0.0, 0.0)) is None
    assert distance_to((0.0, 0.0), None) is None
    assert distance_to((3.0, 4.0), (0.0, 0.0)) == 5.0


def test_an_arrival_trigger_fires_when_the_robot_is_due() -> None:
    eta = Trigger(robot_eta=4.0)
    assert eta.satisfied(3.0, 1.0) and not eta.satisfied(5.0, 1.0)
    assert eta.satisfied(1.5, 0.4) and not eta.satisfied(2.0, 0.4)
    assert not eta.satisfied(1.0, None)
    assert eta.satisfied(0.3, 0.0)
