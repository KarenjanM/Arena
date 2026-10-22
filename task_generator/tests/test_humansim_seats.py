from __future__ import annotations

import math

import pytest

from task_generator.simulators.human.arena_humansim.arena_humansim import reach_radius, seat_capacity

SOFA = [
    {"bbox": [[-0.46, 0.48], [0.03, 0.72], [0.0, 0.47]], "yaw": "-y"},
    {"bbox": [[-0.98, -0.48], [-0.50, 0.03], [0.0, 0.47]], "yaw": "+x"},
    {"bbox": [[-1.30, -0.56], [-1.13, -0.26], [0.0, 0.47]], "yaw": ["+x", "-y"]},
]


def test_capacity_counts_sit_slots_that_can_be_taken_together():
    assert seat_capacity({}, SOFA) == 2


def test_capacity_prefers_the_scenario_capacity_then_its_seats():
    assert seat_capacity({"capacity": 5}, SOFA) == 5
    assert seat_capacity({"seats": [{"x": 0.0, "y": 0.0}] * 3}, SOFA) == 3
    assert seat_capacity({"seats": []}, SOFA) == 1


def test_capacity_is_one_without_usable_slots():
    assert seat_capacity({}, []) == 1
    assert seat_capacity({}, [{"bbox": [[0.5, 0.0], [0.0, 0.5]], "yaw": 0}]) == 1


def test_reach_radius_covers_the_footprint_from_beside_it():
    assert reach_radius([[-0.7, 0.7], [-0.3, 0.3], [0.0, 0.9]]) == pytest.approx(math.hypot(0.7, 0.3) + 0.3)
    assert reach_radius([[0.0, 2.0], [-0.5, 0.0], [0.0, 0.5]]) == pytest.approx(math.hypot(2.0, 0.5) + 0.3)
