"""Sim time floors a cam riding a run puts on its frame captures."""

from __future__ import annotations

from arena_rclpy_mixins.Time import Time

from arena_cam.client import frame_floor

PERIOD = Time.from_float(1.0 / 30.0)


def test_the_first_frame_takes_the_tick_the_run_froze_on():
    assert frame_floor(None, PERIOD) is None


def test_each_frame_waits_one_period_past_the_last_beat():
    last = Time.from_float(90.0)
    assert frame_floor(last, PERIOD) == last + PERIOD


def test_a_stale_beat_lets_the_frame_take_the_live_tick():
    floor = frame_floor(Time.from_float(0.0333), PERIOD)
    assert floor is not None and floor < Time.from_float(191.1416)


def test_a_beat_ahead_of_the_sim_keeps_its_own_count():
    last = Time.from_float(50.0)
    assert frame_floor(last, PERIOD) == last + PERIOD
