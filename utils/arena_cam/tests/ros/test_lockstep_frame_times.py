"""Lockstep stamps a cam riding a run gives its frames."""

from __future__ import annotations

from arena_rclpy_mixins.Time import Time

from arena_cam.client import next_frame_time

PERIOD = Time.from_float(1.0 / 30.0)


def test_the_first_frame_is_one_period_after_the_live_sim_time():
    assert next_frame_time(None, Time.from_float(14.3856), PERIOD) == Time.from_float(14.3856) + PERIOD


def test_frames_advance_one_period_while_the_sim_leads_by_step_rounding():
    last = Time.from_float(90.0)
    assert next_frame_time(last, last + PERIOD + Time.from_float(0.0003), PERIOD) == last + PERIOD


def test_a_stale_anchor_snaps_to_the_live_sim_time():
    assert next_frame_time(Time.from_float(0.0333), Time.from_float(191.1416), PERIOD) == Time.from_float(191.1416) + PERIOD


def test_a_cam_ahead_of_the_sim_keeps_its_own_count():
    last = Time.from_float(50.0)
    assert next_frame_time(last, Time.from_float(49.9), PERIOD) == last + PERIOD
