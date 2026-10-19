"""Criticality metrics are computed from the trajectory, not from the designed encounter."""

from __future__ import annotations

import json
import math

import pytest

from task_generator.tasks.obstacles.edge_case.provenance import CaseRecord
from task_generator.tasks.obstacles.edge_case.criticality import (
    Criticality,
    Sample,
    post_encroachment_time,
    score,
    time_to_collision,
)


def test_head_on_closing_gives_the_expected_time() -> None:
    ttc = time_to_collision((0.0, 0.0), (1.0, 0.0), 0.3, (10.0, 0.0), (-1.0, 0.0), 0.3)
    assert ttc == pytest.approx((10.0 - 0.6) / 2.0)


def test_parallel_motion_never_collides() -> None:
    assert time_to_collision((0.0, 0.0), (1.0, 0.0), 0.3, (0.0, 5.0), (1.0, 0.0), 0.3) is None


def test_receding_pair_never_collides() -> None:
    assert time_to_collision((0.0, 0.0), (-1.0, 0.0), 0.3, (10.0, 0.0), (1.0, 0.0), 0.3) is None


def test_already_touching_is_zero_not_none() -> None:
    assert time_to_collision((0.0, 0.0), (1.0, 0.0), 0.3, (0.4, 0.0), (0.0, 0.0), 0.3) == 0.0


def test_no_relative_motion_is_not_a_collision() -> None:
    assert time_to_collision((0.0, 0.0), (1.0, 0.0), 0.3, (5.0, 0.0), (1.0, 0.0), 0.3) is None


def test_clearance_is_surface_to_surface() -> None:
    s = [Sample(t=0.0, robot=(0.0, 0.0), robot_radius=0.3, peds={"p": ((0.6, 0.0), (0.0, 0.0), 0.3)})]
    out = score(s)
    assert out.min_clearance_m == pytest.approx(0.0)
    assert out.collided is True


def test_clearance_tracks_the_closest_moment_and_names_it() -> None:
    s = [
        Sample(t=0.0, robot=(0.0, 0.0), peds={"a": ((5.0, 0.0), (0.0, 0.0), 0.3)}),
        Sample(t=1.0, robot=(0.0, 0.0), peds={"a": ((2.0, 0.0), (0.0, 0.0), 0.3)}),
        Sample(t=2.0, robot=(0.0, 0.0), peds={"a": ((4.0, 0.0), (0.0, 0.0), 0.3)}),
    ]
    out = score(s)
    assert out.min_clearance_m == pytest.approx(2.0 - 0.6)
    assert out.min_clearance_with == "a"
    assert out.min_clearance_at == pytest.approx(1.0)


def test_freeze_ignores_the_settling_window() -> None:
    s = [Sample(t=float(i), robot=(0.0, 0.0), robot_vel=(0.0, 0.0)) for i in range(5)]
    out = score(s, settle_s=2.0)
    assert out.freeze_duration_s == pytest.approx(2.0), "only t>=2 counts"


def test_a_moving_robot_never_freezes() -> None:
    s = [Sample(t=float(i), robot=(float(i), 0.0), robot_vel=(1.0, 0.0)) for i in range(6)]
    assert score(s).freeze_duration_s == 0.0


def test_intrusion_time_counts_only_while_inside_personal_space() -> None:
    s = [
        Sample(t=0.0, robot=(0.0, 0.0), robot_vel=(1.0, 0.0), peds={"a": ((5.0, 0.0), (0.0, 0.0), 0.3)}),
        Sample(t=1.0, robot=(0.0, 0.0), robot_vel=(1.0, 0.0), peds={"a": ((0.8, 0.0), (0.0, 0.0), 0.3)}),
        Sample(t=2.0, robot=(0.0, 0.0), robot_vel=(1.0, 0.0), peds={"a": ((5.0, 0.0), (0.0, 0.0), 0.3)}),
    ]
    assert score(s).intrusion_time_s == pytest.approx(1.0)


def test_only_restricts_scoring_to_the_named_agent() -> None:
    s = [
        Sample(
            t=0.0,
            robot=(0.0, 0.0),
            peds={
                "bystander": ((1.0, 0.0), (0.0, 0.0), 0.3),
                "edge_0": ((4.0, 0.0), (0.0, 0.0), 0.3),
            },
        )
    ]
    assert score(s).min_clearance_with == "bystander"
    assert score(s, only=["edge_0"]).min_clearance_with == "edge_0"
    assert score(s, only=["edge_0"]).min_clearance_m == pytest.approx(4.0 - 0.6)


def test_pet_is_the_time_gap_through_the_encounter_point() -> None:
    E = (5.0, 0.0)
    s = [
        Sample(t=0.0, robot=(0.0, 0.0), peds={"a": ((5.0, 4.0), (0.0, 0.0), 0.3)}),
        Sample(t=2.0, robot=(5.0, 0.0), peds={"a": ((5.0, 2.0), (0.0, 0.0), 0.3)}),
        Sample(t=5.0, robot=(9.0, 0.0), peds={"a": ((5.0, 0.2), (0.0, 0.0), 0.3)}),
    ]
    assert post_encroachment_time(s, E) == pytest.approx(3.0)


def test_pet_is_none_when_the_pedestrian_never_goes_there() -> None:
    E = (5.0, 0.0)
    s = [
        Sample(t=0.0, robot=(0.0, 0.0), peds={"a": ((50.0, 50.0), (0.0, 0.0), 0.3)}),
        Sample(t=2.0, robot=(5.0, 0.0), peds={"a": ((51.0, 50.0), (0.0, 0.0), 0.3)}),
    ]
    assert post_encroachment_time(s, E) is None


def test_an_episode_with_no_pedestrians_is_not_a_safe_episode() -> None:
    out = score([Sample(t=float(i), robot=(float(i), 0.0), robot_vel=(1.0, 0.0)) for i in range(4)])
    assert out.min_clearance_m is None
    assert out.min_ttc_s is None
    assert out.observed is False


def test_no_samples_at_all_is_empty_not_zero() -> None:
    out = score([])
    assert out == Criticality()
    assert out.observed is False


def test_measured_ttc_can_differ_from_a_designed_one() -> None:
    s = [Sample(t=float(i), robot=(float(i), 0.0), robot_vel=(1.0, 0.0), peds={"edge_0": ((10.0, 0.0), (0.0, 0.0), 0.3)}) for i in range(4)]
    out = score(s, only=["edge_0"])
    assert out.min_ttc_s is not None, "the robot is still closing on a stationary agent"
    frozen = [Sample(t=float(i), robot=(0.0, 0.0), robot_vel=(0.0, 0.0), peds={"edge_0": ((10.0, 0.0), (0.0, 0.0), 0.3)}) for i in range(4)]
    assert score(frozen, only=["edge_0"]).min_ttc_s is None


def test_score_row_puts_designed_and_achieved_side_by_side(tmp_path) -> None:
    from task_generator.tasks.obstacles.edge_case.scoring import write_score

    case = CaseRecord(run_seed="abc", episode_id=7, world="arena_arena_002", seed=11, designed_ttc=4.0, designed_pet=0.0)
    result = score(
        [
            Sample(t=0.0, robot=(0.0, 0.0), robot_vel=(1.0, 0.0), peds={"edge_0": ((6.0, 0.0), (-1.0, 0.0), 0.3)}),
            Sample(t=1.0, robot=(1.0, 0.0), robot_vel=(1.0, 0.0), peds={"edge_0": ((5.0, 0.0), (-1.0, 0.0), 0.3)}),
        ],
        only=["edge_0"],
    )

    path = write_score(tmp_path, case, result)
    row = json.loads(path.read_text().splitlines()[-1])

    assert row["episode_id"] == 7
    assert row["designed_ttc"] == 4.0
    assert row["min_ttc_s"] is not None
    assert row["observed"] is True
    assert row["min_clearance_with"] == "edge_0"


def test_an_unobserved_episode_is_flagged_not_silently_zero(tmp_path) -> None:
    from task_generator.tasks.obstacles.edge_case.scoring import write_score

    path = write_score(tmp_path, CaseRecord(run_seed='abc', episode_id=1, world='w', seed=0), score([]))
    row = json.loads(path.read_text().splitlines()[-1])
    assert row["observed"] is False
    assert row["min_ttc_s"] is None
    assert row["min_clearance_m"] is None


def test_scores_are_a_separate_file_from_cases(tmp_path) -> None:
    from task_generator.tasks.obstacles.edge_case.scoring import write_score

    path = write_score(tmp_path, CaseRecord(run_seed='a', episode_id=1, world='w', seed=0), score([]))
    assert path.name == "scores.jsonl"
    write_score(tmp_path, CaseRecord(run_seed='a', episode_id=2, world='w', seed=0), score([]))
    assert len(path.read_text().strip().splitlines()) == 2, "append-only, one row per episode"


def test_formation_hold_reports_when_the_group_broke() -> None:
    from task_generator.tasks.obstacles.edge_case.criticality import Sample, formation_hold

    def sample(t, a, b, c):
        return Sample(t=t, robot=(0.0, 0.0), robot_vel=(0.0, 0.0), robot_radius=0.3, peds={"/env_0/d_1": (a, (0.0, 0.0), 0.3), "/env_0/d_2": (b, (0.0, 0.0), 0.3), "/env_0/d_3": (c, (0.0, 0.0), 0.3)})

    held = [sample(t, (0.0, 0.0), (1.0, 0.0), (0.5, 0.8)) for t in (0.0, 1.0, 2.0)]
    broke = [*held, sample(3.0, (0.0, 0.0), (1.0, 0.0), (4.0, 4.0)), sample(4.0, (0.0, 0.0), (1.0, 0.0), (5.0, 5.0))]
    out = formation_hold(broke, ["d_1", "d_2", "d_3"])
    assert out["seen"] and out["formed_at"] == 0.0 and out["held_s"] == 2.0 and out["broke_at"] == 3.0
    out = formation_hold(held, ["d_1", "d_2", "d_3"])
    assert out["seen"] and out["held_s"] == 2.0 and out["broke_at"] is None
    assert formation_hold(held, ["nobody"])["seen"] is False
    assert formation_hold([], ["d_1"])["seen"] is False
    walking = [sample(0.0, (9.0, 9.0), (1.0, 0.0), (0.5, 0.8)), sample(1.0, (0.2, 0.1), (1.0, 0.0), (0.5, 0.8)), sample(2.0, (0.1, 0.0), (1.0, 0.0), (0.5, 0.8)), sample(3.0, (6.0, 6.0), (1.0, 0.0), (0.5, 0.8))]
    out = formation_hold(walking, ["d_1", "d_2", "d_3"], radius=1.5, centre=(0.5, 0.3))
    assert out["seen"] and out["formed_at"] == 1.0 and out["held_s"] == 1.0 and out["broke_at"] == 3.0
    never = formation_hold(walking[:1], ["d_1", "d_2", "d_3"], radius=1.5, centre=(0.5, 0.3))
    assert never["seen"] and never["formed_at"] is None and never["held_s"] is None
    assert never["arrived"] == 2 and out["arrived"] == 3


def test_an_unrefreshed_pose_keeps_the_last_velocity_briefly() -> None:
    from task_generator.tasks.obstacles.edge_case.scoring import STALE_POSE_S, differenced_velocity

    vel, moved = differenced_velocity((0.0, 0.0), (0.0, 0.0), (0.1, 0.0), 0.1, 0.0)
    assert vel == pytest.approx((1.0, 0.0)) and moved == 0.1
    vel, moved = differenced_velocity((0.1, 0.0), vel, (0.1, 0.0), 0.2, moved)
    assert vel == pytest.approx((1.0, 0.0)) and moved == 0.1
    vel, moved = differenced_velocity((0.1, 0.0), vel, (0.1, 0.0), 0.1 + STALE_POSE_S, moved)
    assert vel == (0.0, 0.0), "unchanged for the stale window: standing"


def test_a_changed_pose_is_differenced_over_the_time_since_it_last_changed() -> None:
    from task_generator.tasks.obstacles.edge_case.scoring import differenced_velocity

    vel, moved = differenced_velocity((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), 0.3, 0.0)
    vel, moved = differenced_velocity((0.0, 0.0), vel, (0.2, 0.0), 0.4, moved)
    assert vel == pytest.approx((0.5, 0.0)) and moved == 0.4
