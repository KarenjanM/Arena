from __future__ import annotations

import math

import numpy as np
import pytest

from arena_simulation_setup.utils.affordance import AffordanceClaims, AffordanceKind, AffordanceSlot, Slot, SlotOccupied, max_occupancy, placed_slots, slot_box, structure_affordances, unstructure_affordances

SOFA_B = {
    "sit": [
        {"bbox": [[-0.46, 0.48], [0.03, 0.72]], "yaw": "-y"},
        {"bbox": [[-0.98, -0.48], [-0.50, 0.03]], "yaw": "+x"},
        {"bbox": [[-1.30, -0.56], [-1.13, -0.26]], "yaw": ["+x", "-y"]},
        {"bbox": [[0.48, 1.29], [-0.51, 0.08]], "yaw": "-x"},
        {"bbox": [[0.59, 1.36], [-1.06, -0.38]], "yaw": ["-x", "+x", "-y"]},
    ]
}


def sofa() -> AffordanceClaims:
    return AffordanceClaims.parse(SOFA_B)


def slot(claims: AffordanceClaims, index: int) -> AffordanceSlot:
    return claims.slots[index]


def test_parse_reads_kind_box_and_facing():
    parsed = slot(sofa(), 2)
    assert (parsed.kind, parsed.index) == (AffordanceKind.SIT, 2)
    assert parsed.bbox == ((-1.30, -0.56), (-1.13, -0.26), (0.0, 0.0))
    assert parsed.facing.yaws == pytest.approx([0.0, -math.pi / 2])


def test_placed_slots_skip_slots_the_build_has_not_placed():
    assert [placed.index for placed in placed_slots({"sit": [{}, {"bbox": [[0.0, 0.4], [0.0, 0.4], [0.0, 0.45]], "yaw": 0}]})] == [1]


def test_unknown_kind_names_the_known_ones():
    with pytest.raises(ValueError, match="unknown affordance 'fly', known: sit"):
        AffordanceClaims.parse({"fly": [{"bbox": [[0.0, 1.0], [0.0, 1.0]], "yaw": 0}]})


@pytest.mark.parametrize("bbox", [[[0.5, 0.0], [0.0, 0.5]], [[0.0, 0.5]], [[0.0, 0.5], [0.0, 0.5], [1.0, 0.5]], [[0.0, 0.5, 1.0], [0.0, 0.5]]])
def test_slot_box_rejects_malformed_boxes(bbox):
    with pytest.raises(ValueError, match="slot bbox"):
        slot_box(bbox)


def test_slot_roundtrips_source_and_built_forms():
    assert Slot.parse({"bbox": [[0.0, 0.5], [0.0, 0.5], [0.0, 0.43]], "yaw": 0.5}).unstructure() == {"bbox": [[0.0, 0.5], [0.0, 0.5], [0.0, 0.43]], "yaw": 0.5}
    assert Slot.parse({"bbox": [[0.0, 0.5], [0.0, 0.5]], "yaw": "-y"}).unstructure() == {"bbox": [[0.0, 0.5], [0.0, 0.5]], "yaw": "-y"}
    assert Slot.parse({}).unstructure() == {}


def test_slot_with_a_bbox_needs_a_yaw():
    with pytest.raises(ValueError, match="needs a yaw"):
        Slot.parse({"bbox": [[0.0, 0.5], [0.0, 0.5]]})


def test_slot_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown seats"):
        Slot.parse({"seats": 2})


def test_unplaced_slot_refuses_to_be_placed():
    with pytest.raises(ValueError, match="sit slot 2 is not placed yet"):
        Slot.parse({"yaw": "-y"}).placed("sit", 2)


def test_affordances_roundtrip_and_allow_one_slot_left_to_the_build():
    assert unstructure_affordances(structure_affordances(SOFA_B)) == {"sit": [{"bbox": slot["bbox"], "yaw": slot["yaw"]} for slot in SOFA_B["sit"]]}
    with pytest.raises(ValueError, match="at most one sit slot"):
        structure_affordances({"sit": [{}, {"yaw": "-y"}]})


def test_max_occupancy_counts_slots_that_can_be_taken_together():
    assert max_occupancy(sofa().slots) == 3
    assert max_occupancy(sofa().slots[:1]) == 1
    assert max_occupancy([]) == 0


def test_claim_refuses_the_same_slot_twice():
    claims = sofa()
    claims.claim(slot(claims, 0))
    with pytest.raises(SlotOccupied, match="sit slot 0 overlaps the occupied sit slot 0"):
        claims.claim(slot(claims, 0))


def test_claim_refuses_overlapping_slots():
    claims = sofa()
    claims.claim(slot(claims, 1))
    with pytest.raises(SlotOccupied, match="sit slot 2 overlaps the occupied sit slot 1"):
        claims.claim(slot(claims, 2))


def test_claim_allows_disjoint_slots_of_the_whole_sofa():
    claims = sofa()
    for index in (0, 1, 3):
        claims.claim(slot(claims, index))
    assert claims.blocking(slot(claims, 4)) == slot(claims, 3)


def test_occupied_lists_the_claimed_slots():
    claims = sofa()
    claims.claim(slot(claims, 3))
    claims.claim(slot(claims, 0))
    assert claims.occupied() == [slot(claims, 3), slot(claims, 0)]


def test_release_frees_the_slot():
    claims = sofa()
    claims.claim(slot(claims, 1))
    claims.release(slot(claims, 1))
    claims.claim(slot(claims, 2))
    with pytest.raises(KeyError, match="sit slot 1 is not claimed"):
        claims.release(slot(claims, 1))


def test_claims_compare_z_once_both_slots_are_built():
    claims = AffordanceClaims.parse(
        {
            "sit": [
                {"bbox": [[0.0, 0.5], [0.0, 0.5], [0.45, 1.45]], "yaw": "-y"},
                {"bbox": [[0.0, 0.5], [0.0, 0.5], [1.6, 2.6]], "yaw": "-y"},
                {"bbox": [[0.0, 0.5], [0.0, 0.5]], "yaw": "-y"},
            ]
        }
    )
    claims.claim(slot(claims, 0))
    claims.claim(slot(claims, 1))
    with pytest.raises(SlotOccupied):
        claims.claim(slot(claims, 2))


def test_available_leaves_out_occupied_and_overlapping_slots():
    claims = sofa()
    claims.claim(slot(claims, 1))
    assert [free.index for free in claims.available()] == [0, 3, 4]
    assert claims.available("sit") == claims.available(AffordanceKind.SIT)


def test_available_names_unknown_kinds():
    with pytest.raises(ValueError, match="'lie'"):
        sofa().available("lie")


def test_closest_takes_the_nearest_available_slot():
    claims = sofa()
    assert claims.closest((1.0, -0.9)) == slot(claims, 4)
    claims.claim(slot(claims, 3))
    assert claims.closest((1.0, -0.9)) == slot(claims, 0)


def test_closest_inside_two_boxes_takes_the_nearer_center():
    claims = sofa()
    assert claims.closest((-0.7, -0.4), "sit") == slot(claims, 1)
    assert claims.closest((-0.9, -0.45), "sit") == slot(claims, 2)


def test_closest_is_none_when_every_slot_is_blocked():
    claims = sofa()
    for index in (0, 1, 3):
        claims.claim(slot(claims, index))
    assert claims.closest((0.0, 0.0)) is None


def test_pose_sits_at_the_box_center_facing_its_yaw():
    pose = Slot.parse({"bbox": [[-0.46, 0.48], [0.03, 0.72], [0.0, 0.47]], "yaw": "-y"}).placed("sit", 0).pose(np.random.default_rng(0))
    assert (pose.position.x, pose.position.y) == pytest.approx((0.01, 0.375))
    assert pose.orientation.to_yaw() == pytest.approx(-math.pi / 2)


def test_pose_draws_each_allowed_yaw_and_repeats_per_seed():
    placed = slot(sofa(), 4)
    directions = set()
    for seed in range(32):
        yaw = placed.pose(np.random.default_rng(seed)).orientation.to_yaw()
        directions.add((round(math.cos(yaw), 4) + 0.0, round(math.sin(yaw), 4) + 0.0))
    assert directions == {(1.0, 0.0), (-1.0, 0.0), (0.0, -1.0)}
    assert placed.pose(np.random.default_rng(7)) == placed.pose(np.random.default_rng(7))
