import math

import numpy as np
import pytest
import yaml

from task_generator.simulators.human.support import BodyMeasurements, Legs, PelvisClearance, Support, load_body, planted_pelvis, supported_pose

LEGS = Legs(hip=0.015, thigh=0.36, shin=0.375, ankle=0.08)
BODY = BodyMeasurements(clearance=PelvisClearance(sit=0.12, lie=0.11), legs=LEGS)
SEATED = {"l_r_hip": math.pi / 2, "r_r_hip": math.pi / 2, "l_knee": -math.pi / 2, "r_knee": -math.pi / 2}


def rotate(xyzw: np.ndarray, v: tuple[float, float, float]) -> np.ndarray:
    x, y, z, w = xyzw
    matrix = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return matrix @ np.array(v)


def test_planted_pelvis_stands_on_the_whole_leg_and_sits_on_the_shin():
    assert planted_pelvis({}, LEGS) == pytest.approx(0.015 + 0.36 + 0.375 + 0.08)
    assert planted_pelvis(SEATED, LEGS) == pytest.approx(0.015 + 0.375 + 0.08)


def test_sitter_is_lifted_onto_a_seat_above_its_planted_pelvis_only():
    position, orientation = supported_pose(Support("sit", 0.45), BODY, SEATED, 1.0, 2.0, math.pi / 2)
    assert position == pytest.approx([1.0, 2.0, 0.45 + 0.12 - 0.47])
    assert rotate(orientation, (1.0, 0.0, 0.0)) == pytest.approx([0.0, 1.0, 0.0])
    assert supported_pose(Support("sit", 0.3), BODY, SEATED, 1.0, 2.0, 0.0)[0][2] == 0.0
    assert supported_pose(Support("sit", 0.45), BODY, {}, 1.0, 2.0, 0.0)[0][2] == 0.0


@pytest.mark.parametrize("yaw", [0.0, 2.0])
def test_lying_ped_rests_its_pelvis_over_the_seat_on_its_back(yaw):
    position, orientation = supported_pose(Support("lie", 0.6), BODY, {}, 1.0, 2.0, yaw)
    pelvis = position + rotate(orientation, (0.0, 0.0, planted_pelvis({}, LEGS)))
    assert pelvis == pytest.approx([1.0, 2.0, 0.6 + 0.11])
    assert rotate(orientation, (1.0, 0.0, 0.0)) == pytest.approx([0.0, 0.0, 1.0])
    assert rotate(orientation, (0.0, 0.0, 1.0)) == pytest.approx([-math.cos(yaw), -math.sin(yaw), 0.0])


def test_body_is_read_from_the_annotation_beside_the_actor(tmp_path):
    sdf = tmp_path / "nurse" / "model.sdf"
    sdf.parent.mkdir()
    assert load_body(str(sdf)) is None
    (sdf.parent / "annotation.yaml").write_text(yaml.safe_dump({"name": "nurse", "pelvis_clearance": {"sit": 0.12, "lie": 0.11}}))
    assert load_body(str(sdf)) is None
    (sdf.parent / "annotation.yaml").write_text(yaml.safe_dump({"pelvis_clearance": {"sit": 0.12, "lie": 0.11}, "legs": {"hip": 0.015, "thigh": 0.36, "shin": 0.375, "ankle": 0.08}}))
    assert load_body(str(sdf)) == BODY
    assert load_body("") is None
