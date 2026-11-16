from __future__ import annotations

import math

import attrs
import numpy as np
import pytest
from arena_runtime.humans.gestures import GestureRequest, world_to_local
from arena_runtime.humans.gestures.profiles.handover.handover_profile import handover_profile
from arena_runtime.humans.support import TYPICAL_BODY


def test_profile_preserves_requested_hand_and_transforms_with_the_pedestrian():
    pose = (4.0, -2.0, math.pi / 2)
    left = handover_profile("bottle", TYPICAL_BODY, "l")
    right = handover_profile("bottle", TYPICAL_BODY, "r")
    assert left.side == "l" and right.side == "r"
    assert left.at[1] == -right.at[1] and left.at[1] > 0
    assert left.channel(pose).slot == "receive_l"
    assert right.channel(pose).slot == "receive_r"
    assert world_to_local(left.channel(pose).at, pose) == pytest.approx(left.at)
    assert left.channel(pose).grip == 0.0
    assert left.channel(pose, grip=0.4).grip == 0.4


def test_profile_changes_with_object_and_body_geometry():
    taller = attrs.evolve(TYPICAL_BODY, legs=attrs.evolve(TYPICAL_BODY.legs, thigh=TYPICAL_BODY.legs.thigh + 0.1))
    longer_arm = attrs.evolve(TYPICAL_BODY, proportions=attrs.evolve(TYPICAL_BODY.proportions, forearm=TYPICAL_BODY.proportions.forearm + 0.05))
    bottle = handover_profile("bottle", TYPICAL_BODY, "r")
    assert handover_profile("bottle", taller, "r").at[2] > bottle.at[2]
    assert handover_profile("bottle", longer_arm, "r").at[0] > bottle.at[0]
    box = handover_profile("box", TYPICAL_BODY, "l")
    assert box.side == "both" and box.at[1] == 0
    assert box.orientation == "palm_up" and bottle.orientation == "thumb_up"
    assert box.channel().slot == "receive_both"
    assert "hands" not in box.channel().opts
    request = GestureRequest((box.channel(),), (0.0, 0.0, 0.0), False, body=TYPICAL_BODY)
    assert np.isfinite(request.channels[0].at).all()


@pytest.mark.parametrize("kind, hand", [("missing", "l"), ("bottle", "auto")])
def test_profile_rejects_unknown_object_or_handedness(kind: str, hand: str) -> None:
    with pytest.raises(ValueError):
        handover_profile(kind, TYPICAL_BODY, hand)
