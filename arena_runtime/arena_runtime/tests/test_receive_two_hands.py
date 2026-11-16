from __future__ import annotations

import attrs
import numpy as np
import pytest

from arena_runtime.humans.animation_manager import AnimationManager
from arena_runtime.humans.clips import NoClips
from arena_runtime.humans.gestures import Channel, GestureLayer, GestureRequest, resample
from arena_runtime.humans.gestures.receive import ReceiveGesture
from arena_runtime.humans.hands import GRIP, receive_hand
from arena_runtime.humans.pointing import skeleton as S
from arena_runtime.humans.pointing.contract import SPINE_SEGMENTS, arm_dofs, wrist_dofs

pytestmark = pytest.mark.clips("point_at_template")
TARGET = np.array([2.0, 0.0, 1.25])
SPINE = {f"{axis}_{s}" for s in SPINE_SEGMENTS for axis in ("y", "r")} | set(SPINE_SEGMENTS)


class Log:
    def __init__(self):
        self.warnings = []

    def info(self, message):
        pass

    def warning(self, message):
        self.warnings.append(message)


@pytest.fixture(scope="module")
def receive():
    return ReceiveGesture()


@pytest.mark.parametrize("orientation", ["palm_up", "thumb_up"])
@pytest.mark.parametrize("moving", [False, True])
def test_one_clip_controls_both_arms_against_one_torso(orientation, moving):
    gesture = ReceiveGesture(orientation)
    options = gesture.bind("receive_both", TARGET, {"moving": moving})
    clip = gesture.start(TARGET, options)
    assert clip.side == "both" and clip.sides == ("l", "r")
    assert clip.hold_end == len(clip.frames) - 1
    expected = {joint for side in clip.sides for joint in (*arm_dofs(side), *wrist_dofs(side))}
    assert gesture.joints(clip.side, moving) == expected | (set() if moving else SPINE)
    assert all(frame["angles"][joint] == 0.0 for frame in clip.frames for joint in SPINE)
    pos, _ = S.fk(clip.frames[-1]["angles"], gesture._body)
    for side in clip.sides:
        fore = S.unit(pos[f"{side}_wrist"] - pos[f"{side}_elbow"])
        want = S.unit(TARGET - pos[f"{side}_wrist"])
        assert float(fore @ want) > np.cos(np.radians(10.0))
        assert clip.frames[-1]["angles"][f"{side}_wrist"] == 0.0
        assert clip.release[-1]["angles"][f"{side}_r_wrist"] == pytest.approx(0.0)
    assert clip.report["ok"]
    assert clip.report["clearance_m"] >= 0.0
    assert set(clip.report["arms"]) == {"l", "r"}
    assert set(gesture.breathing("both")) == {f"{s}_{joint}" for s in ("l", "r") for joint in ("p_shoulder", "elbow")}


@pytest.mark.parametrize("side", ["l", "r", "both"])
def test_explicit_receive_slots_select_the_requested_side(receive, side):
    options = receive.bind(f"receive_{side}", TARGET, {})
    assert options["side"] == side
    clip = receive.start(TARGET, options)
    assert clip.side == side
    assert clip.sides == (("l", "r") if side == "both" else (side,))


@pytest.mark.parametrize("dominant", ["l", "r"])
def test_general_receive_resolves_one_hand_from_handedness(receive, dominant):
    options = receive.bind("receive", TARGET, {"dominant": dominant})
    assert options["side"] == dominant
    assert receive.start(TARGET, options).side == dominant


@pytest.mark.parametrize("moving", [False, True])
def test_retarget_preserves_both_starting_arm_poses(receive, moving):
    options = receive.bind("receive_both", TARGET, {"moving": moving})
    first = receive.start(TARGET, options)
    second = receive.retarget(first.hold, np.array([2.0, 0.25, 1.3]), options)
    assert second.side == "both"
    for side in second.sides:
        for joint in (*arm_dofs(side), *wrist_dofs(side)):
            assert second.frames[0]["angles"][joint] == pytest.approx(first.frames[-1]["angles"][joint], abs=1e-6)
            assert second.release[-1]["angles"][joint] == pytest.approx(0.0, abs=1e-9)
    assert second.report["ok"]


def test_different_arm_timelines_share_ramp_hold_and_release_boundaries(receive):
    left = receive._start_one(TARGET, {"hand": "l", "moving": True})
    right = receive._start_one(TARGET, {"hand": "r", "moving": True})
    original_held = dict(left.frames[-1]["angles"])
    left = attrs.evolve(left, frames=resample(left.frames[: left.hold_start], 12) + resample(left.frames[left.hold_start :], 7), hold_start=12, hold_end=18, release=resample(left.release, 31))
    clip = receive._combine(left, right, TARGET)
    assert clip.hold_start == max(left.hold_start, right.hold_start)
    assert clip.hold_end == clip.hold_start + max(7, right.hold_end - right.hold_start + 1) - 1
    assert len(clip.release) == 31
    for joint in (*arm_dofs("l"), *wrist_dofs("l")):
        assert clip.frames[-1]["angles"][joint] == pytest.approx(original_held[joint])
    assert [f["t"] for f in clip.frames] == pytest.approx(np.arange(len(clip.frames)) / clip.fps)


@pytest.mark.parametrize("grip", [0.4, 1.0])
def test_one_receive_channel_owns_one_overlay_and_grips_both_hands(receive, grip):
    log = Log()
    manager = AnimationManager(logger=log, fps=20.0, clips=NoClips())
    layer = GestureLayer(manager, log)
    layer._gestures["receive"] = receive
    manager.gesture_hook = layer
    channel = Channel("receive_both", tuple(TARGET))

    def tick(ch, moving=False):
        return manager.compute(1, 0, 0.0, 0.05, gesture=GestureRequest((ch,), (0.0, 0.0, 0.0), moving))

    for _ in range(30):
        tick(channel)
    slot = layer._agents[1].slots["arm"]
    clip = slot.clip
    assert list(layer._agents[1].slots) == ["arm"]
    assert [name for name, *_ in manager.overlays(1)] == ["arm"]
    assert slot.joints == receive.joints("both", False)
    for _ in range(7):
        tick(attrs.evolve(channel, grip=grip))
    assert layer._agents[1].slots["arm"] is slot and slot.clip is clip
    side, weight = layer.grip_weights(1)["arm"]
    assert side == "both" and weight == pytest.approx(grip)
    fingers = receive_hand(side, weight)
    for s in ("l", "r"):
        assert fingers[f"{s}_curl_index"] == pytest.approx(grip * GRIP[1])
    tick(attrs.evolve(channel, grip=grip), moving=True)
    assert layer._agents[1].slots["arm"] is slot
    assert slot.clip.side == "both"
    assert not slot.joints & SPINE
    assert layer.grip_weights(1)["arm"][1] == pytest.approx(grip)
    assert not log.warnings


def test_duplicate_receive_arm_channels_are_still_rejected(receive):
    log = Log()
    layer = GestureLayer(None, log)
    first = Channel("receive_both", tuple(TARGET))
    duplicate = Channel("receive_l", tuple(TARGET))
    wanted = layer._wanted(1, GestureRequest((first, duplicate), (0.0, 0.0, 0.0), False))
    assert wanted == {"arm": first}
    assert len(log.warnings) == 1


def test_a_head_channel_can_run_beside_the_combined_arm_clip(receive):
    log = Log()
    manager = AnimationManager(logger=log, fps=20.0, clips=NoClips())
    layer = GestureLayer(manager, log)
    layer._gestures["receive"] = receive
    manager.gesture_hook = layer
    request = GestureRequest((Channel("receive_both", tuple(TARGET)), Channel("head", tuple(TARGET))), (0.0, 0.0, 0.0), False)
    for _ in range(30):
        manager.compute(1, 0, 0.0, 0.05, gesture=request)
    assert set(layer._agents[1].slots) == {"head", "arm"}
    assert layer._agents[1].slots["arm"].clip.side == "both"
    assert {name for name, *_ in manager.overlays(1)} == {"head", "arm"}
    assert not log.warnings


@pytest.mark.parametrize("side", ["left", "right", 2, True])
def test_invalid_receive_sides_are_rejected(receive, side):
    with pytest.raises(ValueError, match="side"):
        receive.start(TARGET, {"side": side})
