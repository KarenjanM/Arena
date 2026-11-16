from __future__ import annotations

import attrs
import numpy as np
import pytest
from arena_runtime.humans.animation_manager import AnimationManager
from arena_runtime.humans.clips import NoClips
from arena_runtime.humans.gestures import Channel, GestureLayer, GestureRequest
from arena_runtime.humans.gestures.point import PointGesture
from arena_runtime.humans.gestures.receive import ReceiveGesture
from arena_runtime.humans.pointing import BakedPointAt
from arena_runtime.humans.pointing import skeleton as S
from arena_runtime.humans.pointing.contract import SPINE_SEGMENTS, arm_dofs, wrist_dofs
from arena_runtime.humans.support import TYPICAL_BODY

pytestmark = pytest.mark.clips("point_at_template")
TARGET = np.array([2.0, 0.0, 1.25])


class Log:
    def info(self, message: str) -> None:
        pass

    def warning(self, message: str) -> None:
        pass


def test_receive_solves_each_measured_body_and_keeps_the_point_generator_baked():
    longer = attrs.evolve(TYPICAL_BODY, proportions=attrs.evolve(TYPICAL_BODY.proportions, forearm=0.35))
    short, long = ReceiveGesture(body=TYPICAL_BODY), ReceiveGesture(body=longer)
    assert short._gen.body is short._body and long._gen.body is long._body
    assert short._body.forearm == TYPICAL_BODY.proportions.forearm
    assert long._body.forearm == longer.proportions.forearm
    a = short.start(TARGET, {"hand": "r", "moving": True})
    b = long.start(TARGET, {"hand": "r", "moving": True})
    for gesture, clip in ((short, a), (long, b)):
        pos, _ = S.fk(clip.frames[-1]["angles"], gesture._body)
        assert np.linalg.norm(pos["r_wrist"] - pos["r_elbow"]) == pytest.approx(gesture._body.forearm)
        fore = S.unit(pos["r_wrist"] - pos["r_elbow"])
        want = S.unit(TARGET - pos["r_wrist"])
        assert float(fore @ want) > np.cos(np.radians(10))
    assert a.frames[-1]["angles"]["r_p_shoulder"] != pytest.approx(b.frames[-1]["angles"]["r_p_shoulder"])
    assert isinstance(PointGesture()._gen, BakedPointAt)


@pytest.mark.parametrize("side", ["l", "r", "both"])
def test_measured_receive_retargets_from_the_previous_hold_with_an_upright_torso(side: str) -> None:
    gesture = ReceiveGesture(body=TYPICAL_BODY)
    options = gesture.bind(f"receive_{side}", TARGET, {"moving": True})
    first = gesture.start(TARGET, options)
    second = gesture.retarget(first.hold, TARGET + [0.0, 0.15, 0.05], options)
    for side in first.sides:
        for name in (*arm_dofs(side), *wrist_dofs(side)):
            assert second.frames[0]["angles"][name] == pytest.approx(first.frames[-1]["angles"][name], abs=1e-6)
    for frame in second.frames:
        assert all(frame["angles"][name] == 0 for seg in SPINE_SEGMENTS for name in (seg, f"r_{seg}", f"y_{seg}"))


def test_live_slots_keep_their_body_and_profile_when_another_agent_starts():
    manager = AnimationManager(logger=Log(), fps=20.0, clips=NoClips())
    layer = GestureLayer(manager, Log())
    manager.gesture_hook = layer
    other = attrs.evolve(TYPICAL_BODY, proportions=attrs.evolve(TYPICAL_BODY.proportions, forearm=0.32))
    channel = Channel("receive_l", tuple(TARGET), orientation="thumb_up")
    for agent, body in ((1, TYPICAL_BODY), (2, other)):
        manager.compute(agent, 0, 0.0, 0.05, gesture=GestureRequest((channel,), (0.0, 0.0, 0.0), False, body))
    a = layer._agents[1].slots["arm"]
    b = layer._agents[2].slots["arm"]
    assert a.gesture is not b.gesture
    assert a.gesture._body.forearm == TYPICAL_BODY.proportions.forearm
    assert b.gesture._body.forearm == other.proportions.forearm
    assert a.gesture._orientation == b.gesture._orientation == "thumb_up"
    first = a.clip
    layer._retarget(1, a, TARGET + [0.0, 0.2, 0.0])
    assert a.clip is not first
    assert a.gesture._body.forearm == TYPICAL_BODY.proportions.forearm
