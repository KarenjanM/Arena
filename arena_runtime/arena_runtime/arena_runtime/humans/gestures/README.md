# gestures

World-frame attention channels from the `arena_peds` bus become upper-body overlay slots on the `AnimationManager`.

## Wire

Every `Pedestrian.gestures[]` entry is one channel: `slot` names the body part (`head`, `arm`, `arm_l`, `arm_r`, `halt`, `halt_l`, `halt_r`, `receive`, `receive_l`, `receive_r`, `receive_both`, `body`),
`at` is a world-frame point (aimed slots), `clip` an animation name (`body`), `hand` the dominant hand (`arm`, `halt`, `receive`),
`render_pose_override` makes `at` a display-only position for the ped (`body`, contact clips).
`BaseHumanSimulator.publish_arena_peds` builds one `GestureRequest(channels, pose, moving)` per ped with a
`Channel(slot, at, clip, hand, grip=...)` per entry (`Channel.opts` presents generator options as a dict;
grip is tracked separately by the layer)
and passes it to `AnimationManager.compute(..., gesture=...)`, which calls the layer through `gesture_hook`.

| slot | kind | wire | hand |
| --- | --- | --- | --- |
| `head` | `look` | `at` | |
| `arm` | `point` | `at`, `hand` (default `r`) | resolved once at slot start: `hand` within the 20 deg midline band, else the target-side arm |
| `arm_l` / `arm_r` | `point` | `at` | forced left / right |
| `halt` / `halt_l` / `halt_r` | `halt` | as `arm` | the `point` swing held at a fixed 15 deg elevation on the target's bearing, palm pushed out at the target (forearm pronation plus 70 deg wrist extension, opening with the forearm's rise), shares the `arm` overlay |
| `receive` / `receive_l` / `receive_r` / `receive_both` | `receive` | as `arm` | automatic single hand / forced left / forced right / both arms; straight wrist, palm or thumb side upward, shares the `arm` overlay |
| `body` | `clip` | `clip` | |

`clip` plays the manager's cached Animation asset of that name as an overlay over the joints its
`annotation.yaml` lists (none listed = whole body, see [Animation clips](../README.md#animation-clips)), looping when
annotated so, else parking on the last frame until the channel disappears. It needs no target and no generator:
the slot holds from the first tick and fades back into the gait on release.

The engine publishes exactly the active channels each tick. Nothing is implicit: no head channel = the head idles
while the arm points.

### Receive orientation

Select the hand orientation when constructing the generator:

```python
ReceiveGesture()                        # default: orientation="palm_up"
ReceiveGesture(orientation="palm_up")    # palm faces upward
ReceiveGesture(orientation="thumb_up")   # thumb side faces upward, palm sideways
```

The selected direction is aligned with the projection of upward perpendicular to the forearm,
subject to the existing wrist limits. Wrist extension stays zero in both modes. The same selection
applies to the reach, retargeting and release. Other orientation values raise `ValueError`.
The gesture layer's `_load_receive` factory uses these defaults. Python channels may override
`orientation`; the ROS `Gesture` message has no orientation field. The channel name selects the arms.

### Measured bodies and the first handover profile

`BaseHumanSimulator` supplies `_body_for(model_uri)` as `GestureRequest.body`. Receive uses the
existing native arm solver with those measured proportions, including the floor-to-pelvis offset.
Point and halt retain their baked `BODY_HEIGHT` solutions. A slot keeps its configured receive
generator through retargeting; a configured generator is created when a slot starts, not per tick.
Requests without body measurements retain the previous baked receive behavior.

`handover_profile` selects a deterministic preferred point and hand configuration:

```python
from arena_runtime.humans.gestures import GestureRequest
from arena_runtime.humans.gestures.profiles.handover.handover_profile import handover_profile

profile = handover_profile("bottle", body, handedness="l")
pose = (2.0, 3.0, 0.0)  # pedestrian x, y and yaw, standing on the world floor
request = GestureRequest((profile.channel(pose, grip=False),), pose, moving=False, body=body)
```

The point is derived from elbow height and forearm/hand length, with half the object's extent
added to describe its centre. A vertical grip axis chooses thumb-up; other axes choose palm-up.
One-hand entries use the requested handedness; two-hand entries use a point on the midline.
The profile resolves this once into `side="l"`, `"r"` or `"both"`, and its channel is correspondingly
`receive_l`, `receive_r` or `receive_both`. No hand-count field is carried by the profile or channel.
`objects.yaml` contains explicit example dimensions and hand counts. These are prototype settings,
not literature-calibrated thresholds; mass is recorded but does not yet affect the pose. The palm
offset uses the same 0.45-of-hand-length approximation as body support placement.

This profile is a Python entry point for the later handover trigger. It does not yet consume an
object-kind ROS message, sample variation or add a torso-bend parameter. The generated receive
still aims along the forearm rather than solving palm contact at the preferred point. Finger grip
continues to use the existing open/GRIP fade, with no object mesh or contact calculation.
Close preferred points can trigger the existing direction-only fallback, so the resulting hand
position, and even the forearm ray from that position, need not meet the preferred point.
Measured-body clip generation uses the native solver and is more expensive than baked playback;
it runs on start or retarget, rather than on every animation tick.

### Two-handed receive

`Channel("receive_both", at=...)` makes receive generate both arms. `receive_l` and `receive_r`
force one side; the general `receive` channel resolves one arm from the target and handedness.
`bind` freezes that selection as `side="l"`, `"r"` or `"both"`, also used by the generated clip.
All four channels map to the single `arm` overlay; publishing two arm channels does not bypass
the existing first-channel-wins rule. `ReceiveGesture` has no hand-count constructor parameter.

The native pointing generator produces each arm against an upright torso. Receive then synchronizes
the ramp, settled segment and release separately, and merges only each arm's own joints into one
`GestureClip`. The shared torso is zeroed; it is blended while stationary and left to locomotion
while moving. `side="both"` identifies the combined clip, and `clip.sides` returns `("l", "r")`.
Its blend mask includes both arms and wrists. Breathing and the shared grip fade affect both hands.

The hold stores both native arm states, allowing a target update to continue both arms from their
previous held poses. The combined report preserves both arm reports and measures held-pose aim
errors and capsule clearance with both arms present. A flagged pose is reported using the existing
layer warning mechanism; this is not a coupled collision-avoidance optimizer.

Both arms aim toward the same handover point. This provides a two-handed receiving motion, not
object-specific palm contact positions or a physical grasp. Wrist limits and the existing upright
torso approximation during locomotion still apply.

### Receive grip

`Gesture.grip` and `Channel.grip` default to `false`. For a receive channel, `false` selects
straight fingers and zero spread; `true` selects `hands.GRIP` on the arm chosen by the slot.
The fingers interpolate between these poses over `FADE_S` (0.25 s) of simulation time:

```python
Channel("receive_r", at=(2.0, 0.0, 1.2), grip=True)
```

`grip` is separate from `Channel.opts`. Changing only this flag keeps the same arm slot,
clip and playhead. Each change starts from the current interpolated pose, so reversing
mid-transition does not jump. Repeated identical requests do not restart the fade.
A fresh receive with `grip=true` also closes from the open pose over `FADE_S`.
The grip state and fade clock survive target updates and remain active during release
until the slot ends. Other gesture kinds ignore the flag. Humansim forwards it from
`GestureIntent` through both gesture message types to the runtime channel.

`GestureLayer.grips` exposes the requested booleans; `grip_weights` exposes the current
0..1 amounts used by `hands.receive_hand`. `GRIP` is a fixed finger-pose preset. The
interpolation does not adapt it to an object's shape or prevent finger/object intersections.

## Slots

Every overlay lives in a named slot on the manager (`set_ped_blend(..., slot="arm")`), each with its own
playhead, envelope and `on_end`. Slots are applied in insertion order. `head` drives the `head` overlay, the
point, halt and receive channels share the `arm` overlay, so at most one arm channel runs at a time.
That channel may control both physical arms for two-handed receive. The layer keeps one `_Agent`
per ped with a `_Slot` per (agent, overlay), and each slot runs its own phase machine, independent of the other:

- start: a channel appears (or reappears). `Gesture.bind(slot, local, opts)` freezes the
  generator options for the life of the slot: for `point` that is the hand, so a chained list keeps its arm
  across the midline and retargets never switch hands. The layer adds `moving` to the opts it binds: a walking
  ped does not blend its torso, so `point` solves its arm against an upright spine, and a `moving` flip on a
  held slot re-solves through a retarget.
- retarget: the channel's target moved by more than `RETARGET_EPS_M` while the slot holds.
- release: the channel disappeared, its `opts` changed, or a different arm channel
  wants the `arm` overlay (that one starts once the tail has played). Restarts (`opts` change, arm switch,
  a `ValueError` from `retarget`) skip `HOLD_MIN_S`.
- a generator fault drops only that slot, with one warning per (ped, kind), and the same channel is not
  resubmitted until its slot, target, or opts change.

## Phases

```
requested -> ramp -> hold -> (transition -> hold)* -> release -> gone
```

- `requested`: the clip is generated inline on the publishing tick. Nothing solves at runtime: `point` swings the
  recorded template onto a hold interpolated from the baked `pointing` table, a few milliseconds of matrix products.
- `ramp`, `hold`: the clip up to its park frame is an overlay that holds its park pose (or loops a breathing
  segment, see Timing) as long as the channel does.
- `transition`: target moved: `Gesture.retarget(hold, local, opts)` continues from the held pose without
  dropping the arm. A `ValueError` from `retarget` releases and restarts.
- `release`: eases from the pose the slot shows now into the current clip's lowering arc (`GestureClip.release`)
  with a `FADE_S` fade-out, then drops the transient clips. Never earlier than `HOLD_MIN_S` after the hold is
  reached unless it is a restart. A request for the same overlay waits until the tail has played.

## Reach

Reach is the engine's call (`arena_humansim.core.behavior.reach`, hysteresis per channel): a channel on the bus is
one the engine wants shown, the layer serves whatever arrives. The baked table covers the full azimuth circle,
so an off-envelope aim yields a flagged hold, never a fault.

## Timing

- Ramp of a fresh `point` = `clamp(swept / OMEGA_RAD_S, MOVE_MIN_S, MOVE_MAX_S)` (3.5 rad/s, 0.3 .. 0.9 s), the
  swept angle being from the hanging arm to the aim. The template's ramp segment is resampled to that length,
  hold and settle are kept as recorded, the release is the recorded lowering arc stretched by `RELEASE_STRETCH`
  (1.25). A retargeted `point` instead lowers from its new hold pose with an ease to rest over
  `move_time(angle from the new aim to down) * RELEASE_STRETCH`, so the arm never swings back through the first target.
- Retarget transitions use the same clamp on the angle between the old and new aim: an eased blend from the
  held pose into the new hold, then the template's settled plateau.
- Breathing: `Gesture.breathing(side)` names joints and amplitudes (`point`: shoulder pitch and elbow of the
  pointing arm, +-1.5 deg). The layer bakes a slow sinusoid (nominal `BREATH_HZ` 0.3, rounded to a whole
  number of frames) into every clip of the slot from the first hold onwards, phase seeded from the agent id
  and continuous across retargets, and appends one period after the park frame as a loop tail
  (`Animation.loop_from`). No wall clock anywhere.
- `look`: closed-form head yaw/pitch from the rest-pose eye point (`skeleton.fk` `eye`, torso assumed upright),
  clamped to the head limits. A look slot plays no clips: every tick a `HeadFollower` steps its yaw, pitch and
  overlay weight as critically damped followers (`FOLLOW_OMEGA_RAD_S` 7 rad/s, exact step over the real dt) toward
  the current aim and toward 1 while the channel lives or 0 on release, the shown head capped at `PEAK_RAD_S`
  (200 deg/s). Look slots retarget every tick (no `RETARGET_EPS_M`, no `HOLD_MIN_S`), and the slot ends once the
  released weight is below `SETTLED_WEIGHT`.

The world target is converted into the ped frame (x forward, y left, z up, origin on the floor under the ped)
once per clip from the bus pose. Generator reports that flag `ok=False`, `near_target_fallback`, or `relaxed`
are logged once per (ped, kind) and the clip still plays.

Blend joints come from `Gesture.joints(side, moving)`. `point` blends the pointing arm always and the spine
(yaw, pitch, roll of waist, spine, chest) only while the ped is idle, so walking gaits keep their torso sway. `halt` adds the
wrist pair of its arm, as does `receive`. `look` blends the head triple.
When `moving` flips on a live slot the layer recomputes the set and swaps it in place through
`AnimationManager.set_overlay_joints(..., fade_s=FADE_S)`: leaving joints ramp their weight to 0 and entering
joints ramp to 1, playhead and envelope untouched, no reinstall.

## QA

[qa.py](qa.py) replays scripted cases (single points, bursts, chains, hand switches, grid cells with rests
between, a walker tracking a target) through the real manager and layer and asserts what the sim would show:
no warnings, no wrist/elbow/head snap over `MAX_LINK_STEP_M`, clavicle below `MAX_COLLAR_RAD`, the parked
forearm within `MAX_HOLD_AIM_DEG` of the target, the arm hanging inside every `rest_windows`, and at least
`min_clips` arm clips. `python3 -m arena_runtime.humans.gestures.qa` renders filmstrips.

## Registry

`GESTURES` is a `ClassRegistry[str, Callable[[AnimationManager, BodyMeasurements | None, str], Gesture]]`.
Factories receive the manager, body and palm orientation; each factory passes the arguments its
gesture needs. Receive is created per slot; other kinds reuse their layer instance. `SLOT_KIND`
maps wire slots to kinds and `SLOT_OVERLAY` maps them to manager overlays. The `Gesture` protocol:

```python
class Gesture(Protocol):
    def joints(self, side: str, moving: bool) -> set[str]: ...
    def bind(self, slot: str, local: np.ndarray, opts: dict) -> dict: ...
    def start(self, local: np.ndarray, opts: dict) -> GestureClip: ...
    def retarget(self, hold: object, local: np.ndarray, opts: dict) -> GestureClip: ...
    def breathing(self, side: str) -> dict[str, float]: ...
```

`GestureClip` carries wire-format frames (`{"angles": {joint: rad}}`, only the blended joints are required),
`fps`, `hold_start`, `hold_end` (the park frame, its pose is what `hold` continues from), `side`, an opaque
`hold` handle, the generator report, and `release` (the lowering arc from the park pose, `None` keeps the
previous one). Generator exceptions drop the slot with one warning per ped and never reach the roster publish.

| kind | module | generator |
| --- | --- | --- |
| `point` | [point.py](point.py) | [`pointing.BakedPointAt`](../pointing/table.py) `point_at` / `retarget` on the baked hold table, `resolve_hand` wraps `choose_hand` |
| `look` | [look.py](look.py) | closed form on [`pointing.skeleton`](../pointing/skeleton.py) |

### Adding a kind

1. `gestures/<kind>.py` with a class satisfying `Gesture`. Reuse `pointing/` for aimed chains, or return frames
   from any clip. Return `dict(opts)` from `bind()` and `{}` from `breathing()` unless you want them.
2. Register in `gestures/__init__.py` and map a wire slot to it in `SLOT_KIND` / `SLOT_OVERLAY`:

```python
@GESTURES.register("wave")
def _load_wave() -> Callable[[AnimationManager, BodyMeasurements | None, str], Gesture]:
    from .wave import WaveGesture
    return lambda _manager, _body, _orientation: WaveGesture()
```

3. The engine side then needs a channel that publishes that slot. Slots and opts are opaque strings on the wire.

Non-aimed gestures ignore `local` in `start` and can raise `ValueError` from `retarget` to always restart.
