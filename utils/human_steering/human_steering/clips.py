"""Wire-clip loading, sampling and blending.

ROS-free, Qt-free. Wire-clip schema (see README.md "Wire-clip schema"):

    {
      "version": 1,
      "rate_hz": 30,
      "clips": {
        "<name>": {
          "duration": <seconds>,
          "cyclic": <bool>,
          "tracks": {"<bare base joint>": [v, ...]},
          "root_z": [z, ...]        # optional
        }
      }
    }

Sources: `dirname(model_uri)/clips/wire.json` (the model's authored clip
bundle) and `$ARENA_DATA_DIR/peds/poses/*.json` (user-authored poses,
nested under the existing `peds` data bucket).
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import pathlib
import re
import tempfile
from collections.abc import Mapping, Sequence

WIRE_RATE_HZ = 30.0
BLEND_S = 0.3
SCHEMA_VERSION = 1


@dataclasses.dataclass(frozen=True)
class Clip:
    """One sampled motion clip: bare joint name -> keyframe track, at rate_hz."""

    name: str
    duration: float
    cyclic: bool
    rate_hz: float
    tracks: dict[str, tuple[float, ...]]
    root_z: tuple[float, ...] | None = None


def slug(name: str) -> str:
    """Collapse non-alphanumeric runs to a single underscore, for filenames and topic tokens."""
    collapsed = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")
    return collapsed or "_"


def parse_wire_json(doc: Mapping[str, object]) -> dict[str, Clip]:
    """Parse a wire-clip document (already json.load'ed) into name -> Clip."""
    rate_hz = float(doc.get("rate_hz", WIRE_RATE_HZ))  # type: ignore[arg-type]
    raw_clips = doc.get("clips", {})
    clips: dict[str, Clip] = {}
    if not isinstance(raw_clips, Mapping):
        return clips
    for name, raw in raw_clips.items():
        if not isinstance(raw, Mapping):
            continue
        raw_tracks = raw.get("tracks", {})
        tracks = {str(joint): tuple(float(v) for v in values) for joint, values in raw_tracks.items()} if isinstance(raw_tracks, Mapping) else {}
        raw_root_z = raw.get("root_z")
        root_z = tuple(float(v) for v in raw_root_z) if isinstance(raw_root_z, Sequence) else None
        clips[str(name)] = Clip(
            name=str(name),
            duration=float(raw.get("duration", 0.0)),
            cyclic=bool(raw.get("cyclic", False)),
            rate_hz=rate_hz,
            tracks=tracks,
            root_z=root_z,
        )
    return clips


def load_wire_json_file(path: pathlib.Path) -> dict[str, Clip]:
    """Load and parse one wire-clip file. Missing file -> empty library."""
    if not path.is_file():
        return {}
    with path.open() as handle:
        doc = json.load(handle)
    return parse_wire_json(doc)


def clips_path_for_model(model_uri: str) -> pathlib.Path:
    """The bundle's clip file for a resolved model_uri: dirname(model_uri)/clips/wire.json."""
    return pathlib.Path(os.path.dirname(model_uri)) / "clips" / "wire.json"


def poses_root() -> pathlib.Path:
    """$ARENA_DATA_DIR/peds/poses: user-authored poses."""
    env = os.environ.get("ARENA_DATA_DIR")
    if env:
        return pathlib.Path(env) / "peds" / "poses"
    return pathlib.Path(tempfile.gettempdir()) / "peds" / "poses"


def load_poses_dir(root: pathlib.Path | None = None) -> dict[str, Clip]:
    """Merge every *.json wire-schema file under poses_root() into one library.

    Later files (sorted by name) win on a clip-name collision.
    """
    root = root if root is not None else poses_root()
    if not root.is_dir():
        return {}
    merged: dict[str, Clip] = {}
    for path in sorted(root.glob("*.json")):
        merged.update(load_wire_json_file(path))
    return merged


def load_library(model_uri: str) -> dict[str, Clip]:
    """Full clip inventory for a ped: bundle clips plus the shared poses library."""
    library = load_wire_json_file(clips_path_for_model(model_uri))
    library.update(load_poses_dir())
    return library


def _cosine_interp(a: float, b: float, frac: float) -> float:
    """Cosine-eased interpolation between two keyframes, frac in [0, 1]."""
    mu = (1.0 - math.cos(frac * math.pi)) / 2.0
    return a * (1.0 - mu) + b * mu


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def sample_track(
    values: Sequence[float],
    rate_hz: float,
    duration: float,
    cyclic: bool,
    t: float,
) -> float:
    """Cosine-interpolated sample of one keyframe track at time t, with loop wrap."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    if cyclic:
        t = t % duration if duration > 0.0 else 0.0
    else:
        t = _clamp(t, 0.0, duration)
    pos = t * rate_hz
    i0 = int(math.floor(pos))
    frac = pos - i0
    n = len(values)
    if cyclic:
        v0 = values[i0 % n]
        v1 = values[(i0 + 1) % n]
    else:
        i0 = min(i0, n - 1)
        i1 = min(i0 + 1, n - 1)
        v0, v1 = values[i0], values[i1]
    return _cosine_interp(v0, v1, frac)


def sample(clip: Clip, t: float) -> dict[str, float]:
    """Sample every track of a clip at time t."""
    return {joint: sample_track(values, clip.rate_hz, clip.duration, clip.cyclic, t) for joint, values in clip.tracks.items()}


def sample_root_z(clip: Clip, t: float) -> float | None:
    """Sample the optional root_z track at time t, None if the clip carries none."""
    if clip.root_z is None:
        return None
    return sample_track(clip.root_z, clip.rate_hz, clip.duration, clip.cyclic, t)


def blend(
    prev: Mapping[str, float],
    next_: Mapping[str, float],
    elapsed_s: float,
    blend_s: float = BLEND_S,
) -> dict[str, float]:
    """Cosine-blend from prev to next_ over blend_s seconds, >= blend_s returns next_ verbatim."""
    if blend_s <= 0.0 or elapsed_s >= blend_s:
        return dict(next_)
    frac = elapsed_s / blend_s
    keys = set(prev) | set(next_)
    return {key: _cosine_interp(prev.get(key, next_.get(key, 0.0)), next_.get(key, prev.get(key, 0.0)), frac) for key in keys}
