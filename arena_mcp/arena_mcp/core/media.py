from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

from mcp.server.mcpserver import Image

from .errors import ToolError

if TYPE_CHECKING:
    from nav_msgs.msg import OccupancyGrid
    from numpy.typing import ArrayLike
    from sensor_msgs.msg import CompressedImage
    from sensor_msgs.msg import Image as ImageMsg

MAX_SIDE = 1568


def png(data: bytes | None = None, *, path: str | Path | None = None) -> Image:
    """MCP image content from PNG bytes or a file."""
    if path is not None:
        return Image(path=path)
    if data is None:
        raise ToolError("no image data")
    return Image(data=data, format="png")


def array_png(pixels: ArrayLike, *, max_side: int = MAX_SIDE) -> Image:
    """MCP image content from an HxW or HxWx3/4 uint8 array, downscaled to max_side."""
    import numpy as np
    from PIL import Image as PilImage

    array = np.asarray(pixels)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    image = PilImage.fromarray(array)
    image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return png(buffer.getvalue())


def image_msg(msg: ImageMsg, *, max_side: int = MAX_SIDE) -> Image:
    """MCP image content from a sensor_msgs Image (rgb, bgr, mono and 16/32 bit depth)."""
    import numpy as np

    encoding = msg.encoding.lower()
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if encoding in ("rgb8", "bgr8", "rgba8", "bgra8"):
        channels = 4 if "a" in encoding[3:] else 3
        pixels = raw.reshape(msg.height, msg.step)[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
        if encoding.startswith("bgr"):
            pixels = pixels[..., [2, 1, 0, *([3] if channels == 4 else [])]]
        return array_png(pixels, max_side=max_side)
    if encoding in ("mono8", "8uc1"):
        return array_png(raw.reshape(msg.height, msg.step)[:, : msg.width], max_side=max_side)
    if encoding in ("16uc1", "mono16", "32fc1"):
        dtype = np.float32 if encoding == "32fc1" else np.uint16
        depth = np.frombuffer(bytes(msg.data), dtype=dtype).reshape(msg.height, -1)[:, : msg.width].astype(np.float64)
        finite = depth[np.isfinite(depth) & (depth > 0)]
        top = float(finite.max()) if finite.size else 1.0
        return array_png(np.where(np.isfinite(depth), depth / top * 255.0, 0.0), max_side=max_side)
    raise ToolError(f"cannot render image encoding {msg.encoding!r}")


def compressed_msg(msg: CompressedImage) -> Image:
    """MCP image content from a sensor_msgs CompressedImage."""
    fmt = "jpeg" if "jpeg" in msg.format.lower() or "jpg" in msg.format.lower() else "png"
    return Image(data=bytes(msg.data), format=fmt)


def occupancy_png(grid: OccupancyGrid, *, max_side: int = MAX_SIDE) -> Image:
    """MCP image content from a nav_msgs OccupancyGrid: free white, occupied black, unknown grey, row 0 at the bottom."""
    import numpy as np

    cells = np.asarray(grid.data, dtype=np.int16).reshape(grid.info.height, grid.info.width)
    pixels = np.full(cells.shape, 128, dtype=np.uint8)
    known = cells >= 0
    pixels[known] = (255 - cells[known].clip(0, 100) * 255 // 100).astype(np.uint8)
    return array_png(pixels[::-1], max_side=max_side)
