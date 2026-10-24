from __future__ import annotations

from mcp.server.mcpserver.exceptions import ToolError as _SdkToolError


class ToolError(_SdkToolError):
    """A tool failure the caller can act on, with an optional next step."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        self.message = message
        self.hint = hint
        super().__init__(f"{message}\nhint: {hint}" if hint else message)
