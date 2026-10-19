from __future__ import annotations

import os

import pytest


@pytest.fixture
def local_root(tmp_path):
    """ARENA_ASSETS_DIR_LOCAL pointed at tmp_path for one test, restored after."""
    previous = os.environ.get("ARENA_ASSETS_DIR_LOCAL")
    os.environ["ARENA_ASSETS_DIR_LOCAL"] = str(tmp_path)
    try:
        yield tmp_path
    finally:
        if previous is None:
            del os.environ["ARENA_ASSETS_DIR_LOCAL"]
        else:
            os.environ["ARENA_ASSETS_DIR_LOCAL"] = previous
