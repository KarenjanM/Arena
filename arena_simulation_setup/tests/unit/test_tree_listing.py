"""The child-process listing must agree with the in-process walk, pruning included."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from pathlib import Path

import attrs
import pytest
from arena_simulation_setup.tree import LocalResolver, SimplePathResolver


@attrs.define(frozen=True)
class _Ident:
    parts: tuple[str, ...]

    @classmethod
    def from_relpath(cls, relpath: Path) -> _Ident:
        if len(relpath.parts) != 2:
            raise ValueError(relpath)
        return cls(relpath.parts)

    @classmethod
    def label(cls) -> str:
        return "test"


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "pkg" / "asset" / "deeper").mkdir(parents=True)
    (tmp_path / "pkg" / "asset" / "deeper" / "ignored.yaml").touch()
    (tmp_path / "loose").mkdir()
    (tmp_path / "loose" / "thing.yaml").touch()
    (tmp_path / "loose" / "README.md").touch()
    (tmp_path / "toplevel.yaml").touch()
    return tmp_path


def test_async_listing_matches_the_walk(tree: Path) -> None:
    resolver = SimplePathResolver(_Ident, path=tree)
    assert set(resolver.listall()) == set(asyncio.run(resolver.listall_async()))


def test_prunes_below_an_asset_and_skips_readme(tree: Path) -> None:
    resolver = SimplePathResolver(_Ident, path=tree)
    found = {ident.parts for ident in asyncio.run(resolver.listall_async())}
    assert ("pkg", "asset") in found
    assert ("loose", "thing.yaml") in found
    assert not any("deeper" in parts for parts in found)
    assert not any("README.md" in parts for parts in found)


def test_listing_is_cached_until_invalidated(tree: Path) -> None:
    resolver = SimplePathResolver(_Ident, path=tree)
    asyncio.run(resolver.listall_async())
    (tree / "fresh").mkdir()
    (tree / "fresh" / "thing.yaml").touch()
    assert not any("fresh" in ident.parts for ident in asyncio.run(resolver.listall_async()))
    resolver.invalidate()
    assert any("fresh" in ident.parts for ident in asyncio.run(resolver.listall_async()))


def test_missing_directory_lists_empty(tmp_path: Path) -> None:
    resolver = SimplePathResolver(_Ident, path=tmp_path / "nope")
    assert asyncio.run(resolver.listall_async()) == []


@pytest.fixture
def local_root(tmp_path: Path) -> Iterator[Path]:
    previous = os.environ.get("ARENA_ASSETS_DIR_LOCAL")
    os.environ["ARENA_ASSETS_DIR_LOCAL"] = str(tmp_path)
    try:
        yield tmp_path
    finally:
        if previous is None:
            del os.environ["ARENA_ASSETS_DIR_LOCAL"]
        else:
            os.environ["ARENA_ASSETS_DIR_LOCAL"] = previous


def test_local_listing_walks_like_the_base_and_sees_every_write(local_root: Path, tree: Path) -> None:
    resolver = LocalResolver(_Ident, subdir="")
    assert set(asyncio.run(resolver.listall_async())) == set(resolver.listall())
    (tree / "fresh").mkdir()
    (tree / "fresh" / "thing.yaml").touch()
    assert any("fresh" in ident.parts for ident in asyncio.run(resolver.listall_async()))


def test_local_listing_follows_the_variable_at_every_call(local_root: Path, tree: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    resolver = LocalResolver(_Ident, subdir="")
    assert asyncio.run(resolver.listall_async())
    os.environ["ARENA_ASSETS_DIR_LOCAL"] = str(tmp_path_factory.mktemp("elsewhere"))
    assert asyncio.run(resolver.listall_async()) == []
