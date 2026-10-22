"""Generic loader-registries for lazy kind -> class / factory resolution.

This module is intentionally dependency-free. It lives in arena_rclpy_mixins
because that's currently the only cross-package Python utility library in the
workspace (see utils/arena_rclpy_mixins/). If a dedicated arena_utils package
is ever introduced, this module should move there.
"""

from __future__ import annotations

import typing

K = typing.TypeVar('K')
V = typing.TypeVar('V')


class ClassRegistry(typing.Generic[K, V]):
    """Lazy sync registry: loader returns a class, result cached on first get.

    Typical use:
        REGISTRY: ClassRegistry[str, type[Foo]] = ClassRegistry()

        @REGISTRY.register("kind_a")
        def _load_a() -> type[Foo]:
            from .a import FooA
            return FooA
    """

    def __init__(self) -> None:
        self._loaders: dict[K, typing.Callable[[], V]] = {}
        self._cache: dict[K, V] = {}

    def register(self, key: K) -> typing.Callable[[typing.Callable[[], V]], typing.Callable[[], V]]:
        def _dec(loader: typing.Callable[[], V]) -> typing.Callable[[], V]:
            if key in self._loaders:
                raise ValueError(f"registry key {key!r} already registered")
            self._loaders[key] = loader
            return loader

        return _dec

    def get(self, key: K) -> V:
        if key in self._cache:
            return self._cache[key]
        try:
            loader = self._loaders[key]
        except KeyError:
            raise KeyError(f"no entry for {key!r}; known: {sorted(self._loaders, key=str) if self._loaders else []!r}") from None
        value = loader()
        self._cache[key] = value
        return value

    def keys(self) -> typing.KeysView[K]:
        return self._loaders.keys()

    def __contains__(self, key: object) -> bool:
        return key in self._loaders


class FactoryRegistry(typing.Generic[K, V]):
    """Sync factory registry: loader constructs a fresh instance per get. Not cached.

    Typical use:
        REGISTRY: FactoryRegistry[str, Foo] = FactoryRegistry()

        @REGISTRY.register("kind_a")
        def _make_a(**kwargs) -> Foo:
            from .a import FooA
            return FooA(**kwargs)

        instance = REGISTRY.get("kind_a", arg=1)
    """

    def __init__(self, entries: dict[K, typing.Callable[..., V]] | None = None) -> None:
        self._registry: dict[K, typing.Callable[..., V]] = dict(entries) if entries else {}

    def register(self, key: K) -> typing.Callable[[typing.Callable[..., V]], typing.Callable[..., V]]:
        def _dec(loader: typing.Callable[..., V]) -> typing.Callable[..., V]:
            if key in self._registry:
                raise ValueError(f"registry key {key!r} already registered")
            self._registry[key] = loader
            return loader

        return _dec

    def get(self, key: K, /, *args: object, **kwargs: object) -> V:
        try:
            loader = self._registry[key]
        except KeyError:
            raise KeyError(f"no entry for {key!r}; known: {sorted(self._registry, key=str) if self._registry else []!r}") from None
        return loader(*args, **kwargs)

    def keys(self) -> typing.KeysView[K]:
        return self._registry.keys()

    def __contains__(self, key: object) -> bool:
        return key in self._registry


class AsyncFactoryRegistry(typing.Generic[K, V]):
    """Async factory registry: loader constructs a fresh instance per get. Not cached.

    Typical use:
        REGISTRY: AsyncFactoryRegistry[str, Foo] = AsyncFactoryRegistry()

        @REGISTRY.register("kind_a")
        async def _make_a(**kwargs) -> Foo:
            from .a import FooA
            return FooA(**kwargs)

        instance = await REGISTRY.get("kind_a", arg=1)
    """

    def __init__(self, entries: dict[K, typing.Callable[..., typing.Awaitable[V]]] | None = None) -> None:
        self._registry: dict[K, typing.Callable[..., typing.Awaitable[V]]] = dict(entries) if entries else {}

    def register(self, key: K) -> typing.Callable[[typing.Callable[..., typing.Awaitable[V]]], typing.Callable[..., typing.Awaitable[V]]]:
        def _dec(loader: typing.Callable[..., typing.Awaitable[V]]) -> typing.Callable[..., typing.Awaitable[V]]:
            if key in self._registry:
                raise ValueError(f"registry key {key!r} already registered")
            self._registry[key] = loader
            return loader

        return _dec

    def get(self, key: K, /, *args: object, **kwargs: object) -> typing.Awaitable[V]:
        try:
            loader = self._registry[key]
        except KeyError:
            raise KeyError(f"no entry for {key!r}; known: {sorted(self._registry, key=str) if self._registry else []!r}") from None
        return loader(*args, **kwargs)

    def keys(self) -> typing.KeysView[K]:
        return self._registry.keys()

    def __contains__(self, key: object) -> bool:
        return key in self._registry
