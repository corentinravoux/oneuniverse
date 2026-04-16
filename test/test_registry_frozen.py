"""Tests for the frozen public registry view (Phase 6 Task 5)."""

from __future__ import annotations

import pytest

from oneuniverse.data._registry import REGISTRY, _REGISTRY


def test_public_registry_is_mapping_proxy():
    from types import MappingProxyType

    assert isinstance(REGISTRY, MappingProxyType)


def test_public_registry_rejects_writes():
    with pytest.raises(TypeError):
        REGISTRY["__new_survey__"] = object()  # type: ignore[index]


def test_public_registry_reflects_internal_dict():
    # Mutating _REGISTRY is allowed (tests + @register do this) and the
    # proxy observes the change immediately.
    key = "__proxy_observes_writes__"
    assert key not in REGISTRY
    _REGISTRY[key] = object
    try:
        assert key in REGISTRY
    finally:
        _REGISTRY.pop(key, None)
