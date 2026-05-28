"""Extensible registry of ``z_type`` tag values.

The CORE ``z_type`` column carries a short string label describing what
kind of redshift each row holds (``spec``, ``phot``, ``phot_pdf``,
``pv``, ``none`` for the legacy set). Phase 16 promotes the set to a
runtime registry so surveys can declare new variants
(``spec_lya``, ``cluster_z``, ``xcorr_z``, ...) without editing core
schema code.

This module is **observational metadata only**: a ``z_type`` value is a
label for what a column contains, not a cosmological choice. Frame
disambiguation (CMB vs heliocentric) lives in
:class:`oneuniverse.data.schema.ColumnDef` (``frame`` field) and in
:class:`oneuniverse.data.coordinate_spec.CoordinateSpec`.
"""
from __future__ import annotations

import re
from typing import Iterable, Set

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

Z_TYPE_REGISTRY: Set[str] = {
    "spec",
    "phot",
    "phot_pdf",
    "pv",
    "none",
}

_DESCRIPTIONS: dict = {
    "spec": "spectroscopic redshift",
    "phot": "photometric point estimate",
    "phot_pdf": "photometric redshift with PDF on disk",
    "pv": "peculiar-velocity-derived redshift",
    "none": "no redshift available",
}


def register_z_type(name: str, *, description: str = "") -> None:
    """Add ``name`` to :data:`Z_TYPE_REGISTRY`. Idempotent.

    Raises
    ------
    ValueError
        If ``name`` is not lowercase ASCII matching ``[a-z][a-z0-9_]*``.
    """
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError(
            f"z_type names must be lowercase ASCII matching "
            f"[a-z][a-z0-9_]*; got {name!r}"
        )
    Z_TYPE_REGISTRY.add(name)
    if description and name not in _DESCRIPTIONS:
        _DESCRIPTIONS[name] = description


def is_registered(name: str) -> bool:
    return name in Z_TYPE_REGISTRY


def assert_valid(values: Iterable[str]) -> None:
    """Raise :class:`ValueError` if any value is not registered."""
    unknown = sorted({v for v in values if v not in Z_TYPE_REGISTRY})
    if unknown:
        raise ValueError(
            f"unregistered z_type value(s): {unknown!r}; "
            f"call register_z_type() first, or use one of "
            f"{sorted(Z_TYPE_REGISTRY)!r}"
        )
