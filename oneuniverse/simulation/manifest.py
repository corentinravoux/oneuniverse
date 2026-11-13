"""OUF-Sim 1.0 JSON manifest: typed, validated on read, atomically written."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from oneuniverse._core.errors import FormatError, NotFoundError, ValidationError
from oneuniverse._core.fs import atomic_write_json
from oneuniverse.simulation.specs import (
    CosmologySpec,
    ProductDecl,
    ProvenanceSpec,
    UnitFrameSpec,
    _plain,
    _segment,
)

OUFSIM_FORMAT_VERSION = "1.0.0"
MANIFEST_FILENAME = "manifest.json"
LAYOUT_FILENAME = "layout.json"

_REQUIRED = ("format_version", "name", "box_size", "cosmology", "units", "provenance",
             "products", "native", "created_utc")


def _major(version: Any) -> str:
    return str(version).split(".")[0]


@dataclass(frozen=True, slots=True)
class OUFSimManifest:
    format_version: str
    name: str
    box_size: float
    cosmology: CosmologySpec
    units: UnitFrameSpec
    provenance: ProvenanceSpec
    products: tuple[ProductDecl, ...]
    native: dict
    created_utc: str
    parent: str | None = None
    lineage: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        _segment(self.name, "simulation name")
        box = float(self.box_size)
        if not box > 0 or box == float("inf"):
            raise ValidationError(f"box_size must be finite and > 0, got {self.box_size}")
        object.__setattr__(self, "box_size", box)
        products = tuple(self.products)
        names = [p.name for p in products]
        if len(set(names)) != len(names):
            raise ValidationError(f"duplicate product names in {names}")
        object.__setattr__(self, "products", products)
        object.__setattr__(self, "native", _plain(dict(self.native)))
        object.__setattr__(self, "lineage", _plain(dict(self.lineage)))
        if self.parent is not None:
            _segment(self.parent, "parent name")

    def product(self, name: str) -> ProductDecl:
        for p in self.products:
            if p.name == name:
                return p
        raise NotFoundError(f"no product {name!r} in simulation {self.name!r}; have "
                            f"{[p.name for p in self.products]}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version, "name": self.name, "box_size": self.box_size,
            "cosmology": self.cosmology.to_dict(), "units": self.units.to_dict(),
            "provenance": self.provenance.to_dict(),
            "products": [p.to_dict() for p in self.products],
            "native": _plain(self.native), "created_utc": self.created_utc,
            "parent": self.parent, "lineage": _plain(self.lineage),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OUFSimManifest:
        """Rebuild a manifest; ``FormatError`` on a missing key, bad value or other major version."""
        if not isinstance(d, dict):
            raise FormatError(f"manifest must be a JSON object, got {type(d).__name__}")
        missing = [k for k in _REQUIRED if k not in d]
        if missing:
            raise FormatError(f"manifest is missing required key(s) {missing}")
        if _major(d["format_version"]) != _major(OUFSIM_FORMAT_VERSION):
            raise FormatError(
                f"unsupported OUF-Sim format version {d['format_version']!r}; "
                f"this reader handles {OUFSIM_FORMAT_VERSION} (major {_major(OUFSIM_FORMAT_VERSION)})")
        try:
            return cls(
                d["format_version"], d["name"], d["box_size"],
                CosmologySpec.from_dict(d["cosmology"]), UnitFrameSpec.from_dict(d["units"]),
                ProvenanceSpec.from_dict(d["provenance"]),
                tuple(ProductDecl.from_dict(p) for p in d["products"]),
                d["native"], d["created_utc"], d.get("parent"), d.get("lineage", {}))
        except (ValueError, TypeError) as exc:  # ValidationError is a ValueError
            raise FormatError(f"invalid manifest: {exc}") from exc


def write_manifest(path: str | Path, m: OUFSimManifest) -> None:
    atomic_write_json(path, m.to_dict())


def read_manifest(path: str | Path) -> OUFSimManifest:
    path = Path(path)
    try:
        d = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise FormatError(f"{path} is not valid JSON: {exc}") from exc
    return OUFSimManifest.from_dict(d)
