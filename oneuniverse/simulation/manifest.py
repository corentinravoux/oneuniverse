"""OUFSimManifest — the typed contract for one OUF-Sim record.

Mirrors the OUF data ``Manifest`` discipline: pinned format version,
typed sub-specs, ``to_dict`` / ``from_dict``, YAML read/write with a
hard version-compat check. The manifest points at native files +
sidecar indexes; it never holds bulk particle data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import yaml

from oneuniverse.simulation._version import (
    LAYOUT_SCHEMAS,
    PRODUCT_KINDS,
    SIM_KINDS,
)
from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.product import ProductDecl
from oneuniverse.simulation.provenance import ProvenanceSpec
from oneuniverse.simulation.unit_frame import UnitFrameSpec


class OUFSimManifestError(ValueError):
    """Raised on a malformed or version-incompatible OUF-Sim manifest."""


@dataclass(frozen=True)
class OUFSimManifest:
    oufsim_format_version: str
    sim_name: str
    sim_kind: str
    code: str
    code_version: Optional[str]
    layout_schema: str
    backends: Tuple[str, ...]
    has_input: bool
    has_output: bool
    products: Tuple[str, ...]
    n_snapshots: int
    redshifts: Tuple[float, ...]
    box_size: Optional[float]
    n_particles: Optional[int]
    cosmology: Optional[CosmologySpec]
    unit_frame: Optional[UnitFrameSpec]
    provenance: Optional[ProvenanceSpec]
    product_decls: Tuple[ProductDecl, ...] = ()

    def __post_init__(self) -> None:
        if self.sim_kind not in SIM_KINDS:
            raise ValueError(
                f"OUFSimManifest: unknown sim_kind {self.sim_kind!r}; "
                f"allowed: {list(SIM_KINDS)}"
            )
        if self.layout_schema not in LAYOUT_SCHEMAS:
            raise ValueError(
                f"OUFSimManifest: unknown layout_schema "
                f"{self.layout_schema!r}; allowed: {list(LAYOUT_SCHEMAS)}"
            )
        bad = [p for p in self.products if p not in PRODUCT_KINDS]
        if bad:
            raise ValueError(
                f"OUFSimManifest: unknown products {bad!r}; "
                f"allowed: {list(PRODUCT_KINDS)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "oufsim_format_version": self.oufsim_format_version,
            "sim_name": self.sim_name,
            "sim_kind": self.sim_kind,
            "code": self.code,
            "code_version": self.code_version,
            "layout_schema": self.layout_schema,
            "backends": list(self.backends),
            "has_input": bool(self.has_input),
            "has_output": bool(self.has_output),
            "products": list(self.products),
            "n_snapshots": int(self.n_snapshots),
            "redshifts": [float(z) for z in self.redshifts],
            "box_size": self.box_size,
            "n_particles": self.n_particles,
            "cosmology": (
                self.cosmology.to_dict() if self.cosmology is not None else None
            ),
            "unit_frame": (
                self.unit_frame.to_dict()
                if self.unit_frame is not None else None
            ),
            "provenance": (
                self.provenance.to_dict()
                if self.provenance is not None else None
            ),
            "product_decls": [p.to_dict() for p in self.product_decls],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OUFSimManifest":
        cosmo = d.get("cosmology")
        uf = d.get("unit_frame")
        prov = d.get("provenance")
        return cls(
            oufsim_format_version=d["oufsim_format_version"],
            sim_name=d["sim_name"],
            sim_kind=d["sim_kind"],
            code=d["code"],
            code_version=d.get("code_version"),
            layout_schema=d["layout_schema"],
            backends=tuple(d.get("backends", ())),
            has_input=bool(d.get("has_input", False)),
            has_output=bool(d.get("has_output", False)),
            products=tuple(d.get("products", ())),
            n_snapshots=int(d.get("n_snapshots", 0)),
            redshifts=tuple(float(z) for z in d.get("redshifts", ())),
            box_size=d.get("box_size"),
            n_particles=d.get("n_particles"),
            cosmology=CosmologySpec.from_dict(cosmo) if cosmo else None,
            unit_frame=UnitFrameSpec.from_dict(uf) if uf else None,
            provenance=ProvenanceSpec.from_dict(prov) if prov else None,
            product_decls=tuple(
                ProductDecl.from_dict(p) for p in d.get("product_decls", ())
            ),
        )


def write_manifest(path: Union[str, Path], manifest: OUFSimManifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest.to_dict(), sort_keys=False))


def read_manifest(path: Union[str, Path]) -> OUFSimManifest:
    path = Path(path)
    if not path.is_file():
        raise OUFSimManifestError(f"OUF-Sim manifest not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise OUFSimManifestError(f"{path}: invalid YAML ({e})") from e
    if not isinstance(raw, dict):
        raise OUFSimManifestError(f"{path}: top-level must be a mapping")
    fmt = raw.get("oufsim_format_version")
    if not (isinstance(fmt, str) and fmt.startswith("0.1")):
        raise OUFSimManifestError(
            f"{path}: oufsim_format_version={fmt!r} is not compatible "
            f"with this library (expected 0.1.x)."
        )
    try:
        return OUFSimManifest.from_dict(raw)
    except (KeyError, ValueError) as e:
        raise OUFSimManifestError(f"{path}: {e}") from e
