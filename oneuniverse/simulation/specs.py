"""Typed, validated, serialisable specifications shared by the simulation layer.

``CosmologySpec`` is the only cosmology container in the package.  Every spec is
a frozen dataclass whose ``to_dict`` output is plain JSON and whose ``from_dict``
inverts it exactly.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from oneuniverse._core.errors import ValidationError
from oneuniverse._core.geometry import Cube

PRODUCT_KINDS = ("particles", "halos", "field", "lightcone", "checkpoint")
PHYSICS = ("dm", "hydro")
IC_STRATEGIES = ("parent_field", "resample")
REQUEST_STATUSES = ("pending", "running", "ingested", "failed")

_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


def _segment(value: Any, what: str) -> str:
    if not isinstance(value, str) or not _SEGMENT.match(value):
        raise ValidationError(f"{what} must be a non-empty safe path segment, got {value!r}")
    return value


def _choice(value: Any, allowed: tuple[str, ...], what: str) -> str:
    if value not in allowed:
        raise ValidationError(f"{what} must be one of {allowed}, got {value!r}")
    return value


def _plain(obj: Any) -> Any:
    """Deep-convert tuples to lists so dict fields survive a JSON round trip unchanged."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def _need(d: Any, keys: tuple[str, ...], what: str) -> None:
    if not isinstance(d, dict):
        raise ValidationError(f"{what} must be a mapping, got {type(d).__name__}")
    for k in keys:
        if k not in d:
            raise ValidationError(f"{what} is missing required key {k!r}")


def _set(obj: Any, name: str, value: Any) -> None:
    object.__setattr__(obj, name, value)


def _positive_int_or_none(value: Any, what: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or int(value) != value or value <= 0:
        raise ValidationError(f"{what} must be a positive integer or None, got {value!r}")
    return int(value)


@dataclass(frozen=True, slots=True)
class CosmologySpec:
    h: float
    omega_m: float
    omega_b: float
    sigma8: float
    n_s: float
    w0: float = -1.0
    wa: float = 0.0

    def __post_init__(self) -> None:
        for name in ("h", "omega_m", "omega_b", "sigma8", "n_s", "w0", "wa"):
            v = float(getattr(self, name))
            if not math.isfinite(v):
                raise ValidationError(f"{name} must be finite, got {v}")
            _set(self, name, v)
        if self.h <= 0:
            raise ValidationError(f"h must be > 0, got {self.h}")
        if not 0 < self.omega_b < self.omega_m:
            raise ValidationError(
                f"require 0 < omega_b < omega_m, got omega_b={self.omega_b}, omega_m={self.omega_m}")
        if self.sigma8 <= 0:
            raise ValidationError(f"sigma8 must be > 0, got {self.sigma8}")
        if self.n_s <= 0:
            raise ValidationError(f"n_s must be > 0, got {self.n_s}")

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ("h", "omega_m", "omega_b", "sigma8", "n_s", "w0", "wa")}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CosmologySpec:
        _need(d, ("h", "omega_m", "omega_b", "sigma8", "n_s"), "CosmologySpec")
        return cls(**{k: d[k] for k in ("h", "omega_m", "omega_b", "sigma8", "n_s", "w0", "wa") if k in d})


@dataclass(frozen=True, slots=True)
class UnitFrameSpec:
    length: str = "Mpc/h"
    velocity: str = "km/s"
    mass: str = "Msun/h"
    comoving: bool = True

    def __post_init__(self) -> None:
        for name in ("length", "velocity", "mass"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValidationError(f"{name} unit must be a non-empty string")
        _set(self, "comoving", bool(self.comoving))

    def to_dict(self) -> dict[str, Any]:
        return {"length": self.length, "velocity": self.velocity, "mass": self.mass,
                "comoving": self.comoving}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> UnitFrameSpec:
        _need(d, (), "UnitFrameSpec")
        return cls(**{k: d[k] for k in ("length", "velocity", "mass", "comoving") if k in d})


@dataclass(frozen=True, slots=True)
class ProvenanceSpec:
    code: str
    version: str
    seed: int | None
    config_hash: str
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("code", "version", "config_hash"):
            if not isinstance(getattr(self, name), str):
                raise ValidationError(f"{name} must be a string")
        if self.seed is not None:
            _set(self, "seed", int(self.seed))
        _set(self, "extra", _plain(dict(self.extra)))

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "version": self.version, "seed": self.seed,
                "config_hash": self.config_hash, "extra": _plain(self.extra)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProvenanceSpec:
        _need(d, ("code", "version", "seed", "config_hash"), "ProvenanceSpec")
        return cls(d["code"], d["version"], d["seed"], d["config_hash"], d.get("extra", {}))


@dataclass(frozen=True, slots=True)
class ProductDecl:
    name: str
    kind: str
    redshifts: tuple[float, ...]
    columns: tuple[str, ...] = ()
    n_grid: int | None = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        _segment(self.name, "product name")
        _choice(self.kind, PRODUCT_KINDS, "product kind")
        z = tuple(float(v) for v in self.redshifts)
        if not z:
            raise ValidationError("redshifts must be non-empty")
        if not all(math.isfinite(v) and v >= 0 for v in z):
            raise ValidationError(f"redshifts must be finite and >= 0, got {z}")
        _set(self, "redshifts", z)
        cols = tuple(self.columns)
        if not all(isinstance(c, str) for c in cols):
            raise ValidationError("columns must be strings")
        _set(self, "columns", cols)
        _set(self, "n_grid", _positive_int_or_none(self.n_grid, "n_grid"))
        _set(self, "extra", _plain(dict(self.extra)))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "redshifts": list(self.redshifts),
                "columns": list(self.columns), "n_grid": self.n_grid, "extra": _plain(self.extra)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProductDecl:
        _need(d, ("name", "kind", "redshifts"), "ProductDecl")
        return cls(d["name"], d["kind"], tuple(d["redshifts"]), tuple(d.get("columns", ())),
                   d.get("n_grid"), d.get("extra", {}))


def _cube_to_dict(c: Cube) -> dict[str, list[float]]:
    return {"lo": list(c.lo), "hi": list(c.hi)}


def _cube_from_dict(d: dict[str, Any]) -> Cube:
    _need(d, ("lo", "hi"), "Cube")
    return Cube(tuple(d["lo"]), tuple(d["hi"]))


@dataclass(frozen=True, slots=True)
class RegionSpec:
    cube: Cube
    buffer: float = 0.0
    n_grid: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cube, Cube):
            raise ValidationError("RegionSpec.cube must be a Cube")
        b = float(self.buffer)
        if not math.isfinite(b) or b < 0:
            raise ValidationError(f"buffer must be finite and >= 0, got {b}")
        _set(self, "buffer", b)
        _set(self, "n_grid", _positive_int_or_none(self.n_grid, "n_grid"))

    @property
    def outer(self) -> Cube:
        """The cube grown by ``buffer`` on every side (may exceed the parent box)."""
        b = self.buffer
        return Cube(tuple(v - b for v in self.cube.lo), tuple(v + b for v in self.cube.hi))

    def to_dict(self) -> dict[str, Any]:
        return {"cube": _cube_to_dict(self.cube), "buffer": self.buffer, "n_grid": self.n_grid}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RegionSpec:
        _need(d, ("cube",), "RegionSpec")
        return cls(_cube_from_dict(d["cube"]), d.get("buffer", 0.0), d.get("n_grid"))


@dataclass(frozen=True, slots=True)
class SimulationRequest:
    id: str
    parent: str
    region: RegionSpec
    physics: str = "dm"
    ic_strategy: str = "parent_field"
    status: str = "pending"
    child: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        _segment(self.id, "request id")
        _segment(self.parent, "parent name")
        if not isinstance(self.region, RegionSpec):
            raise ValidationError("SimulationRequest.region must be a RegionSpec")
        _choice(self.physics, PHYSICS, "physics")
        _choice(self.ic_strategy, IC_STRATEGIES, "ic_strategy")
        _choice(self.status, REQUEST_STATUSES, "status")
        if self.child is not None:
            _segment(self.child, "child name")

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "parent": self.parent, "region": self.region.to_dict(),
                "physics": self.physics, "ic_strategy": self.ic_strategy, "status": self.status,
                "child": self.child, "error": self.error}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SimulationRequest:
        _need(d, ("id", "parent", "region"), "SimulationRequest")
        kw = {k: d[k] for k in ("physics", "ic_strategy", "status", "child", "error") if k in d}
        return cls(d["id"], d["parent"], RegionSpec.from_dict(d["region"]), **kw)
