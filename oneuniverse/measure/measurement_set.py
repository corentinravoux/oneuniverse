"""MeasurementSet: named products, a spec and shared jackknife regions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pyarrow.compute as pc

from oneuniverse._core.errors import ValidationError
from oneuniverse.measure.atoms import Regions
from oneuniverse.measure.product import FieldMap, Product

_COVARIANCES = ("jackknife", "mocks", "analytic", "external")


def _names(value, what: str) -> tuple[str, ...]:
    if isinstance(value, str) or not hasattr(value, "__iter__"):
        raise ValidationError(f"{what} must be a sequence of names, not {type(value).__name__}")
    out = tuple(value)
    if not all(isinstance(v, str) for v in out):
        raise ValidationError(f"{what} must contain only strings")
    return out


@dataclass(frozen=True)
class MeasurementSpec:
    name: str
    tracers: tuple[str, ...]
    statistics: tuple[str, ...]
    pairs: tuple[tuple[str, str], ...] = ()
    covariance: Literal["jackknife", "mocks", "analytic", "external"] = "jackknife"
    notes: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "tracers", _names(self.tracers, "tracers"))
        object.__setattr__(self, "statistics", _names(self.statistics, "statistics"))
        pairs = tuple(_names(p, "pair") for p in self.pairs)
        if any(len(p) != 2 for p in pairs):
            raise ValidationError("each pair must name exactly two tracers")
        object.__setattr__(self, "pairs", pairs)
        if self.covariance not in _COVARIANCES:
            raise ValidationError(f"covariance must be one of {_COVARIANCES}, got {self.covariance!r}")


def _region_range(table, label: str, n_regions: int, tracer: str) -> None:
    if table is None or table.num_rows == 0:
        return
    col = table.column("region_id")
    if col.null_count:
        raise ValidationError(
            f"{label} of {tracer!r} has {col.null_count} null region_id value(s)")
    mm = pc.min_max(col)
    lo, hi = mm["min"].as_py(), mm["max"].as_py()
    if lo < 0 or hi >= n_regions:
        raise ValidationError(
            f"{label} of {tracer!r} has region_id in [{lo}, {hi}], "
            f"outside [0, {n_regions}) of the regions")


@dataclass(frozen=True)
class MeasurementSet:
    products: dict[str, Product]
    spec: MeasurementSpec
    regions: Regions

    def __post_init__(self):
        object.__setattr__(self, "products", dict(self.products))
        if self.spec.tracers != tuple(self.products):
            raise ValidationError(
                f"spec tracers {self.spec.tracers} do not match product names {tuple(self.products)}")
        for a, b in self.spec.pairs:
            for t in (a, b):
                if t not in self.products:
                    raise ValidationError(f"spec pairs name unknown tracer {t!r} in ({a!r}, {b!r})")
        for name, p in self.products.items():
            if isinstance(p, FieldMap):
                continue
            _region_range(p.catalog, "catalog", self.regions.n_regions, name)
            _region_range(p.randoms, "randoms", self.regions.n_regions, name)

    def summary(self) -> dict:
        prods = {}
        for name, p in self.products.items():
            prods[name] = {
                "kind": p.kind,
                "n": int(p.n),
                "n_randoms": 0 if p.randoms is None else int(p.randoms.num_rows),
                "has_window": p.window is not None,
                "has_nz": p.nz is not None,
                "extras": sorted(p.extras),
            }
        return {
            "name": self.spec.name,
            "tracers": list(self.spec.tracers),
            "statistics": list(self.spec.statistics),
            "pairs": [list(p) for p in self.spec.pairs],
            "covariance": self.spec.covariance,
            "products": prods,
            "regions": {"nside": int(self.regions.nside), "nest": bool(self.regions.nest),
                        "n_regions": int(self.regions.n_regions), "method": self.regions.method},
        }
