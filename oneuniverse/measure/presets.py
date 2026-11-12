"""Probe presets: thin wrappers over :func:`build_pointset`.

Each preset builds Products on one shared Regions object and packs them in a
MeasurementSet. Nothing here depends on a cosmology.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from oneuniverse._core.errors import ValidationError
from oneuniverse.combine.weights import ShearWeight, Weight
from oneuniverse.data.view import DatasetView
from oneuniverse.measure.atoms import Regions, Window
from oneuniverse.measure.measurement_set import MeasurementSet, MeasurementSpec
from oneuniverse.measure.photoz import attach_photoz, tomographic_nz
from oneuniverse.measure.pipeline import _build_pointset

_SHEAR_COLUMNS = ("e1", "e2")


def galaxy_clustering(view: DatasetView, *, z: tuple[float, float], weight: Weight | None = None,
                      nz_edges: Sequence[float], mask: Window | None = None,
                      regions: Regions | int = 20, randoms="generate",
                      statistics: Sequence[str] = ("pk_multipoles",), name: str = "clustering",
                      **kw) -> MeasurementSet:
    """One tracer "gal" with randoms; extra keywords go to ``build_pointset``."""
    prod, reg = _build_pointset(view, z=z, weight=weight, nz_edges=nz_edges, mask=mask,
                                regions=regions, randoms=randoms, **kw)
    return MeasurementSet({"gal": prod}, MeasurementSpec(name, ("gal",), tuple(statistics)), reg)


def cosmic_shear(view: DatasetView, *, z: tuple[float, float], shear_weight: ShearWeight,
                 tomo_edges: Sequence[float], nz_edges: Sequence[float],
                 mask: Window | None = None, regions: Regions | int = 20, n_draws: int = 20,
                 name: str = "cosmic_shear", **kw) -> MeasurementSet:
    """One tracer "src": shear weight, e1/e2 extras, photo-z draws, tomographic n(z)."""
    for k in ("randoms", "keep_columns"):
        if k in kw:
            raise ValidationError(f"cosmic_shear sets {k} itself; do not pass it")
    seed = int(kw.get("seed", 0))
    prod, reg = _build_pointset(
        view, z=z, weight=shear_weight, nz_edges=nz_edges, mask=mask, regions=regions,
        randoms="none", keep_columns=[*_SHEAR_COLUMNS, *shear_weight.columns], **kw)
    for c in _SHEAR_COLUMNS:
        prod = prod.with_extra(c, np.asarray(prod.catalog[c].to_numpy(), np.float64))
    prod = attach_photoz(prod, view, n_draws=n_draws, seed=seed)
    prod, _ = tomographic_nz(prod, bin_edges=tomo_edges, nz_edges=nz_edges)
    spec = MeasurementSpec(name, ("src",), ("xi_plus", "xi_minus"))
    return MeasurementSet({"src": prod}, spec, reg)


def three_by_two(lens: MeasurementSet, source: MeasurementSet, *,
                 name: str = "3x2pt") -> MeasurementSet:
    """Combine a lens and a source set (one product each, same regions)."""
    if lens.regions != source.regions:
        raise ValidationError("lens and source sets must share the same regions")
    if len(lens.products) != 1 or len(source.products) != 1:
        raise ValidationError("three_by_two needs exactly one product in each set")
    (lt,), (st,) = lens.spec.tracers, source.spec.tracers
    if lt == st:
        raise ValidationError(f"lens and source tracer names must differ, both are {lt!r}")
    stats = tuple(dict.fromkeys([*lens.spec.statistics, *source.spec.statistics]))
    spec = MeasurementSpec(name, (lt, st), stats, pairs=((lt, lt), (lt, st), (st, st)))
    return MeasurementSet({lt: lens.products[lt], st: source.products[st]}, spec, lens.regions)
