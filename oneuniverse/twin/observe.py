"""Data -> twin socket: grid a dataset view or a measure product into an :class:`Observation`.

Positions must already be box coordinates in ``[0, box)`` (no sky-to-comoving
conversion lives in the package). Counts are deposited with CIC through
``oneuniverse.reference.pm.deposit_cic`` (``observe`` may import ``reference``;
``twin.engine`` may not).
"""
from __future__ import annotations

import numpy as np

from oneuniverse._core.errors import ValidationError
from oneuniverse.data.view import DatasetView
from oneuniverse.measure.product import Product
from oneuniverse.reference.pm import deposit_cic
from oneuniverse.twin.engine import Observation


def _check_inside(pos: np.ndarray, box: float) -> None:
    if len(pos) and not (np.all(np.isfinite(pos)) and pos.min() >= 0.0 and pos.max() < box):
        raise ValidationError(
            f"positions must lie in [0, {box}) (box coordinates); got range "
            f"[{np.nanmin(pos)}, {np.nanmax(pos)}]")


def _product_positions(source: Product, cols) -> np.ndarray:
    """Look each column up in ``extras`` first (the catalog ``z`` is a redshift), then the catalog."""
    arrays = []
    for c in cols:
        if c in source.extras and isinstance(source.extras[c], np.ndarray):
            a = source.extras[c]
        elif c in source.catalog.column_names:
            a = source.catalog[c].to_numpy()
        else:
            raise ValidationError(f"position column {c!r} found neither in extras nor in the catalog")
        if a.shape != (source.n,):
            raise ValidationError(f"position column {c!r} must have shape ({source.n},), got {a.shape}")
        arrays.append(np.asarray(a, dtype=np.float64))
    return np.stack(arrays, axis=1)


def _grid(source, cols, box: float, n_grid: int) -> tuple[np.ndarray, int, dict]:
    counts = np.zeros((n_grid,) * 3)
    total = 0
    if isinstance(source, DatasetView):
        missing = [c for c in cols if c not in source.columns]
        if missing:
            raise ValidationError(f"dataset has no column(s) {missing}; have {source.columns}")
        for batch in source.scan(columns=list(cols)):       # only the position columns, streamed
            pos = np.stack([batch.column(c).to_numpy(zero_copy_only=False).astype(np.float64)
                            for c in cols], axis=1)
            _check_inside(pos, box)
            counts += deposit_cic(pos, n_grid, box)
            total += len(pos)
        meta = {"source": "dataset", "dataset_version": list(source.version)}
    elif isinstance(source, Product):
        pos = _product_positions(source, cols)
        _check_inside(pos, box)
        counts += deposit_cic(pos, n_grid, box)
        total = len(pos)
        meta = {"source": "product", "product_provenance": dict(source.provenance)}
    else:
        raise ValidationError(f"source must be a DatasetView or a Product, got {type(source).__name__}")
    return counts, total, meta


def observe_from_view(source, *, box: float, n_grid: int, position_columns=("x", "y", "z"),
                      nbar: float | None = None, bias: float = 1.0, mask=None) -> Observation:
    """Grid objects into ``delta_g = n/<n>_mask - 1`` (0 outside ``mask``; ``mask=None`` is all cells).

    ``source`` is a ``DatasetView`` (only ``position_columns`` are read, batch by batch)
    or a measure ``Product`` (each column is taken from ``extras`` first, then the
    catalog). ``nbar`` defaults to ``N / box**3`` over all objects when ``mask`` is None; with
    a ``mask`` it defaults to the in-mask number density
    ``counts[mask].sum() / (mask.sum() * (box/n_grid)**3)`` (``delta_g`` is renormalised by
    the in-mask mean, so the Wiener noise term needs the in-mask density). Positions outside
    ``[0, box)`` raise ``ValidationError``.
    """
    box = float(box)
    if not box > 0:
        raise ValidationError(f"box must be > 0, got {box}")
    cols = tuple(position_columns)
    if len(cols) != 3:
        raise ValidationError(f"position_columns must name 3 columns, got {cols}")
    counts, total, meta = _grid(source, cols, box, int(n_grid))
    if mask is None:
        footprint = np.ones(counts.shape, dtype=bool)
    else:
        footprint = np.asarray(mask, dtype=bool)
        if footprint.shape != counts.shape:
            raise ValidationError(f"mask shape {footprint.shape} != grid {counts.shape}")
    mean = counts[footprint].mean() if footprint.any() else 0.0
    if not mean > 0:
        raise ValidationError("no objects inside the mask: cannot form a count contrast")
    delta = np.where(footprint, counts / mean - 1.0, 0.0)
    if nbar is None:
        nbar = (total / box ** 3 if mask is None
                else counts[footprint].sum() / (footprint.sum() * (box / int(n_grid)) ** 3))
    meta.update(columns=list(cols), n_objects=int(total))
    return Observation(delta, float(nbar), float(bias), None if mask is None else footprint,
                       box, meta)
