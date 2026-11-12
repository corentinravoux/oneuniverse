"""The measurement spine: OUF view -> Product (or Sightline).

``build_pointset`` runs select -> clean -> quality -> weight -> window ->
regions -> n(z) -> randoms -> provenance in one pass. Every stage that leaves
no row raises ValidationError ("empty ...") before randoms are constructed.
Nothing here depends on a cosmology.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import healpy as hp
import numpy as np
import pyarrow as pa

from oneuniverse._core.errors import FormatError, ValidationError
from oneuniverse.combine.weights import QualityMaskWeight, Weight
from oneuniverse.data.schema import Geometry
from oneuniverse.data.view import DatasetView
from oneuniverse.measure.atoms import Nz, Regions, Window
from oneuniverse.measure.product import CATALOG_CORE, RANDOMS_CORE, Product, Sightline

_WINDOW_NSIDE = 256     # footprint fallback resolution
_REGION_NSIDE = 64      # upper bound of the jackknife-region resolution


def _empty(stage: str) -> ValidationError:
    return ValidationError(f"empty selection after {stage}: no rows left to build a product from")


def _f64(table: pa.Table, name: str) -> np.ndarray:
    col = table[name].cast(pa.float64()).combine_chunks()
    return np.asarray(col.fill_null(np.nan).to_numpy(zero_copy_only=False), np.float64)


def _filter(table: pa.Table, keep: np.ndarray) -> pa.Table:
    return table if keep.all() else table.filter(pa.array(keep))


def _dedupe(names) -> list[str]:
    return list(dict.fromkeys(names))


def _clamped_regions(window: Window, n: int, seed: int) -> Regions:
    """Regions.from_window with n clamped to the masked pixels at the region nside."""
    nside = min(_REGION_NSIDE, window.nside)
    mask = window.mask if window.nest else hp.reorder(window.mask, r2n=True)
    fine = np.flatnonzero(mask > 0)
    n_pix = len(fine) if nside == window.nside else len(np.unique(
        fine // (4 ** int(np.log2(window.nside // nside)))))
    return Regions.from_window(window, max(1, min(int(n), n_pix)), nside=nside, seed=seed)


def _regions_arg(window: Window, regions, seed: int) -> Regions:
    if isinstance(regions, Regions):
        return regions
    if isinstance(regions, bool) or not isinstance(regions, (int, np.integer)):
        raise ValidationError(f"regions must be a Regions or an int, got {type(regions).__name__}")
    return _clamped_regions(window, int(regions), seed)


def _window_for(ra: np.ndarray, dec: np.ndarray, mask: Window | None) -> Window:
    if mask is not None:
        if not isinstance(mask, Window):
            raise ValidationError(f"mask must be a Window, got {type(mask).__name__}")
        return mask
    return Window.footprint(ra, dec, nside=_WINDOW_NSIDE)


def generate_randoms(window: Window, nz: Nz, regions: Regions, *, n: int, seed: int) -> pa.Table:
    """Exactly *n* random points following the window and the (1-D) n(z).

    Pixels are drawn with probability proportional to the mask; the position is
    the centre of a uniformly chosen NEST child at nside*8; z comes from the
    inverse CDF of ``nz.normalised()`` (uniform inside each bin); weight 1.
    Deterministic for a seed.
    """
    n = int(n)
    if n < 1:
        raise ValidationError("n must be >= 1")
    nzn = nz.normalised()
    if nzn.counts.ndim != 1:
        raise ValidationError("generate_randoms needs a 1-D Nz")
    mask = np.asarray(window.mask, np.float64)
    if not window.nest:
        mask = hp.reorder(mask, r2n=True)
    if not mask.sum() > 0:
        raise _empty("window")
    cdf_pix = np.cumsum(mask)
    cdf_pix /= cdf_pix[-1]
    cdf_z = np.cumsum(nzn.counts)
    cdf_z /= cdf_z[-1]
    edges = nzn.edges
    fine = window.nside * 8
    rng = np.random.default_rng(seed)
    chunks, have = [], 0
    for _ in range(100):
        need = n - have
        pix = np.minimum(np.searchsorted(cdf_pix, rng.random(need), side="right"), len(mask) - 1)
        child = pix * 64 + rng.integers(0, 64, need)
        ra, dec = hp.pix2ang(fine, child, nest=True, lonlat=True)
        u = rng.random(need)
        b = np.minimum(np.searchsorted(cdf_z, u, side="right"), len(edges) - 2)
        lo_cdf = np.where(b > 0, cdf_z[np.maximum(b - 1, 0)], 0.0)
        frac = (u - lo_cdf) / np.maximum(cdf_z[b] - lo_cdf, 1e-300)
        z = edges[b] + np.clip(frac, 0.0, 1.0) * (edges[b + 1] - edges[b])
        rid = regions.assign(ra, dec)
        ok = rid >= 0
        chunks.append((ra[ok], dec[ok], z[ok], rid[ok]))
        have += int(ok.sum())
        if have >= n:
            break
    else:
        raise ValidationError("could not draw enough randoms inside the regions")
    ra, dec, z, rid = (np.concatenate([c[i] for c in chunks])[:n] for i in range(4))
    return pa.table({"ra": ra, "dec": dec, "z": z, "weight": np.ones(n),
                     "region_id": rid.astype(np.int32)}).select(list(RANDOMS_CORE))


def randoms_from_view(view: DatasetView, *, z: tuple[float, float], window: Window,
                      regions: Regions, n_max: int | None = None) -> pa.Table:
    """Randoms ingested from a view: rows in *z* and inside *window*.

    Region ids are assigned from *regions* (rows outside are dropped); the
    ``weight`` column is used when present, else 1. *n_max* subsamples
    deterministically (fixed seed, row order kept). No positions are carried.
    """
    cols = ["ra", "dec", "z"] + (["weight"] if "weight" in view.columns else [])
    t = view.table(columns=cols, z=z)
    if t.num_rows == 0:
        raise _empty("selecting the randoms")
    ra, dec, zz = (_f64(t, c) for c in ("ra", "dec", "z"))
    w = _f64(t, "weight") if "weight" in cols else np.ones(t.num_rows)
    rid = regions.assign(ra, dec)
    keep = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(zz) & np.isfinite(w) \
        & window.contains(ra, dec) & (rid >= 0)
    ra, dec, zz, w, rid = ra[keep], dec[keep], zz[keep], w[keep], rid[keep]
    if len(ra) == 0:
        raise _empty("applying the window to the randoms")
    if n_max is not None and len(ra) > int(n_max):
        sel = np.sort(np.random.default_rng(0).choice(len(ra), int(n_max), replace=False))
        ra, dec, zz, w, rid = ra[sel], dec[sel], zz[sel], w[sel], rid[sel]
    return pa.table({"ra": ra, "dec": dec, "z": zz, "weight": w,
                     "region_id": rid.astype(np.int32)}).select(list(RANDOMS_CORE))


def build_pointset(view: DatasetView, *, z: tuple[float, float], weight: Weight | None = None,
                   nz_edges: Sequence[float], mask: Window | None = None,
                   regions: Regions | int = 20,
                   randoms: Literal["generate", "none"] | DatasetView = "generate",
                   random_factor: int = 10, keep_columns: Sequence[str] = (),
                   quality: tuple[str, str, float] | None = None, cone=None, patch=None,
                   seed: int = 0) -> Product:
    """Build a point-set Product from an OUF view.

    Steps: scan (with positions) -> drop non-finite ra/dec/z -> quality cut ->
    weight (None means 1) -> window (the given mask, else the footprint of the
    selected points at nside 256) -> drop rows outside the window -> regions ->
    weighted spec_hist n(z) on *nz_edges* -> randoms -> provenance.

    An int *regions* builds ``Regions.from_window(window, n, nside=min(64,
    window.nside), seed=seed)``: the region resolution is capped at nside 64,
    and n is clamped to the number of masked region pixels so small windows
    still work (the clamped count is recorded as provenance["n_regions"]).
    Rows that fall in no region are dropped. randoms="generate" draws
    ``random_factor * n`` points (see :func:`generate_randoms`); a DatasetView
    is ingested (see :func:`randoms_from_view`); "none" gives no randoms.
    The catalog holds CATALOG_CORE then *keep_columns*.
    """
    if isinstance(keep_columns, str):
        raise ValidationError("keep_columns must be a sequence of names, not a str")
    keep_columns = [str(c) for c in keep_columns]
    if not (isinstance(randoms, DatasetView) or randoms in ("generate", "none")):
        raise ValidationError(f"randoms must be 'generate', 'none' or a DatasetView, got {randoms!r}")
    if randoms == "generate" and int(random_factor) < 1:
        raise ValidationError("random_factor must be >= 1")
    qual = None
    if quality is not None:
        qcol, qop, qthr = quality
        qual = QualityMaskWeight(qcol, qop, float(qthr))
    need = _dedupe(["ra", "dec", "z", "source_row", *(weight.columns if weight else ()),
                    *keep_columns, *([qual.column] if qual else [])])
    t = view.table(columns=need, with_position=True, z=z, cone=cone, patch=patch)
    if t.num_rows == 0:
        raise _empty("selection")
    ra, dec, zz = (_f64(t, c) for c in ("ra", "dec", "z"))
    t = _filter(t, np.isfinite(ra) & np.isfinite(dec) & np.isfinite(zz))
    if t.num_rows == 0:
        raise _empty("dropping non-finite ra/dec/z")
    if qual is not None:
        t = _filter(t, qual(t) > 0)
        if t.num_rows == 0:
            raise _empty("the quality cut")
    ra, dec = _f64(t, "ra"), _f64(t, "dec")
    window = _window_for(ra, dec, mask)
    inside = window.contains(ra, dec)
    t = _filter(t, inside)
    if t.num_rows == 0:
        raise _empty("applying the window")
    ra, dec = _f64(t, "ra"), _f64(t, "dec")
    reg = _regions_arg(window, regions, seed)
    rid = reg.assign(ra, dec)
    t = _filter(t, rid >= 0)
    if t.num_rows == 0:
        raise _empty("region assignment")
    ra, dec, zz = (_f64(t, c) for c in ("ra", "dec", "z"))
    rid = reg.assign(ra, dec)
    w = weight(t) if weight is not None else np.ones(t.num_rows)
    catalog = pa.table({
        "ra": ra, "dec": dec, "z": zz, "weight": w, "region_id": rid.astype(np.int32),
        "source_row": t["source_row"], "partition": t["partition"], "offset": t["offset"],
        **{c: t[c] for c in keep_columns if c not in CATALOG_CORE},
    })
    counts, _ = np.histogram(zz, np.asarray(nz_edges, np.float64), weights=w)
    nz = Nz(nz_edges, counts, "spec_hist")
    if randoms == "generate":
        rnd = generate_randoms(window, nz, reg, n=int(random_factor) * catalog.num_rows, seed=seed)
        rsrc = "generated"
    elif randoms == "none":
        rnd, rsrc = None, "none"
    else:
        rnd, rsrc = randoms_from_view(randoms, z=z, window=window, regions=reg), "ingested"
    prov = {
        "dataset": view.manifest.survey.name, "version": list(view.version),
        "z": [float(z[0]), float(z[1])],
        "weight": weight.describe() if weight is not None else "none",
        "window_source": window.source, "randoms_source": rsrc, "nz_method": nz.method,
        "n": catalog.num_rows, "seed": int(seed), "n_regions": reg.n_regions,
        "quality": list(quality) if quality is not None else None,
        "keep_columns": keep_columns,
    }
    return Product(catalog=catalog, randoms=rnd, window=window, nz=nz, provenance=prov)


def build_sightlines(view: DatasetView, *, z: tuple[float, float],
                     regions: Regions | int = 20, mask: Window | None = None,
                     cone=None, patch=None, seed: int = 0) -> Sightline:
    """Build a Sightline product from a SIGHTLINE-geometry view.

    Window and regions follow :func:`build_pointset` (footprint fallback at
    nside 256, regions capped at nside 64). The OUF list column ``weight``
    becomes ``pixel_weight``; the scalar per-sightline ``weight`` is 1.
    """
    if view.geometry != Geometry.SIGHTLINE:
        raise FormatError(
            f"build_sightlines needs a SIGHTLINE dataset, got geometry {view.geometry.value!r}")
    cols = ["sightline_id", "ra", "dec", "z", "source_row", "wavelength", "delta", "weight"]
    t = view.table(columns=cols, with_position=True, z=z, cone=cone, patch=patch)
    if t.num_rows == 0:
        raise _empty("selection")
    ra, dec, zz = (_f64(t, c) for c in ("ra", "dec", "z"))
    t = _filter(t, np.isfinite(ra) & np.isfinite(dec) & np.isfinite(zz))
    if t.num_rows == 0:
        raise _empty("dropping non-finite ra/dec/z")
    ra, dec = _f64(t, "ra"), _f64(t, "dec")
    window = _window_for(ra, dec, mask)
    t = _filter(t, window.contains(ra, dec))
    if t.num_rows == 0:
        raise _empty("applying the window")
    ra, dec = _f64(t, "ra"), _f64(t, "dec")
    reg = _regions_arg(window, regions, seed)
    rid = reg.assign(ra, dec)
    t = _filter(t, rid >= 0)
    if t.num_rows == 0:
        raise _empty("region assignment")
    ra, dec = _f64(t, "ra"), _f64(t, "dec")
    n = t.num_rows
    catalog = pa.table({
        "sightline_id": t["sightline_id"], "ra": ra, "dec": dec, "z": _f64(t, "z"),
        "weight": np.ones(n), "region_id": reg.assign(ra, dec).astype(np.int32),
        "source_row": t["source_row"], "partition": t["partition"], "offset": t["offset"],
        "wavelength": t["wavelength"], "delta": t["delta"], "pixel_weight": t["weight"],
    })
    prov = {"dataset": view.manifest.survey.name, "version": list(view.version),
            "z": [float(z[0]), float(z[1])], "window_source": window.source,
            "n": n, "seed": int(seed), "n_regions": reg.n_regions}
    return Sightline(catalog=catalog, window=window, provenance=prov)
