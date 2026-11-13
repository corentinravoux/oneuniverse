"""The measurement spine: OUF view -> Product (or Sightline).

``build_pointset`` runs select -> clean -> quality -> weight -> window ->
regions -> n(z) -> randoms -> provenance in one pass. Every stage that leaves
no row raises ValidationError ("empty ...") before randoms are constructed.
Nothing here depends on a cosmology.
"""
from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Sequence
from typing import Literal

import healpy as hp
import numpy as np
import pyarrow as pa

from oneuniverse._core.errors import FormatError, ValidationError
from oneuniverse._core.healpix import cone_pixels, patch_pixels, radec_to_pix
import oneuniverse
from oneuniverse.combine.weights import QualityMaskWeight, Weight
from oneuniverse.data.schema import Geometry
from oneuniverse.data.view import DatasetView
from oneuniverse.measure.atoms import Nz, Regions, Window
from oneuniverse.measure.product import CATALOG_CORE, RANDOMS_CORE, Product, Sightline

_WINDOW_NSIDE = 256     # footprint fallback resolution
_REGION_NSIDE = 64      # upper bound of the jackknife-region resolution
_MIN_POINTS_PER_PIXEL = 10
_MIN_WINDOW_NSIDE = 8


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


def _footprint(ra: np.ndarray, dec: np.ndarray) -> tuple[Window, dict]:
    """Footprint window at the largest nside <= 256 (floor 8) holding >= 10 points
    per occupied pixel on average."""
    pix = np.unique(radec_to_pix(_WINDOW_NSIDE, ra, dec, nest=True))
    nside = _WINDOW_NSIDE
    while True:
        occ = len(np.unique(pix >> (2 * int(np.log2(_WINDOW_NSIDE // nside)))))
        if len(ra) / occ >= _MIN_POINTS_PER_PIXEL or nside <= _MIN_WINDOW_NSIDE:
            break
        nside //= 2
    return (Window.footprint(ra, dec, nside=nside),
            {"window_nside": nside, "points_per_pixel": float(len(ra) / occ)})


def _window_for(ra: np.ndarray, dec: np.ndarray, mask: Window | None, cone=None,
                patch=None) -> tuple[Window, dict]:
    if mask is None:
        return _footprint(ra, dec)
    if not isinstance(mask, Window):
        raise ValidationError(f"mask must be a Window, got {type(mask).__name__}")
    info = {"window_nside": mask.nside}
    if cone is None and patch is None:
        return mask, info
    cover = np.zeros(12 * mask.nside ** 2, bool)
    if cone is not None:
        cover[cone_pixels(mask.nside, cone.ra, cone.dec, cone.radius, nest=mask.nest)] = True
        if patch is not None:
            both = np.zeros_like(cover)
            both[patch_pixels(mask.nside, patch.ra_min, patch.ra_max, patch.dec_min,
                              patch.dec_max, nest=mask.nest)] = True
            cover &= both
    else:
        cover[patch_pixels(mask.nside, patch.ra_min, patch.ra_max, patch.dec_min,
                           patch.dec_max, nest=mask.nest)] = True
    win = Window(mask.nside, mask.nest, np.where(cover, mask.mask, 0.0), "mask", mask.systematics)
    if not (win.mask > 0).any():
        raise _empty("restricting the mask to the cone/patch")
    return win, info


def _in_selection(ra: np.ndarray, dec: np.ndarray, cone, patch) -> np.ndarray:
    ok = np.ones(len(ra), bool)
    if cone is not None:
        ok &= cone.contains(ra, dec)
    if patch is not None:
        ok &= patch.contains(ra, dec)
    return ok


def _uniform_in_pixels(nside: int, pix: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """One uniform (ra, dec) in degrees inside each NEST pixel of *pix*.

    Rejection: uniform on the sphere in a cap around the pixel centre whose
    radius is the maximal pixel radius; keep points that fall in the pixel.
    """
    r = hp.max_pixrad(nside) * 1.0001
    cos_r = np.cos(r)
    th0, ph0 = hp.pix2ang(nside, pix, nest=True)
    ra = np.empty(len(pix)); dec = np.empty(len(pix))
    todo = np.arange(len(pix))
    while len(todo):
        m = len(todo)
        ct = 1.0 - rng.random(m) * (1.0 - cos_r)
        st = np.sqrt(1.0 - ct * ct)
        ph = rng.random(m) * 2 * np.pi
        x, y, zc = st * np.cos(ph), st * np.sin(ph), ct
        t, p = th0[todo], ph0[todo]
        # rotate local frame (z toward the centre) by R_z(p) R_y(t)
        xr = np.cos(t) * x + np.sin(t) * zc
        zr = -np.sin(t) * x + np.cos(t) * zc
        X, Y, Z = np.cos(p) * xr - np.sin(p) * y, np.sin(p) * xr + np.cos(p) * y, zr
        lon, lat = hp.vec2ang(np.stack([X, Y, Z], 1), lonlat=True)
        ok = hp.ang2pix(nside, lon, lat, nest=True, lonlat=True) == pix[todo]
        ra[todo[ok]], dec[todo[ok]] = lon[ok], lat[ok]
        todo = todo[~ok]
    return ra, dec


def generate_randoms(window: Window, nz: Nz, regions: Regions, *, n: int, seed: int,
                     cone=None, patch=None, z: tuple[float, float] | None = None) -> pa.Table:
    """Exactly *n* random points following the window and the (1-D) n(z).

    Pixels are drawn with probability proportional to the mask; the position is
    a uniform point inside a uniformly chosen NEST child at nside*8 (rejection
    in a cap around the child centre); z comes from the inverse CDF of
    ``nz.normalised()`` (uniform inside each bin); weight 1. With *z* = (z0, z1)
    the n(z) is truncated to [z0, z1] and renormalised first (piecewise-constant
    density, partial bins kept in proportion to their width), so no random lies
    outside the data range. Points outside the
    optional exact *cone* / *patch* or outside every region are rejected and
    redrawn until exactly *n* remain. Deterministic for a seed.
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
    lo, hi = nzn.edges[:-1].copy(), nzn.edges[1:].copy()
    if z is not None:
        lo, hi = np.maximum(lo, float(z[0])), np.minimum(hi, float(z[1]))
    mass = nzn.counts / np.diff(nzn.edges) * np.maximum(hi - lo, 0.0)
    if not mass.sum() > 0:
        raise ValidationError("the n(z) has no mass inside the requested z range")
    cdf_z = np.cumsum(mass)
    cdf_z /= cdf_z[-1]
    fine = window.nside * 8
    rng = np.random.default_rng(seed)
    chunks, have = [], 0
    rate = 1.0
    drawn = 0
    for _ in range(200):
        need = int(min(max((n - have) / max(rate, 0.02), 16), 20 * n + 1000))
        pix = np.minimum(np.searchsorted(cdf_pix, rng.random(need), side="right"), len(mask) - 1)
        child = pix * 64 + rng.integers(0, 64, need)
        ra, dec = _uniform_in_pixels(fine, child, rng)
        u = rng.random(need)
        b = np.minimum(np.searchsorted(cdf_z, u, side="right"), len(lo) - 1)
        lo_cdf = np.where(b > 0, cdf_z[np.maximum(b - 1, 0)], 0.0)
        frac = (u - lo_cdf) / np.maximum(cdf_z[b] - lo_cdf, 1e-300)
        zz = lo[b] + np.clip(frac, 0.0, 1.0) * (hi[b] - lo[b])
        rid = regions.assign(ra, dec)
        ok = (rid >= 0) & _in_selection(ra, dec, cone, patch)
        chunks.append((ra[ok], dec[ok], zz[ok], rid[ok]))
        have += int(ok.sum())
        drawn += need
        rate = max(have / drawn, 1e-3)
        if have >= n:
            break
    else:
        raise ValidationError("could not draw enough randoms inside the regions")
    ra, dec, zr, rid = (np.concatenate([c[i] for c in chunks])[:n] for i in range(4))
    return pa.table({"ra": ra, "dec": dec, "z": zr, "weight": np.ones(n),
                     "region_id": rid.astype(np.int32)}).select(list(RANDOMS_CORE))


def randoms_from_view(view: DatasetView, *, z: tuple[float, float], window: Window,
                      regions: Regions, n_max: int | None = None, cone=None,
                      patch=None, seed: int = 0) -> pa.Table:
    """Randoms ingested from a view: rows in *z* and inside *window*.

    Region ids are assigned from *regions* (rows outside are dropped); the
    ``weight`` column is used when present, else 1. *n_max* subsamples
    deterministically (``default_rng(seed)``, row order kept). Optional exact *cone* /
    *patch* cut the randoms too. No positions are carried.
    """
    cols = ["ra", "dec", "z"] + (["weight"] if "weight" in view.columns else [])
    t = view.table(columns=cols, z=z, cone=cone, patch=patch)
    if t.num_rows == 0:
        raise _empty("selecting the randoms")
    ra, dec, zz = (_f64(t, c) for c in ("ra", "dec", "z"))
    w = _f64(t, "weight") if "weight" in cols else np.ones(t.num_rows)
    rid = regions.assign(ra, dec)
    keep = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(zz) & np.isfinite(w) \
        & window.contains(ra, dec) & (rid >= 0) & _in_selection(ra, dec, cone, patch)
    ra, dec, zz, w, rid = ra[keep], dec[keep], zz[keep], w[keep], rid[keep]
    if len(ra) == 0:
        raise _empty("applying the window to the randoms")
    if n_max is not None and len(ra) > int(n_max):
        sel = np.sort(np.random.default_rng(int(seed)).choice(len(ra), int(n_max), replace=False))
        ra, dec, zz, w, rid = ra[sel], dec[sel], zz[sel], w[sel], rid[sel]
    return pa.table({"ra": ra, "dec": dec, "z": zz, "weight": w,
                     "region_id": rid.astype(np.int32)}).select(list(RANDOMS_CORE))


def _build_pointset(view: DatasetView, *, z: tuple[float, float], weight: Weight | None = None,
                   nz_edges: Sequence[float], mask: Window | None = None,
                   regions: Regions | int = 20,
                   randoms: Literal["generate", "none"] | DatasetView = "generate",
                   random_factor: int = 10, keep_columns: Sequence[str] = (),
                   quality: tuple[str, str, float] | None = None, cone=None, patch=None,
                   seed: int = 0) -> tuple[Product, Regions]:
    """Build a point-set Product from an OUF view.

    Steps: scan (with positions) -> drop non-finite ra/dec/z -> quality cut ->
    weight (None means 1) -> window -> drop rows outside the window -> regions ->
    weighted spec_hist n(z) on *nz_edges* -> randoms -> provenance.

    Window: the given *mask*; with a *cone* / *patch* it is restricted to the
    selection (mask times the inclusive pixel cover), and randoms are cut with
    the exact cone / patch. Without a mask the window is the footprint of the
    selected points at the largest nside <= 256 (floor 8) with >= 10 points per
    occupied pixel (recorded as provenance["window_nside"] and
    ["points_per_pixel"]). A footprint window is only a fallback: sparse data
    leave holes and the edge is ragged, so pass a survey mask for science.

    An int *regions* builds ``Regions.from_window(window, n, nside=min(64,
    window.nside), seed=seed)``: the region resolution is capped at nside 64,
    and n is clamped to the number of masked region pixels so small windows
    still work (the clamped count is recorded as provenance["n_regions"]).
    Rows that fall in no region are dropped. randoms="generate" draws
    ``random_factor * n`` points (see :func:`generate_randoms`); a DatasetView
    is ingested (see :func:`randoms_from_view`), capped at ``random_factor * n``
    by a deterministic subsample; "none" gives no randoms.
    The catalog holds CATALOG_CORE then *keep_columns*.

    This private form returns ``(Product, Regions)``; :func:`build_pointset` is the
    public Product-only view of it (the presets need the Regions object).
    """
    if isinstance(keep_columns, str):
        raise ValidationError("keep_columns must be a sequence of names, not a str")
    keep_columns = [str(c) for c in keep_columns]
    if not (isinstance(randoms, DatasetView) or randoms in ("generate", "none")):
        raise ValidationError(f"randoms must be 'generate', 'none' or a DatasetView, got {randoms!r}")
    if randoms == "generate" and int(random_factor) < 1:
        raise ValidationError("random_factor must be >= 1")
    ne = np.asarray(nz_edges, np.float64)
    Nz(ne, np.zeros(max(len(ne) - 1, 1)) if ne.ndim == 1 else np.zeros(1), "spec_hist")
    if not (ne[0] <= float(z[0]) and ne[-1] >= float(z[1])):
        raise ValidationError(
            f"nz_edges [{ne[0]}, {ne[-1]}] must cover z = ({z[0]}, {z[1]})")
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
    window, winfo = _window_for(ra, dec, mask, cone, patch)
    inside = window.contains(ra, dec)
    t = _filter(t, inside)
    if t.num_rows == 0:
        raise _empty("applying the window")
    ra, dec = _f64(t, "ra"), _f64(t, "dec")
    reg = _regions_arg(window, regions, seed)
    rid = reg.assign(ra, dec)
    n_dropped = int((rid < 0).sum())
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
        rnd = generate_randoms(window, nz, reg, n=int(random_factor) * catalog.num_rows, seed=seed,
                               cone=cone, patch=patch, z=z)
        rsrc = "generated"
    elif randoms == "none":
        rnd, rsrc = None, "none"
    else:
        rnd = randoms_from_view(randoms, z=z, window=window, regions=reg, cone=cone, patch=patch,
                                n_max=int(random_factor) * catalog.num_rows, seed=seed)
        rsrc = "ingested"
    prov = {
        "dataset": view.manifest.survey.name, "version": list(view.version),
        "z": [float(z[0]), float(z[1])],
        "weight": weight.describe() if weight is not None else "none",
        "window_source": window.source, "randoms_source": rsrc, "nz_method": nz.method,
        "n": catalog.num_rows, "seed": int(seed), "n_regions": reg.n_regions,
        "quality": list(quality) if quality is not None else None,
        "keep_columns": keep_columns, **winfo,
        "cone": dataclasses.asdict(cone) if cone is not None else None,
        "patch": dataclasses.asdict(patch) if patch is not None else None,
        "random_factor": int(random_factor),
        "nz_edges": [float(e) for e in ne],
        "mask": ({"nside": mask.nside, "nest": mask.nest,
                  "sha256": hashlib.sha256(np.ascontiguousarray(mask.mask).tobytes()).hexdigest()}
                 if mask is not None else None),
        "oneuniverse_version": oneuniverse.__version__,
        "randoms_nz": "weighted",
        "n_dropped_outside_regions": n_dropped,
    }
    if isinstance(randoms, DatasetView):
        prov["randoms_dataset"] = randoms.manifest.survey.name
        prov["randoms_version"] = list(randoms.version)
    return Product(catalog=catalog, randoms=rnd, window=window, nz=nz, provenance=prov), reg


def build_pointset(view: DatasetView, *, z: tuple[float, float], weight: Weight | None = None,
                  nz_edges: Sequence[float], mask: Window | None = None,
                  regions: Regions | int = 20,
                  randoms: Literal["generate", "none"] | DatasetView = "generate",
                  random_factor: int = 10, keep_columns: Sequence[str] = (),
                  quality: tuple[str, str, float] | None = None, cone=None, patch=None,
                  seed: int = 0) -> Product:
    """Build a point-set Product from an OUF view (select, clean, weight, window,
    regions, n(z), randoms, provenance).

    *nz_edges* must cover *z* (``nz_edges[0] <= z[0]`` and ``nz_edges[-1] >=
    z[1]``), else ValidationError. The n(z) is the weighted histogram of the
    data on *nz_edges*; generated randoms follow that weighted n(z) truncated to
    *z* and carry weight 1, so do not re-apply the same weights to the randoms.
    Without *mask* the window is a footprint fallback; an int *regions* builds
    k-means jackknife regions (capped at nside 64). Rows falling outside a given
    Regions' pixels are dropped and counted in
    ``provenance["n_dropped_outside_regions"]``. The provenance records the
    selection (cone, patch, z, quality), weights, mask hash, nz_edges, randoms
    settings, seed and the oneuniverse version, enough to rebuild the product.

    Returns the Product (see ``_build_pointset`` for the full step list).
    """
    return _build_pointset(view, z=z, weight=weight, nz_edges=nz_edges, mask=mask,
                           regions=regions, randoms=randoms, random_factor=random_factor,
                           keep_columns=keep_columns, quality=quality, cone=cone, patch=patch,
                           seed=seed)[0]


def build_sightlines(view: DatasetView, *, z: tuple[float, float],
                     regions: Regions | int = 20, mask: Window | None = None,
                     cone=None, patch=None, seed: int = 0) -> Sightline:
    """Build a Sightline product from a SIGHTLINE-geometry view.

    Window and regions follow :func:`build_pointset` (adaptive footprint
    fallback, regions capped at nside 64). The OUF list column ``weight``
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
    window, winfo = _window_for(ra, dec, mask, cone, patch)
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
            "n": n, "seed": int(seed), "n_regions": reg.n_regions, **winfo}
    return Sightline(catalog=catalog, window=window, provenance=prov)
