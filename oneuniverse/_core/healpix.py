"""The single HEALPix helper layer (NEST by default). All outputs are int32."""
from __future__ import annotations

import healpy as hp
import numpy as np
from numpy.typing import ArrayLike

NSIDE_FINE = 32

_POLE_EPS = 1e-9
_MAX_SLICES = 720


def _bulge_deg(w_rad: float, dec_deg: float) -> float:
    return float(np.degrees(w_rad ** 2 / 8.0 * abs(np.tan(np.radians(dec_deg)))))


def _check_nside(nside: int) -> None:
    if not hp.isnsideok(nside, nest=True):
        raise ValueError(f"nside must be a power of 2, got {nside}")


def _finish(pix: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(pix)).astype(np.int32)


def radec_to_pix(nside: int, ra: ArrayLike, dec: ArrayLike, *, nest: bool = True) -> np.ndarray:
    """HEALPix pixel of (ra, dec) in degrees."""
    pix = hp.ang2pix(nside, np.asarray(ra, dtype=np.float64),
                     np.asarray(dec, dtype=np.float64), nest=nest, lonlat=True)
    return np.asarray(pix).astype(np.int32)


def coarsen(pix: ArrayLike, nside_from: int, nside_to: int) -> np.ndarray:
    """Degrade NEST pixels from *nside_from* to *nside_to* (right shift)."""
    _check_nside(nside_from)
    _check_nside(nside_to)
    if nside_to > nside_from:
        raise ValueError(f"cannot coarsen to a finer grid: {nside_from} -> {nside_to}")
    shift = 2 * (int(np.log2(nside_from)) - int(np.log2(nside_to)))
    return (np.asarray(pix, dtype=np.int64) >> shift).astype(np.int32)


def cone_pixels(nside: int, ra: float, dec: float, radius_deg: float, *,
                nest: bool = True) -> np.ndarray:
    """Inclusive pixel cover of a cone (degrees)."""
    vec = hp.ang2vec(ra, dec, lonlat=True)
    return _finish(hp.query_disc(nside, vec, np.radians(radius_deg), inclusive=True, nest=nest))


def _strip_cover(nside: int, dec_min: float, dec_max: float, nest: bool) -> np.ndarray:
    theta1 = np.radians(90.0 - dec_max)
    theta2 = np.radians(90.0 - dec_min)
    return hp.query_strip(nside, theta1, theta2, inclusive=True, nest=nest)


def _piece_cover(nside: int, ra0: float, ra1: float, dec_min: float, dec_max: float,
                 nest: bool) -> np.ndarray:
    """Cover of a non-wrapping RA piece [ra0, ra1].

    Pieces wider than 180 deg, degenerate, or touching a pole use the full-RA
    ``query_strip`` cover of the Dec band (a superset, still a valid cover).
    Otherwise the piece is covered by a union of narrow convex polygons.
    """
    width = ra1 - ra0
    if (width > 180.0 or width <= 0.0
            or dec_max >= 90.0 - _POLE_EPS or dec_min <= -90.0 + _POLE_EPS):
        return _strip_cover(nside, dec_min, dec_max, nest)
    # Lat/lon boxes are not convex polygons once sampled along constant-Dec
    # edges, so the piece is cut into narrow slices, each a convex 4-corner
    # polygon. A great-circle edge between two points at the same Dec bows
    # poleward by ~ (w^2 / 8) tan(dec); the equatorward corners are pushed out
    # by that amount so the slice still covers its Dec band.
    step = np.degrees(hp.nside2resol(nside)) / 2.0
    n = min(_MAX_SLICES, max(1, int(np.ceil(width / step))))
    edges = np.linspace(ra0, ra1, n + 1)
    w = np.radians(width / n)
    covers = []
    for a, b in zip(edges[:-1], edges[1:]):
        lo = dec_min - _bulge_deg(w, dec_min) if dec_min > 0 else dec_min
        hi = dec_max + _bulge_deg(w, dec_max) if dec_max < 0 else dec_max
        lo, hi = max(lo, -90.0), min(hi, 90.0)
        vertices = hp.ang2vec([a, b, b, a], [lo, lo, hi, hi], lonlat=True)
        covers.append(hp.query_polygon(nside, vertices, inclusive=True, nest=nest))
    return np.concatenate(covers)


def patch_pixels(nside: int, ra_min: float, ra_max: float, dec_min: float, dec_max: float, *,
                 nest: bool = True) -> np.ndarray:
    """Inclusive pixel cover of an RA/Dec patch (degrees); ``ra_min > ra_max`` wraps."""
    if ra_min > ra_max:
        pieces = [(ra_min, 360.0), (0.0, ra_max)]
    else:
        pieces = [(ra_min, ra_max)]
    covers = [_piece_cover(nside, a, b, dec_min, dec_max, nest) for a, b in pieces]
    return _finish(np.concatenate(covers))
