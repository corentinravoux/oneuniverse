"""The single HEALPix helper layer (NEST by default). All outputs are int32."""
from __future__ import annotations

import healpy as hp
import numpy as np
from numpy.typing import ArrayLike

from oneuniverse._core.errors import ValidationError
from oneuniverse._core.geometry import Cone, SkyPatch

NSIDE_FINE = 32

_MIN_RA_WIDTH = 1e-3  # degrees; thinner RA pieces are widened to this
_ZERO_WIDTH = 1e-12   # degrees; a piece this thin is degenerate (split artefact)
MAX_NSIDE = 8192      # 12 * nside**2 must fit in int32
_MAX_SLICES = 720


def _pad_dec(w_rad: float, dec_deg: float) -> float:
    """Corner Dec whose great-circle arc, over RA width *w_rad*, dips to *dec_deg*.

    A great circle through two points at latitude phi separated by w in
    longitude reaches latitude atan(tan(phi) / cos(w/2)) at its midpoint
    (exact). Choosing the corner latitude lo = atan(tan(dec) * cos(w/2))
    makes the midpoint land exactly on *dec_deg*, so the slice covers its
    whole Dec band. Finite for every |dec| < 90 (no divergence near poles).
    """
    return float(np.degrees(np.arctan(np.tan(np.radians(dec_deg)) * np.cos(w_rad / 2.0))))


def _check_nside(nside: int) -> None:
    """Shared validation: power of 2 with 1 <= nside <= 8192."""
    if not hp.isnsideok(nside, nest=True) or not 1 <= nside <= MAX_NSIDE:
        raise ValidationError(
            f"nside must be a power of 2 in [1, {MAX_NSIDE}], got {nside}")


def _finish(pix: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(pix)).astype(np.int32)


def radec_to_pix(nside: int, ra: ArrayLike, dec: ArrayLike, *, nest: bool = True) -> np.ndarray:
    """HEALPix pixel of (ra, dec) in degrees."""
    _check_nside(nside)
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    if not (np.all(np.isfinite(ra)) and np.all(np.isfinite(dec))):
        raise ValidationError("ra and dec must be finite")
    if np.any(np.abs(dec) > 90.0):
        raise ValidationError("|dec| must be <= 90 degrees")
    pix = hp.ang2pix(nside, ra, dec, nest=nest, lonlat=True)
    return np.asarray(pix).astype(np.int32)


def coarsen(pix: ArrayLike, nside_from: int, nside_to: int) -> np.ndarray:
    """Degrade NEST pixels from *nside_from* to *nside_to* (right shift)."""
    _check_nside(nside_from)
    _check_nside(nside_to)
    if nside_to > nside_from:
        raise ValidationError(f"cannot coarsen to a finer grid: {nside_from} -> {nside_to}")
    shift = 2 * (int(np.log2(nside_from)) - int(np.log2(nside_to)))
    return (np.asarray(pix, dtype=np.int64) >> shift).astype(np.int32)


def cone_pixels(nside: int, ra: float, dec: float, radius_deg: float, *,
                nest: bool = True) -> np.ndarray:
    """Inclusive pixel cover of a cone (degrees)."""
    _check_nside(nside)
    cone = Cone(ra, dec, radius_deg)  # validates finite ra/dec/radius, radius > 0
    vec = hp.ang2vec(cone.ra, cone.dec, lonlat=True)
    return _finish(hp.query_disc(nside, vec, np.radians(cone.radius), inclusive=True, nest=nest))


def _strip_cover(nside: int, dec_min: float, dec_max: float, nest: bool) -> np.ndarray:
    theta1 = np.radians(90.0 - dec_max)
    theta2 = np.radians(90.0 - dec_min)
    return hp.query_strip(nside, theta1, theta2, inclusive=True, nest=nest)


def _piece_cover(nside: int, ra0: float, ra1: float, dec_min: float, dec_max: float,
                 nest: bool) -> np.ndarray:
    """Cover of a non-wrapping RA piece [ra0, ra1].

    Pieces wider than 180 deg, or whose Dec band comes within one pixel
    resolution of a pole, use the full-RA ``query_strip`` cover of the Dec band
    (a superset, still a valid cover). Pieces thinner than ``_MIN_RA_WIDTH``
    are widened to that width (clamped to [0, 360]). Otherwise the piece is covered by a union of narrow convex polygons.
    """
    width = ra1 - ra0
    pole_margin = 90.0 - float(np.degrees(hp.nside2resol(nside)))
    if width > 180.0 or dec_max > pole_margin or dec_min < -pole_margin:
        return _strip_cover(nside, dec_min, dec_max, nest)
    if width < _MIN_RA_WIDTH:
        centre = 0.5 * (ra0 + ra1)
        ra0 = min(max(centre - _MIN_RA_WIDTH / 2, 0.0), 360.0 - _MIN_RA_WIDTH)
        ra1 = ra0 + _MIN_RA_WIDTH
        width = ra1 - ra0
    # Lat/lon boxes are not convex polygons once sampled along constant-Dec
    # edges, so the piece is cut into narrow slices, each a convex 4-corner
    # polygon. A great-circle edge between two points at the same Dec bows
    # poleward (see _pad_dec); the equatorward corners are pushed out
    # accordingly so the slice still covers its Dec band.
    step = np.degrees(hp.nside2resol(nside)) / 2.0
    n = min(_MAX_SLICES, max(1, int(np.ceil(width / step))))
    edges = np.linspace(ra0, ra1, n + 1)
    w = np.radians(width / n)
    covers = []
    for a, b in zip(edges[:-1], edges[1:]):
        lo = _pad_dec(w, dec_min) if dec_min > 0 else dec_min
        hi = _pad_dec(w, dec_max) if dec_max < 0 else dec_max
        vertices = hp.ang2vec([a, b, b, a], [lo, lo, hi, hi], lonlat=True)
        covers.append(hp.query_polygon(nside, vertices, inclusive=True, nest=nest))
    return np.concatenate(covers)


def patch_pixels(nside: int, ra_min: float, ra_max: float, dec_min: float, dec_max: float, *,
                 nest: bool = True) -> np.ndarray:
    """Inclusive pixel cover of an RA/Dec patch (degrees); ``ra_min > ra_max`` wraps."""
    _check_nside(nside)
    SkyPatch(ra_min, ra_max, dec_min, dec_max)  # validates the arguments
    if ra_min > ra_max:
        pieces = [(ra_min, 360.0), (0.0, ra_max)]
        # a degenerate piece from the split (e.g. ra_max == 0) adds nothing:
        # RA 0 and 360 are the same meridian, covered by the other piece.
        pieces = [pc for pc in pieces if pc[1] - pc[0] > _ZERO_WIDTH] or pieces
    else:
        pieces = [(ra_min, ra_max)]
    covers = [_piece_cover(nside, a, b, dec_min, dec_max, nest) for a, b in pieces]
    return _finish(np.concatenate(covers))
