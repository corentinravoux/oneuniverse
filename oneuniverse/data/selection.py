"""
oneuniverse.data.selection
~~~~~~~~~~~~~~~~~~~~~~~~~~
Composable spatial and redshift selection criteria.

A selection object knows how to produce a boolean mask given a DataFrame
with ``ra``, ``dec``, ``z`` columns. Multiple selections can be combined
in a list (AND logic).

Supported selections
--------------------
Cone        : angular cone on the sky (great-circle distance)
Shell       : redshift range  z ∈ [z_min, z_max]
SkyPatch    : rectangular RA/Dec box (with RA wrap-around support)

Usage
-----
>>> from oneuniverse.data.selection import Cone, Shell
>>> sel = [Cone(ra=185, dec=15, radius=5), Shell(z_min=0.02, z_max=0.08)]
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np


class Selection(abc.ABC):
    """Abstract base for catalog selection criteria."""

    @abc.abstractmethod
    def mask(self, ra: np.ndarray, dec: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Return a boolean mask (True = keep) for the given coordinates.

        Parameters
        ----------
        ra, dec : ndarray, float64, degrees (ICRS)
        z : ndarray, float — best redshift
        """

    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({attrs})"


# ── Concrete selections ───────────────────────────────────────────────────


@dataclass
class Cone(Selection):
    """Angular cone on the sky.

    Parameters
    ----------
    ra : float — center RA in degrees
    dec : float — center Dec in degrees
    radius : float — cone radius in degrees
    """

    ra: float
    dec: float
    radius: float

    def mask(self, ra: np.ndarray, dec: np.ndarray, z: np.ndarray) -> np.ndarray:
        return _angular_separation(self.ra, self.dec, ra, dec) <= self.radius

    def healpix_cells(self, nside: int, nest: bool = True) -> np.ndarray:
        """Return HEALPix cells covering this cone (inclusive)."""
        import healpy as hp
        vec = hp.ang2vec(np.radians(90.0 - self.dec), np.radians(self.ra))
        return hp.query_disc(
            nside, vec, radius=np.radians(self.radius),
            inclusive=True, nest=nest,
        ).astype(np.int64)


@dataclass
class Shell(Selection):
    """Redshift shell: z_min <= z <= z_max.

    Parameters
    ----------
    z_min : float — lower redshift bound (inclusive)
    z_max : float — upper redshift bound (inclusive)
    """

    z_min: float
    z_max: float

    def mask(self, ra: np.ndarray, dec: np.ndarray, z: np.ndarray) -> np.ndarray:
        return (z >= self.z_min) & (z <= self.z_max)


@dataclass
class SkyPatch(Selection):
    """Rectangular RA/Dec box.

    Handles RA wrap-around: if ra_min > ra_max the selection wraps through
    360° (e.g., SkyPatch(ra_min=350, ra_max=10, ...) selects 350-360 + 0-10).

    Parameters
    ----------
    ra_min, ra_max : float — RA bounds in degrees
    dec_min, dec_max : float — Dec bounds in degrees
    """

    ra_min: float
    ra_max: float
    dec_min: float
    dec_max: float

    def mask(self, ra: np.ndarray, dec: np.ndarray, z: np.ndarray) -> np.ndarray:
        dec_ok = (dec >= self.dec_min) & (dec <= self.dec_max)
        if self.ra_min <= self.ra_max:
            ra_ok = (ra >= self.ra_min) & (ra <= self.ra_max)
        else:
            # Wrap-around: e.g. 350 → 10 means (ra >= 350) | (ra <= 10)
            ra_ok = (ra >= self.ra_min) | (ra <= self.ra_max)
        return dec_ok & ra_ok

    def healpix_cells(self, nside: int, nest: bool = True) -> np.ndarray:
        """Return HEALPix cells covering this rectangular patch (inclusive).

        Handles RA wrap-around by splitting into two polygons when
        ``ra_min > ra_max``.
        """
        import healpy as hp

        def _polygon(ra_lo: float, ra_hi: float) -> np.ndarray:
            corners = [
                (ra_lo, self.dec_min),
                (ra_hi, self.dec_min),
                (ra_hi, self.dec_max),
                (ra_lo, self.dec_max),
            ]
            vecs = np.array([
                hp.ang2vec(np.radians(90.0 - d), np.radians(r))
                for r, d in corners
            ])
            return hp.query_polygon(
                nside, vecs, inclusive=True, nest=nest,
            )

        if self.ra_min <= self.ra_max:
            cells = _polygon(self.ra_min, self.ra_max)
        else:
            cells = np.concatenate([
                _polygon(self.ra_min, 360.0),
                _polygon(0.0, self.ra_max),
            ])
        return np.unique(cells).astype(np.int64)


# ── Utilities ─────────────────────────────────────────────────────────────


def apply_selections(
    ra: np.ndarray,
    dec: np.ndarray,
    z: np.ndarray,
    selections,
) -> np.ndarray:
    """Apply a list of selections (AND logic) and return a boolean mask.

    Parameters
    ----------
    selections : Selection or list[Selection] or None
        If None, returns all-True mask.
    """
    if selections is None:
        return np.ones(len(ra), dtype=bool)

    if isinstance(selections, Selection):
        selections = [selections]

    mask = np.ones(len(ra), dtype=bool)
    for sel in selections:
        mask &= sel.mask(ra, dec, z)
    return mask


def _angular_separation(
    ra1: float,
    dec1: float,
    ra2: np.ndarray,
    dec2: np.ndarray,
) -> np.ndarray:
    """Great-circle angular separation in degrees (Vincenty formula).

    Parameters
    ----------
    ra1, dec1 : float — reference point in degrees
    ra2, dec2 : ndarray — catalog positions in degrees

    Returns
    -------
    ndarray — angular separation in degrees
    """
    ra1_r = np.radians(ra1)
    dec1_r = np.radians(dec1)
    ra2_r = np.radians(ra2)
    dec2_r = np.radians(dec2)

    dra = ra2_r - ra1_r
    cos_dec1 = np.cos(dec1_r)
    sin_dec1 = np.sin(dec1_r)
    cos_dec2 = np.cos(dec2_r)
    sin_dec2 = np.sin(dec2_r)

    # Vincenty — numerically stable for all separations
    num = np.sqrt(
        (cos_dec2 * np.sin(dra)) ** 2
        + (cos_dec1 * sin_dec2 - sin_dec1 * cos_dec2 * np.cos(dra)) ** 2
    )
    den = sin_dec1 * sin_dec2 + cos_dec1 * cos_dec2 * np.cos(dra)
    return np.degrees(np.arctan2(num, den))
