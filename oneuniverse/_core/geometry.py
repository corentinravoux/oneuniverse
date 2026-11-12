"""Shared selection geometries: cone, sky patch, redshift range, cube."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from oneuniverse._core.errors import ValidationError


@dataclass(frozen=True, slots=True)
class Cone:
    """Sky cone; ``ra``, ``dec``, ``radius`` in degrees.

    Membership is a closed interval: separation <= radius.
    """

    ra: float
    dec: float
    radius: float

    def __post_init__(self) -> None:
        if not all(np.isfinite(v) for v in (self.ra, self.dec, self.radius)):
            raise ValidationError(
                f"Cone ra, dec, radius must be finite, got "
                f"({self.ra}, {self.dec}, {self.radius})")
        if not self.radius > 0:
            raise ValidationError(f"Cone radius must be > 0, got {self.radius}")
        if not -90.0 <= self.dec <= 90.0:
            raise ValidationError(f"Cone dec must be in [-90, 90], got {self.dec}")

    def contains(self, ra: ArrayLike, dec: ArrayLike) -> np.ndarray:
        """Haversine membership test (float64)."""
        ra = np.radians(np.asarray(ra, dtype=np.float64))
        dec = np.radians(np.asarray(dec, dtype=np.float64))
        ra0, dec0 = np.radians(self.ra), np.radians(self.dec)
        a = (np.sin((dec - dec0) / 2) ** 2
             + np.cos(dec) * np.cos(dec0) * np.sin((ra - ra0) / 2) ** 2)
        sep = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        return sep <= np.radians(self.radius)


@dataclass(frozen=True, slots=True)
class SkyPatch:
    """RA/Dec box in degrees; ``ra_min > ra_max`` means the patch wraps through 0.

    Membership is a closed interval on both axes: ra_min <= ra <= ra_max (with
    RA taken modulo 360) and dec_min <= dec <= dec_max.
    """

    ra_min: float
    ra_max: float
    dec_min: float
    dec_max: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.ra_min <= 360.0 and 0.0 <= self.ra_max <= 360.0):
            raise ValidationError("SkyPatch RA bounds must be in [0, 360]")
        if not (-90.0 <= self.dec_min < self.dec_max <= 90.0):
            raise ValidationError("SkyPatch requires -90 <= dec_min < dec_max <= 90")

    @property
    def wraps(self) -> bool:
        return self.ra_min > self.ra_max

    def contains(self, ra: ArrayLike, dec: ArrayLike) -> np.ndarray:
        ra_in = np.asarray(ra, dtype=np.float64)
        ra = np.mod(ra_in, 360.0)
        dec = np.asarray(dec, dtype=np.float64)
        if self.wraps:
            in_ra = (ra >= self.ra_min) | (ra <= self.ra_max)
        else:
            in_ra = (ra >= self.ra_min) & (ra <= self.ra_max)
            if self.ra_max == 360.0:  # keep the closed upper edge (360 == 0 mod 360)
                in_ra |= ra_in == 360.0
        return in_ra & (dec >= self.dec_min) & (dec <= self.dec_max)


@dataclass(frozen=True, slots=True)
class ZRange:
    """Redshift interval; membership is closed: lo <= z <= hi."""

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if not self.lo < self.hi:
            raise ValidationError(f"ZRange requires lo < hi, got ({self.lo}, {self.hi})")

    def contains(self, z: ArrayLike) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        return (z >= self.lo) & (z <= self.hi)


@dataclass(frozen=True, slots=True)
class Cube:
    """Axis-aligned box; membership is closed: lo <= x <= hi on every axis.

    ``lo`` and ``hi`` are normalised to tuples of 3 floats.
    """

    lo: tuple[float, float, float]
    hi: tuple[float, float, float]

    def __post_init__(self) -> None:
        lo = tuple(float(v) for v in self.lo)
        hi = tuple(float(v) for v in self.hi)
        if len(lo) != 3 or len(hi) != 3:
            raise ValidationError("Cube lo and hi must have 3 components")
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)
        if not all(l < h for l, h in zip(self.lo, self.hi)):
            raise ValidationError(f"Cube requires lo < hi componentwise, got {self.lo}, {self.hi}")

    @property
    def volume(self) -> float:
        return float(np.prod(np.asarray(self.hi, float) - np.asarray(self.lo, float)))

    def contains(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> np.ndarray:
        x, y, z = (np.asarray(a, dtype=np.float64) for a in (x, y, z))
        return (
            (x >= self.lo[0]) & (x <= self.hi[0])
            & (y >= self.lo[1]) & (y <= self.hi[1])
            & (z >= self.lo[2]) & (z <= self.hi[2])
        )


Selection = Cone | SkyPatch
