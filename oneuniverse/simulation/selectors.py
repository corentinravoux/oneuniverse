"""Spatial selectors for partial-access reads.

These are the spatial members of the OUF-Sim selector taxonomy. The
view's reader takes one of these (or a HEALPix tile list / octree node
id, added with their backends) to materialise only a sub-region —
never the whole snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Cube:
    """Axis-aligned comoving bounding box (unit-frame length units)."""

    xlo: float
    xhi: float
    ylo: float
    yhi: float
    zlo: float
    zhi: float

    def __post_init__(self) -> None:
        for lo, hi, ax in (
            (self.xlo, self.xhi, "x"),
            (self.ylo, self.yhi, "y"),
            (self.zlo, self.zhi, "z"),
        ):
            if lo > hi:
                raise ValueError(
                    f"Cube: {ax}lo ({lo}) must be <= {ax}hi ({hi})"
                )


@dataclass(frozen=True)
class Cone:
    """Angular cone: centre (lon, lat) in degrees + radius in degrees."""

    lon: float
    lat: float
    radius_deg: float

    def __post_init__(self) -> None:
        if self.radius_deg <= 0.0:
            raise ValueError(
                f"Cone.radius_deg must be > 0, got {self.radius_deg!r}"
            )


@dataclass(frozen=True)
class SkyPatch:
    """Angular rectangle in degrees (lon/lat min/max)."""

    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float

    def __post_init__(self) -> None:
        if self.lon_min > self.lon_max:
            raise ValueError(
                f"SkyPatch: lon_min ({self.lon_min}) must be <= "
                f"lon_max ({self.lon_max})"
            )
        if self.lat_min > self.lat_max:
            raise ValueError(
                f"SkyPatch: lat_min ({self.lat_min}) must be <= "
                f"lat_max ({self.lat_max})"
            )
