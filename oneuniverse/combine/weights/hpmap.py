"""HEALPix map-backed weight primitive.

Given a full-sky HEALPix array (any valid NSIDE, ring or nest), returns
the map value at the pixel containing each object's (ra, dec). Covers
the class of survey weights stored as a map:

* completeness / angular-footprint masks
* imaging-systematic weights (SYSNet, Regressis, linear regressors)
* stellar-density / extinction regressors
* per-band depth maps

No survey-specific knowledge lives here — caller supplies the map.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import healpy as hp
import numpy as np
import pandas as pd

from oneuniverse.combine.weights.base import Weight


class HealpixMapWeight(Weight):
    """Per-object weight ``w_i = map[pix(ra_i, dec_i)]``.

    Parameters
    ----------
    nside
        HEALPix NSIDE of the map. Must satisfy
        ``len(map_array) == 12 * nside**2``.
    map_array
        Full-sky map as a 1-D array of per-pixel values.
    nest
        ``True`` if the map is in NESTED ordering, ``False`` for RING.
    ra_column, dec_column
        DataFrame column names carrying ICRS RA/Dec in degrees.
    nan_fill
        If not ``None``, any NaN pixel hit returns this value instead of
        raising. Use e.g. ``0.0`` to zero out objects falling in
        unsurveyed / masked cells.
    """

    def __init__(
        self,
        nside: int,
        map_array: np.ndarray,
        nest: bool = True,
        ra_column: str = "ra",
        dec_column: str = "dec",
        nan_fill: Optional[float] = None,
        name: str = "hpmap",
    ) -> None:
        map_array = np.asarray(map_array, dtype=np.float64)
        expected = hp.nside2npix(nside)
        if map_array.ndim != 1 or map_array.size != expected:
            raise ValueError(
                f"HealpixMapWeight: map length {map_array.size} does not match "
                f"NSIDE={nside} ({expected} pixels)"
            )
        self.nside = int(nside)
        self.map_array = map_array
        self.nest = bool(nest)
        self.ra_column = ra_column
        self.dec_column = dec_column
        self.nan_fill = nan_fill
        self.name = name

    @classmethod
    def from_fits(
        cls, path: Union[str, Path], nest: bool = True, **kwargs,
    ) -> "HealpixMapWeight":
        """Read a HEALPix FITS map via :func:`healpy.read_map` and wrap it."""
        arr = hp.read_map(str(path), nest=nest)
        nside = hp.npix2nside(arr.size)
        return cls(nside=nside, map_array=arr, nest=nest, **kwargs)

    def compute(self, df: pd.DataFrame) -> np.ndarray:
        ra = df[self.ra_column].to_numpy(dtype=np.float64)
        dec = df[self.dec_column].to_numpy(dtype=np.float64)
        theta = np.radians(90.0 - dec)
        phi = np.radians(ra)
        pix = hp.ang2pix(self.nside, theta, phi, nest=self.nest)
        vals = self.map_array[pix]
        if np.any(~np.isfinite(vals)):
            if self.nan_fill is None:
                raise ValueError(
                    f"HealpixMapWeight({self.name}): NaN/inf in map at "
                    f"{(~np.isfinite(vals)).sum()} object(s); pass nan_fill=... "
                    f"to silently mask them."
                )
            vals = np.where(np.isfinite(vals), vals, self.nan_fill)
        return vals
