"""Multi-order MOC HEALPix → fixed-NSIDE rasteriser.

GW LIGO/Virgo sky-localisation FITS files (BAYESTAR / LALInference)
ship as **multi-order** HEALPix (NUNIQ-indexed). The downstream
:func:`oneuniverse.data.subobject_map.build_subobject_links_to_map`
expects a **fixed-NSIDE** numpy array; this module bridges the two
formats.

`mocpy` is an optional dependency. Importing this module without
`mocpy` installed succeeds; calling :func:`rasterise_moc_to_healpix`
raises an actionable :class:`ImportError`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import healpy as hp
import numpy as np


def rasterise_moc_to_healpix(
    moc_path: Union[str, Path],
    *,
    nside: int,
    nest: bool = True,
) -> np.ndarray:
    """Read a multi-order MOC HEALPix file from ``moc_path`` and
    rasterise it to a fixed-NSIDE ``float32`` array of length
    ``12 * nside²``.

    Cells inside the MOC are set to ``1.0`` (uniform within-MOC
    weight); cells outside are ``0.0``. For a probability-map MOC
    (NUNIQ → PROB), multiply the returned array elementwise by the
    underlying probability extracted via ``mocpy.MOC.serialize``;
    the helper here only honours the coverage geometry.

    Parameters
    ----------
    moc_path
        Path to a FITS file readable by :class:`mocpy.MOC`.
    nside
        Output HEALPix NSIDE (power of two).
    nest
        Output ordering; default ``True`` to match the
        :func:`build_subobject_links_to_map` convention.

    Raises
    ------
    ImportError
        If `mocpy` is not installed in the current environment.
    """
    try:
        import mocpy as _mocpy  # type: ignore[import]
        if _mocpy is None:
            raise ImportError("mocpy is None (monkeypatched out)")
    except (ImportError, TypeError):
        raise ImportError(
            "rasterise_moc_to_healpix requires the optional `mocpy` "
            "dependency. Install with `pip install mocpy>=0.13` or "
            "use the dev extra: `pip install .[dev]`."
        ) from None

    moc = _mocpy.MOC.from_fits(str(moc_path))
    npix = hp.nside2npix(nside)
    pix = np.arange(npix)
    lon, lat = hp.pix2ang(nside, pix, nest=nest, lonlat=True)
    from astropy import units as u

    inside = moc.contains_lonlat(lon * u.deg, lat * u.deg)
    return np.asarray(inside, dtype=np.float32)
