"""Phase 21 T4 — rasterise a multi-order MOC HEALPix file to fixed NSIDE."""
import sys

import healpy as hp
import numpy as np
import pytest


def test_missing_mocpy_raises_actionable_error(monkeypatch):
    """When mocpy is not installed, the import error should explain
    how to add it.
    """
    monkeypatch.setitem(sys.modules, "mocpy", None)
    if "oneuniverse.data.moc" in sys.modules:
        del sys.modules["oneuniverse.data.moc"]
    from oneuniverse.data import moc as mocmod  # noqa: F401
    with pytest.raises(ImportError, match="mocpy"):
        mocmod.rasterise_moc_to_healpix("dummy.fits", nside=32)


def test_rasterise_fixed_nside_circle(tmp_path):
    """A 1-deg radius MOC around (RA=10, Dec=0) at order 7 should map
    onto >0 pixels at NSIDE=32 (NEST) and 0 pixels at the antipode.
    """
    mocpy = pytest.importorskip("mocpy")
    from astropy import units as u

    if "oneuniverse.data.moc" in sys.modules:
        del sys.modules["oneuniverse.data.moc"]
    from oneuniverse.data.moc import rasterise_moc_to_healpix

    moc = mocpy.MOC.from_cone(
        lon=10 * u.deg, lat=0 * u.deg, radius=1 * u.deg, max_depth=7,
    )
    moc_file = tmp_path / "circle.fits"
    moc.write(str(moc_file))

    nside = 32
    arr = rasterise_moc_to_healpix(moc_file, nside=nside, nest=True)
    npix = hp.nside2npix(nside)
    assert arr.shape == (npix,)
    assert arr.sum() > 0
    centre_pix = hp.ang2pix(nside, 10.0, 0.0, nest=True, lonlat=True)
    anti_pix = hp.ang2pix(nside, 190.0, 0.0, nest=True, lonlat=True)
    assert arr[centre_pix] > 0
    assert arr[anti_pix] == 0
