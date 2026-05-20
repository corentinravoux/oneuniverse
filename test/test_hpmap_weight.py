"""Tests for HealpixMapWeight (ring/nest lookup, NaN handling, FITS I/O)."""
from __future__ import annotations

import sys
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.healpix_maps import (  # noqa: E402
    make_footprint_mask, make_systematic_map,
)

from oneuniverse.combine.weights.hpmap import HealpixMapWeight  # noqa: E402


def _df(ra, dec):
    return pd.DataFrame({"ra": np.asarray(ra), "dec": np.asarray(dec)})


def test_systematic_map_weight_matches_direct_lookup():
    nside = 32
    m = make_systematic_map(nside, seed=0)
    w = HealpixMapWeight(nside=nside, map_array=m, nest=True)
    ra = np.array([12.0, 180.0, 350.0])
    dec = np.array([0.0, 10.0, -25.0])
    got = w(_df(ra, dec))
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    expected = m[hp.ang2pix(nside, theta, phi, nest=True)]
    np.testing.assert_allclose(got, expected)


def test_footprint_mask_zeroes_outside_band():
    nside = 32
    m = make_footprint_mask(nside, seed=0)
    w = HealpixMapWeight(nside=nside, map_array=m, nest=True)
    df = _df([100.0, 100.0], [-70.0, 20.0])
    got = w(df)
    assert got[0] == 0.0 and got[1] == 1.0


def test_ring_order_map_accepted():
    nside = 32
    m_nest = make_systematic_map(nside, seed=1)
    m_ring = hp.reorder(m_nest, n2r=True)
    w_ring = HealpixMapWeight(nside=nside, map_array=m_ring, nest=False)
    w_nest = HealpixMapWeight(nside=nside, map_array=m_nest, nest=True)
    df = _df([10, 100, 200, 300], [0, 20, -10, 30])
    np.testing.assert_allclose(w_ring(df), w_nest(df))


def test_rejects_wrong_length_map():
    with pytest.raises(ValueError, match="length"):
        HealpixMapWeight(nside=32, map_array=np.ones(5), nest=True)


def test_nan_pixels_raise_unless_fill():
    nside = 32
    m = make_systematic_map(nside, seed=2)
    # Pick a real pixel id (not 0 — its centre depends on ordering).
    pix = 123
    theta_pix, phi_pix = hp.pix2ang(nside, pix, nest=True)
    ra = float(np.degrees(phi_pix))
    dec = float(90.0 - np.degrees(theta_pix))
    m[pix] = np.nan
    df = _df([ra], [dec])
    with pytest.raises(ValueError, match="NaN"):
        HealpixMapWeight(nside=nside, map_array=m, nest=True)(df)

    w2 = HealpixMapWeight(nside=nside, map_array=m, nest=True, nan_fill=0.0)
    assert w2(df)[0] == 0.0


def test_from_fits_roundtrip(tmp_path):
    nside = 32
    m = make_systematic_map(nside, seed=3)
    path = tmp_path / "sysmap.fits"
    hp.write_map(str(path), m, nest=True, overwrite=True)

    w = HealpixMapWeight.from_fits(path, nest=True)
    df = _df([10, 20, 30], [0, 5, -5])
    np.testing.assert_allclose(
        w(df), HealpixMapWeight(nside, m, nest=True)(df),
    )
