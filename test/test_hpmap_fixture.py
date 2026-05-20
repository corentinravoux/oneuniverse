"""Smoke tests for the HEALPix map fixture factory."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.healpix_maps import (  # noqa: E402
    make_footprint_mask, make_smooth_completeness, make_systematic_map,
)


def test_footprint_binary_and_shape():
    import healpy as hp
    m = make_footprint_mask(nside=32, seed=0)
    assert set(np.unique(m)) <= {0.0, 1.0}
    assert m.shape == (hp.nside2npix(32),)


def test_completeness_in_unit_interval():
    m = make_smooth_completeness(nside=32, seed=0)
    assert (m >= 0).all() and (m <= 1).all()


def test_systematic_map_finite_positive():
    m = make_systematic_map(nside=32, seed=0)
    assert np.all(np.isfinite(m))
    assert (m > 0).all()
