import healpy as hp
import numpy as np
import pytest

from oneuniverse._core.errors import ValidationError
from oneuniverse.measure.atoms import Nz, Regions, Window


def test_nz_normalised_and_pdf():
    nz = Nz(np.array([0.0, 0.5, 1.0]), np.array([2.0, 6.0]), "spec_hist"); n = nz.normalised()
    assert np.isclose(np.trapezoid(n.pdf(np.linspace(0, 1, 1001)), np.linspace(0, 1, 1001)), 1.0, atol=1e-3) and n.pdf(np.array([0.25, 0.75]))[1] == 3 * n.pdf(np.array([0.25, 0.75]))[0]
    with pytest.raises(ValidationError): Nz(np.array([0.0, 1.0]), np.array([-1.0]), "spec_hist")


def test_nz_validation_and_2d():
    with pytest.raises(ValidationError): Nz(np.array([0.0, 1.0, 0.5]), np.array([1.0, 1.0]), "spec_hist")
    with pytest.raises(ValidationError): Nz(np.array([0.0, 1.0]), np.array([1.0, 1.0]), "spec_hist")
    with pytest.raises(ValidationError): Nz(np.array([0.0, 1.0]), np.array([np.nan]), "spec_hist")
    nz = Nz(np.array([0.0, 1.0, 2.0]), np.array([[1.0, 3.0], [2.0, 2.0]]), "photoz_stack", bins=(0, 1)).normalised()
    assert np.allclose((nz.counts * np.diff(nz.edges)).sum(axis=1), 1.0)
    assert (nz.pdf(np.array([-1.0, 5.0])) == 0).all()
    assert nz.pdf(np.array([0.5])).shape == (2, 1)


def test_window_from_mask_and_footprint():
    m = np.zeros(12 * 16**2, np.float32); m[:100] = 1.0; w = Window.from_mask(m, nside=16); assert w.fraction == pytest.approx(100 / len(m)) and w.source == "mask"
    with pytest.raises(ValidationError): Window.from_mask(m[:-1], nside=16)
    rng = np.random.default_rng(0); ra = rng.uniform(0, 10, 500); dec = rng.uniform(0, 10, 500); f = Window.footprint(ra, dec, nside=64)
    assert f.source == "footprint" and f.contains(ra, dec).all() and not f.contains(np.array([180.0]), np.array([0.0]))[0]


def test_window_area_and_systematics():
    m = np.ones(12 * 4**2, np.float32)
    w = Window.from_mask(m, nside=4)
    assert w.area_sr == pytest.approx(4 * np.pi)
    with pytest.raises(ValidationError): Window.from_mask(m * 2, nside=4)
    with pytest.raises(ValidationError): Window(4, True, m, "mask", {"s": np.zeros(3)})


def test_regions_from_window_deterministic_and_balanced():
    m = np.zeros(12 * 32**2, np.float32); m[:3000] = 1.0; w = Window.from_mask(m, nside=32)
    r1 = Regions.from_window(w, 10, seed=1); r2 = Regions.from_window(w, 10, seed=1)
    assert np.array_equal(r1.pixel_region, r2.pixel_region) and r1.n_regions == 10 and (r1.pixel_region[m == 0] == -1).all()
    counts = np.bincount(r1.pixel_region[r1.pixel_region >= 0]); assert counts.min() > 0.5 * counts.mean()
    ra, dec = hp.pix2ang(32, np.arange(10), nest=True, lonlat=True); assert (r1.assign(ra, dec) >= 0).all()
    assert r1 == r2 and r1.pixel_region.dtype == np.int32
    assert (r1.assign(np.array([180.0]), np.array([0.0])) == -1).all()
    with pytest.raises(ValidationError): Regions.from_window(w, 5000)


def test_regions_coarser_nside():
    m = np.zeros(12 * 32**2, np.float32); m[:3000] = 1.0; w = Window.from_mask(m, nside=32)
    r = Regions.from_window(w, 4, nside=8)
    assert r.nside == 8 and len(r.pixel_region) == 12 * 64
    with pytest.raises(ValidationError): Regions.from_window(w, 4, nside=64)


def test_atom_arrays_read_only():
    import pytest
    from oneuniverse.measure.atoms import Nz, Regions, Window
    nz = Nz([0, 1, 2], [1.0, 2.0], "spec_hist")
    w = Window.from_mask(np.ones(12 * 4 ** 2), nside=4)
    w2 = Window(4, True, np.ones(192), "mask", {"s": np.ones(192, np.int32)})
    r = Regions(4, True, 1, np.zeros(192, np.int32), "t")
    for a in (nz.edges, nz.counts, w.mask, w2.systematics["s"], r.pixel_region):
        with pytest.raises(ValueError):
            a[0] = 5
    assert w2.systematics["s"].dtype == np.float64


def test_window_systematics_must_be_numeric():
    import pytest
    from oneuniverse._core.errors import ValidationError
    from oneuniverse.measure.atoms import Window
    with pytest.raises(ValidationError, match="numeric"):
        Window(4, True, np.ones(192), "mask", {"s": ["a"] * 192})
