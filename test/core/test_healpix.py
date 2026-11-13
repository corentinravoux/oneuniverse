import healpy as hp
import numpy as np
import pytest

from oneuniverse._core.geometry import Cone, SkyPatch
from oneuniverse._core.healpix import (
    NSIDE_FINE, coarsen, cone_pixels, patch_pixels, radec_to_pix,
)


def test_radec_to_pix_matches_healpy_nest():
    ra = np.array([0.0, 185.0, 359.9]); dec = np.array([-30.0, 15.0, 89.0])
    exp = hp.ang2pix(32, ra, dec, nest=True, lonlat=True)
    out = radec_to_pix(32, ra, dec)
    assert out.dtype == np.int32 and np.array_equal(out, exp)
    assert NSIDE_FINE == 32


def test_radec_to_pix_ring():
    ra = np.array([10.0]); dec = np.array([20.0])
    assert radec_to_pix(8, ra, dec, nest=False)[0] == hp.ang2pix(8, ra, dec, lonlat=True)[0]


def test_coarsen_is_nest_right_shift():
    pix = radec_to_pix(32, np.array([10.0]), np.array([10.0]))
    assert coarsen(pix, 32, 8)[0] == pix[0] >> 4
    assert coarsen(pix, 32, 8).dtype == np.int32
    with pytest.raises(ValueError): coarsen(pix, 32, 64)
    with pytest.raises(ValueError): coarsen(pix, 33, 8)
    with pytest.raises(ValueError): coarsen(pix, 32, 6)


def _sky(n=5000):
    rng = np.random.default_rng(0)
    return rng.uniform(0, 360, n), np.degrees(np.arcsin(rng.uniform(-1, 1, n)))


def test_cone_pixels_covers_all_member_points():
    ra, dec = _sky()
    cone = Cone(ra=0.5, dec=-0.5, radius=3.0); inside = cone.contains(ra, dec)
    cells = cone_pixels(8, cone.ra, cone.dec, cone.radius)
    assert cells.dtype == np.int32 and np.all(np.diff(cells) > 0)
    assert np.all(np.isin(radec_to_pix(8, ra[inside], dec[inside]), cells))


def test_patch_pixels_wraps_ra():
    cells = patch_pixels(8, 350.0, 10.0, -5.0, 5.0)
    pts = radec_to_pix(8, np.array([355.0, 5.0]), np.array([0.0, 0.0]))
    assert np.all(np.isin(pts, cells))


@pytest.mark.parametrize("patch", [
    (20.0, 60.0, 30.0, 50.0),      # off-equator, great-circle bulge matters
    (350.0, 10.0, -5.0, 5.0),      # wrapped
    (10.0, 350.0, -20.0, 20.0),    # > 180 deg in RA: strip fallback
    (0.0, 360.0, 40.0, 90.0),      # touches the pole
])
def test_patch_pixels_cover_member_points_and_sorted_unique(patch):
    ra, dec = _sky(20000)
    p = SkyPatch(*patch); inside = p.contains(ra, dec)
    assert inside.any()
    cells = patch_pixels(16, *patch)
    assert cells.dtype == np.int32 and np.all(np.diff(cells) > 0)
    assert np.all(np.isin(radec_to_pix(16, ra[inside], dec[inside]), cells))


def test_patch_pixels_fallback_explicit_points():
    cells = patch_pixels(8, 10.0, 350.0, -5.0, 5.0)
    pts = radec_to_pix(8, np.array([20.0, 340.0]), np.zeros(2))
    assert np.all(np.isin(pts, cells))


def test_patch_pixels_ring_ordering():
    cells = patch_pixels(8, 20.0, 60.0, 30.0, 50.0, nest=False)
    pt = radec_to_pix(8, np.array([40.0]), np.array([40.0]), nest=False)
    assert pt[0] in cells


@pytest.mark.parametrize("nside,patch", [
    (1, (10.0, 55.0, 89.6, 89.9)),
    (1, (10.0, 55.0, -89.9, -89.6)),
    (2, (10.0, 100.0, 89.95, 89.99)),
    (8, (10.0, 100.0, 89.999, 89.9995)),
    (32, (10.0, 100.0, 89.9999, 89.99995)),
    (8, (10.0, 100.0, 40.0, 40.0001)),     # thin Dec
    (8, (10.0, 10.0 + 1e-10, 10.0, 20.0)),  # thin RA
    (8, (10.0, 10.5, 10.0, 20.0)),
])
def test_patch_pixels_near_pole_and_thin_patches_do_not_raise(nside, patch):
    p = SkyPatch(*patch)
    rng = np.random.default_rng(1)
    ra = rng.uniform(patch[0], patch[1], 500)
    dec = rng.uniform(patch[2], patch[3], 500)
    assert p.contains(ra, dec).all()
    cells = patch_pixels(nside, *patch)
    assert cells.dtype == np.int32 and len(cells) > 0
    assert np.all(np.isin(radec_to_pix(nside, ra, dec), cells))


def test_radec_to_pix_validates_inputs():
    from oneuniverse._core.errors import ValidationError
    for ra, dec in [(np.nan, 0.0), (0.0, np.inf), (0.0, 91.0), (0.0, -90.5)]:
        with pytest.raises(ValidationError):
            radec_to_pix(8, np.array([ra]), np.array([dec]))
    assert radec_to_pix(8, [0.0], [90.0]).shape == (1,)


@pytest.mark.parametrize("nside", [0, 3, 16384, -8, 8.5])
def test_nside_validated_everywhere(nside):
    from oneuniverse._core.errors import ValidationError
    with pytest.raises(ValidationError):
        radec_to_pix(nside, [1.0], [1.0])
    with pytest.raises(ValidationError):
        cone_pixels(nside, 1.0, 1.0, 1.0)
    with pytest.raises(ValidationError):
        patch_pixels(nside, 1.0, 2.0, 0.0, 1.0)
    with pytest.raises(ValidationError):
        coarsen([1], nside, 1)


def test_max_nside_accepted():
    assert radec_to_pix(8192, [10.0], [10.0]).dtype == np.int32


@pytest.mark.parametrize("radius", [0.0, -1.0, np.nan, np.inf])
def test_cone_pixels_rejects_bad_radius(radius):
    from oneuniverse._core.errors import ValidationError
    with pytest.raises(ValidationError):
        cone_pixels(8, 10.0, 10.0, radius)


@pytest.mark.parametrize("args", [
    (-1.0, 10.0, 0.0, 5.0), (0.0, 361.0, 0.0, 5.0),
    (0.0, 10.0, 5.0, 5.0), (0.0, 10.0, -91.0, 5.0), (np.nan, 10.0, 0.0, 5.0),
])
def test_patch_pixels_validates_arguments(args):
    from oneuniverse._core.errors import ValidationError
    with pytest.raises(ValidationError):
        patch_pixels(8, *args)


def test_wrapped_patch_with_degenerate_piece_is_not_a_full_strip():
    a = patch_pixels(8, 350.0, 0.0, -5.0, 5.0)
    b = patch_pixels(8, 350.0, 360.0, -5.0, 5.0)
    full = patch_pixels(8, 0.0, 360.0, -5.0, 5.0)
    assert len(a) < len(full) / 2 and abs(len(a) - len(b)) <= 2
    ra, dec = _sky(40000)
    p = SkyPatch(350.0, 0.0, -5.0, 5.0); inside = p.contains(ra, dec)
    assert inside.any() and np.all(np.isin(radec_to_pix(8, ra[inside], dec[inside]), a))


def test_thin_patch_is_widened_not_full_strip():
    cells = patch_pixels(8, 10.0, 10.0 + 1e-10, 10.0, 20.0)
    full = patch_pixels(8, 0.0, 360.0, 10.0, 20.0)
    assert 0 < len(cells) < len(full) / 4
    pts = radec_to_pix(8, np.full(50, 10.0), np.linspace(10.0, 20.0, 50))
    assert np.all(np.isin(pts, cells))
