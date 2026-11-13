import numpy as np
import pytest

from oneuniverse._core.geometry import Cone, Cube, SkyPatch, ZRange


def test_cone_contains_haversine_small_radius():
    c = Cone(ra=10.0, dec=10.0, radius=1 / 3600)  # 1 arcsec
    assert c.contains(np.array([10.0, 10.0]), np.array([10.0, 10.0 + 0.5 / 3600])).tolist() == [True, True]
    assert not c.contains(np.array([10.0]), np.array([10.0 + 2 / 3600]))[0]


def test_skypatch_wrap():
    p = SkyPatch(350.0, 10.0, -5.0, 5.0)
    assert p.wraps and p.contains(np.array([355.0, 5.0, 180.0]), np.zeros(3)).tolist() == [True, True, False]


def test_skypatch_nowrap_and_dec():
    p = SkyPatch(10.0, 20.0, 0.0, 5.0)
    assert not p.wraps
    assert p.contains([15.0, 15.0, 25.0], [1.0, 6.0, 1.0]).tolist() == [True, False, False]


def test_zrange_and_cube():
    assert ZRange(0.1, 0.2).contains(np.array([0.15, 0.3])).tolist() == [True, False]
    cu = Cube((0, 0, 0), (10, 10, 10)); assert cu.volume == 1000.0
    assert cu.contains(np.array([5.0, 11.0]), np.zeros(2), np.zeros(2)).tolist() == [True, False]


def test_selections_are_frozen_and_hashable():
    assert hash(Cone(1, 2, 3)) == hash(Cone(1, 2, 3))
    with pytest.raises(AttributeError):
        Cone(1, 2, 3).ra = 5.0


@pytest.mark.parametrize("make", [
    lambda: Cone(0, 0, 0),
    lambda: Cone(0, 91, 1),
    lambda: SkyPatch(0, 10, 5, 5),
    lambda: SkyPatch(0, 10, -91, 0),
    lambda: SkyPatch(-1, 10, 0, 5),
    lambda: ZRange(0.2, 0.2),
    lambda: Cube((0, 0, 0), (1, 0, 1)),
])
def test_invalid_arguments_raise(make):
    with pytest.raises(ValueError):
        make()


def test_cube_normalises_to_float_tuples():
    ref = Cube((0, 0, 0), (1, 2, 3))
    for c in (Cube([0, 0, 0], [1, 2, 3]),
              Cube(np.array([0, 0, 0]), np.array([1, 2, 3]))):
        assert c == ref and hash(c) == hash(ref)
        assert isinstance(c.lo, tuple) and all(isinstance(v, float) for v in c.hi)


@pytest.mark.parametrize("lo,hi", [((0, 0), (1, 1, 1)), ((0, 0, 0), (1, 1, 1, 1))])
def test_cube_wrong_length_raises(lo, hi):
    from oneuniverse._core.errors import ValidationError
    with pytest.raises(ValidationError):
        Cube(lo, hi)


def test_geometry_errors_are_validation_errors():
    from oneuniverse._core.errors import ValidationError
    with pytest.raises(ValidationError):
        Cone(0, 0, -1)


@pytest.mark.parametrize("args", [(np.nan, 0, 1), (0, np.inf, 1), (0, 0, np.inf), (0, 0, np.nan)])
def test_cone_rejects_non_finite(args):
    with pytest.raises(ValueError):
        Cone(*args)


def test_skypatch_contains_accepts_negative_ra():
    p = SkyPatch(350.0, 10.0, -5.0, 5.0)
    assert p.contains([-5.0, 5.0, -90.0], [0, 0, 0]).tolist() == [True, True, False]
    q = SkyPatch(10.0, 360.0, -5.0, 5.0)
    assert q.contains([360.0, -170.0, 5.0], [0, 0, 0]).tolist() == [True, True, False]
