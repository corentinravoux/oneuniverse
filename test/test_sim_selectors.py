"""Phase S2 T4 — spatial selectors."""
import pytest

from oneuniverse.simulation.selectors import Cone, Cube, SkyPatch


def test_cube_ok():
    c = Cube(0.0, 1.0, 0.0, 2.0, 0.0, 3.0)
    assert c.xhi == 1.0


def test_cube_rejects_inverted():
    with pytest.raises(ValueError, match="xlo"):
        Cube(1.0, 0.0, 0.0, 1.0, 0.0, 1.0)


def test_cone_ok():
    c = Cone(lon=120.0, lat=0.0, radius_deg=5.0)
    assert c.radius_deg == 5.0


def test_cone_rejects_nonpositive_radius():
    with pytest.raises(ValueError, match="radius_deg"):
        Cone(lon=0.0, lat=0.0, radius_deg=0.0)


def test_skypatch_ok():
    p = SkyPatch(0.0, 30.0, -10.0, 10.0)
    assert p.lon_max == 30.0


def test_skypatch_rejects_inverted_lat():
    with pytest.raises(ValueError, match="lat"):
        SkyPatch(0.0, 30.0, 10.0, -10.0)


def test_selectors_frozen():
    c = Cube(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    with pytest.raises(Exception):
        c.xlo = 5.0  # type: ignore[misc]
