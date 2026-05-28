"""Phase 16 T4 — CoordinateSpec sub-spec."""
import pytest

from oneuniverse.data.coordinate_spec import CoordinateSpec


def test_defaults():
    spec = CoordinateSpec()
    assert spec.frame == "icrs"
    assert spec.epoch is None
    assert spec.proper_motion_available is False
    assert spec.parallax_available is False


def test_rejects_unknown_frame():
    with pytest.raises(ValueError, match="frame"):
        CoordinateSpec(frame="middle_earth")


def test_to_dict_and_from_dict_roundtrip():
    spec = CoordinateSpec(
        frame="icrs",
        epoch=2016.0,
        proper_motion_available=True,
        parallax_available=True,
    )
    d = spec.to_dict()
    assert d == {
        "frame": "icrs",
        "epoch": 2016.0,
        "proper_motion_available": True,
        "parallax_available": True,
    }
    assert CoordinateSpec.from_dict(d) == spec


def test_from_dict_tolerates_missing_optional_fields():
    spec = CoordinateSpec.from_dict({"frame": "galactic"})
    assert spec.frame == "galactic"
    assert spec.epoch is None
    assert spec.proper_motion_available is False
    assert spec.parallax_available is False


def test_is_frozen():
    spec = CoordinateSpec(frame="icrs")
    with pytest.raises(Exception):
        spec.frame = "galactic"  # type: ignore[misc]
