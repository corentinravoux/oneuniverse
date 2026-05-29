"""Phase 22 T2 — CubeSpec sub-spec."""
import pytest

from oneuniverse.data.cube_spec import CubeSpec


def test_defaults():
    spec = CubeSpec(
        axes=("ra", "dec", "wavelength"),
        axis_units=("deg", "deg", "angstrom"),
        wavelength_convention="vacuum",
    )
    assert spec.axes == ("ra", "dec", "wavelength")
    assert spec.wavelength_convention == "vacuum"


def test_axes_axis_units_must_match_length():
    with pytest.raises(ValueError, match="length"):
        CubeSpec(
            axes=("ra", "dec", "wavelength"),
            axis_units=("deg", "deg"),
        )


def test_to_from_dict_roundtrip():
    spec = CubeSpec(
        axes=("ra", "dec", "frequency"),
        axis_units=("deg", "deg", "MHz"),
        wavelength_convention="vacuum",
    )
    d = spec.to_dict()
    assert CubeSpec.from_dict(d) == spec
