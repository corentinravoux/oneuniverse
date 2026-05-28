"""Phase 16 T5 — SpectrumSpec sub-spec (SIGHTLINE datasets only)."""
import pytest

from oneuniverse.data.spectrum_spec import SpectrumSpec


def test_defaults():
    spec = SpectrumSpec(wavelength_convention="vacuum")
    assert spec.wavelength_convention == "vacuum"
    assert spec.log_binned is True
    assert spec.rest_frame_corrected is False
    assert spec.wavelength_unit == "angstrom"


def test_rejects_unknown_convention():
    with pytest.raises(ValueError, match="wavelength_convention"):
        SpectrumSpec(wavelength_convention="ether")


def test_rejects_unknown_unit():
    with pytest.raises(ValueError, match="wavelength_unit"):
        SpectrumSpec(wavelength_convention="vacuum", wavelength_unit="parsec")


def test_to_dict_and_from_dict_roundtrip():
    spec = SpectrumSpec(
        wavelength_convention="air",
        log_binned=False,
        rest_frame_corrected=True,
        wavelength_unit="nanometer",
    )
    d = spec.to_dict()
    assert d == {
        "wavelength_convention": "air",
        "log_binned": False,
        "rest_frame_corrected": True,
        "wavelength_unit": "nanometer",
    }
    assert SpectrumSpec.from_dict(d) == spec


def test_from_dict_tolerates_missing_optional_fields():
    spec = SpectrumSpec.from_dict({"wavelength_convention": "vacuum"})
    assert spec.log_binned is True
    assert spec.rest_frame_corrected is False
    assert spec.wavelength_unit == "angstrom"
