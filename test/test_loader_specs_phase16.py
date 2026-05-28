"""Phase 16 T8 — loaders declare CoordinateSpec / SpectrumSpec where
the source survey publishes the info.
"""
from oneuniverse.data.surveys.peculiar_velocity.cosmicflows4.loader import (
    CosmicFlows4Loader,
)
from oneuniverse.data.surveys.snia.pantheonplus.loader import PantheonPlusLoader
from oneuniverse.data.surveys.spectroscopic.eboss_qso.loader import EbossQSOLoader
from oneuniverse.data.surveys.spectroscopic.sdss_mgs.loader import SDSSMGSLoader
from oneuniverse.data.surveys.spectroscopic.desi_qso.loader import DESIQSOLoader


def test_eboss_qso_declares_icrs_and_vacuum():
    spec = EbossQSOLoader.coordinate_spec()
    assert spec.frame == "icrs"
    sspec = EbossQSOLoader.spectrum_spec()
    assert sspec.wavelength_convention == "vacuum"


def test_sdss_mgs_declares_icrs_and_air():
    spec = SDSSMGSLoader.coordinate_spec()
    assert spec.frame == "icrs"
    sspec = SDSSMGSLoader.spectrum_spec()
    assert sspec.wavelength_convention == "air"


def test_desi_qso_declares_icrs_and_vacuum():
    spec = DESIQSOLoader.coordinate_spec()
    assert spec.frame == "icrs"
    sspec = DESIQSOLoader.spectrum_spec()
    assert sspec.wavelength_convention == "vacuum"


def test_pantheonplus_declares_icrs_no_spectrum():
    spec = PantheonPlusLoader.coordinate_spec()
    assert spec.frame == "icrs"
    assert PantheonPlusLoader.spectrum_spec() is None


def test_cosmicflows4_declares_icrs_no_spectrum():
    spec = CosmicFlows4Loader.coordinate_spec()
    assert spec.frame == "icrs"
    assert CosmicFlows4Loader.spectrum_spec() is None
