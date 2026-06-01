"""Phase S2 T10 — SimConverter ABC + registry."""
from pathlib import Path

import pytest

from oneuniverse.simulation.capabilities import BackendCapabilities
from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.converter import (
    SimConverter,
    detect_converter,
    get_converter,
    register,
    registered_codes,
)
from oneuniverse.simulation.product import ProductDecl
from oneuniverse.simulation.unit_frame import UnitFrameSpec


@register
class _DummyConverter(SimConverter):
    code = "DUMMY"
    sim_kind = "nbody"
    capabilities = BackendCapabilities(name="dummy", native_format="dummy-fmt")

    def detect(self, path: Path) -> bool:
        return Path(path).name == "dummy_sim"

    def declare_products(self, src: Path):
        return (
            ProductDecl(
                product="snapshots", native_format="dummy-fmt",
                indexes=(), fields=("Coordinates",),
            ),
        )

    def read_cosmology(self, src: Path) -> CosmologySpec:
        return CosmologySpec(omega_m=0.3)

    def read_unit_frame(self, src: Path) -> UnitFrameSpec:
        return UnitFrameSpec(
            length_unit="Mpc/h", mass_unit="Msun/h",
            velocity_unit="km/s peculiar",
        )


def test_registered():
    assert "DUMMY" in registered_codes()
    assert get_converter("DUMMY") is _DummyConverter


def test_get_unknown_raises():
    with pytest.raises(KeyError, match="UNKNOWN"):
        get_converter("UNKNOWN")


def test_register_rejects_duplicate():
    with pytest.raises(ValueError, match="already"):
        register(_DummyConverter)


def test_register_rejects_missing_code():
    with pytest.raises(ValueError, match="code"):
        @register
        class _NoCode(SimConverter):  # noqa: N801
            sim_kind = "nbody"
            capabilities = BackendCapabilities(name="n", native_format="f")

            def detect(self, path): return False
            def declare_products(self, src): return ()
            def read_cosmology(self, src): return CosmologySpec()
            def read_unit_frame(self, src):
                return UnitFrameSpec(
                    length_unit="Mpc/h", mass_unit="Msun/h",
                    velocity_unit="km/s peculiar",
                )


def test_detect_converter(tmp_path):
    target = tmp_path / "dummy_sim"
    target.mkdir()
    assert detect_converter(target) is _DummyConverter
    other = tmp_path / "other_sim"
    other.mkdir()
    assert detect_converter(other) is None


def test_convert_not_implemented_in_s2(tmp_path):
    """convert() lands in Phase S3 — S2 ABC raises NotImplementedError."""
    conv = _DummyConverter()
    with pytest.raises(NotImplementedError, match="S3"):
        conv.convert(tmp_path, tmp_path / "out")
