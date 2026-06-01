"""Phase S2 T5 — cosmology / unit-frame / provenance sidecar specs."""
import pytest

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.provenance import ProvenanceSpec
from oneuniverse.simulation.unit_frame import UnitFrameSpec


def test_cosmology_roundtrip():
    spec = CosmologySpec(
        omega_m=0.3089, omega_b=0.0486, h=0.6774, n_s=0.9667,
        sigma8=0.8159, w0=-1.0, wa=0.0, t_cmb=2.7255,
    )
    assert CosmologySpec.from_dict(spec.to_dict()) == spec


def test_cosmology_all_optional():
    spec = CosmologySpec()
    assert spec.omega_m is None
    assert CosmologySpec.from_dict(spec.to_dict()) == spec


def test_unit_frame_defaults_and_roundtrip():
    spec = UnitFrameSpec(
        length_unit="Mpc/h", mass_unit="Msun/h",
        velocity_unit="km/s peculiar",
    )
    assert spec.time_unit == "Gyr"
    assert spec.h_factor is True
    assert spec.comoving is True
    assert spec.frame == "icrs"
    assert spec.endianness == "native"
    assert UnitFrameSpec.from_dict(spec.to_dict()) == spec


def test_unit_frame_rejects_unknown_velocity():
    with pytest.raises(ValueError, match="velocity_unit"):
        UnitFrameSpec(
            length_unit="Mpc/h", mass_unit="Msun/h",
            velocity_unit="furlongs/fortnight",
        )


def test_provenance_roundtrip():
    spec = ProvenanceSpec(
        code="ABACUS", code_version="2.0", git_hash="deadbeef",
        original_paths=("/data/abacus/slab0",),
        ingested_utc="2026-06-01T00:00:00+00:00",
        converter="AbacusSummitOutputConverter",
    )
    assert ProvenanceSpec.from_dict(spec.to_dict()) == spec
