"""Phase S2 T7 — OUFSimManifest + YAML round-trip."""
import pytest

from oneuniverse.simulation._version import OUFSIM_FORMAT_VERSION
from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.manifest import (
    OUFSimManifest,
    OUFSimManifestError,
    read_manifest,
    write_manifest,
)
from oneuniverse.simulation.product import ProductDecl
from oneuniverse.simulation.provenance import ProvenanceSpec
from oneuniverse.simulation.unit_frame import UnitFrameSpec


def _minimal(**overrides) -> OUFSimManifest:
    defaults = dict(
        oufsim_format_version=OUFSIM_FORMAT_VERSION,
        sim_name="AbacusSummit_base_c000_ph000",
        sim_kind="nbody",
        code="ABACUS",
        code_version="2.0",
        layout_schema="per_cosmology_phase_snapshot",
        backends=("ASDF/pack9", "CompaSO ASDF"),
        has_input=False,
        has_output=True,
        products=("snapshots", "halos"),
        n_snapshots=12,
        redshifts=(0.1, 0.2, 0.5),
        box_size=2000.0,
        n_particles=6912 ** 3,
        cosmology=CosmologySpec(omega_m=0.3137, sigma8=0.8076, h=0.6736),
        unit_frame=UnitFrameSpec(
            length_unit="Mpc/h", mass_unit="Msun/h",
            velocity_unit="km/s peculiar",
        ),
        provenance=ProvenanceSpec(
            code="ABACUS", code_version="2.0", git_hash=None,
            original_paths=("/cfs/abacus/base_c000_ph000",),
            ingested_utc="2026-06-01T00:00:00+00:00",
            converter="AbacusSummitOutputConverter",
        ),
        product_decls=(
            ProductDecl(
                product="snapshots", native_format="ASDF/pack9",
                indexes=("healpix_tiles",), fields=("Coordinates",),
            ),
        ),
    )
    defaults.update(overrides)
    return OUFSimManifest(**defaults)


def test_version_constant():
    assert OUFSIM_FORMAT_VERSION == "0.1.0"


def test_rejects_unknown_sim_kind():
    with pytest.raises(ValueError, match="sim_kind"):
        _minimal(sim_kind="quantum_foam")


def test_rejects_unknown_product():
    with pytest.raises(ValueError, match="products"):
        _minimal(products=("snapshots", "bogus"))


def test_rejects_unknown_layout_schema():
    with pytest.raises(ValueError, match="layout_schema"):
        _minimal(layout_schema="spaghetti")


def test_yaml_roundtrip(tmp_path):
    m = _minimal()
    path = tmp_path / "manifest.yaml"
    write_manifest(path, m)
    read = read_manifest(path)
    assert read == m


def test_read_rejects_incompatible_major(tmp_path):
    import yaml
    payload = {
        "oufsim_format_version": "9.9.9",
        "sim_name": "x", "sim_kind": "nbody", "code": "X",
        "code_version": None, "layout_schema": "per_cosmology_phase_snapshot",
        "backends": [], "has_input": False, "has_output": True,
        "products": [], "n_snapshots": 0, "redshifts": [],
        "box_size": None, "n_particles": None,
        "cosmology": None, "unit_frame": None, "provenance": None,
        "product_decls": [],
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(OUFSimManifestError, match="version"):
        read_manifest(path)


def test_read_missing_file_raises(tmp_path):
    with pytest.raises(OUFSimManifestError, match="not found"):
        read_manifest(tmp_path / "nope.yaml")
