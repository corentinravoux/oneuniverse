"""Phase S2 T12 — public API surface."""
import oneuniverse.simulation as sim


def test_public_exports_present():
    for name in (
        "OUFSIM_FORMAT_VERSION",
        "ExecutionMode", "ExecutionPlan",
        "BackendCapabilities",
        "Cube", "Cone", "SkyPatch",
        "CosmologySpec", "UnitFrameSpec", "ProvenanceSpec",
        "ProductDecl",
        "OUFSimManifest", "read_manifest", "write_manifest",
        "OUFSimManifestError",
        "RegionSpec", "SimulationRequest",
        "SimConverter", "register", "get_converter",
        "detect_converter", "registered_codes",
    ):
        assert hasattr(sim, name), f"missing public export: {name}"
