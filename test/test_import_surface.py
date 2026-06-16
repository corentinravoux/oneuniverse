"""Pins the public import surface refactored in Part A. Every name here is
imported somewhere (notebooks, scripts, downstream). Moving its definition is
fine; changing where it can be imported from is a breaking change."""
import importlib
import pytest

SURFACE = {
    "oneuniverse.data.converter": [
        "write_ouf_dataset", "convert_survey", "convert_sightlines",
        "convert_healpix_map", "read_oneuniverse_parquet", "read_objects_table",
        "fetch_original_columns", "get_manifest", "is_converted", "get_geometry",
    ],
    "oneuniverse.data._registry": [
        "register", "get_loader", "list_surveys", "survey_status",
        "list_survey_types", "get_survey_config", "REGISTRY",
    ],
    "oneuniverse.simulation.converter": ["SimConverter", "register", "get_converter"],
    "oneuniverse.twin.engine": [
        "register_engine", "get_engine", "registered_engines",
        "ForwardEngine", "ReconstructionEngine", "Observation", "ProductBundle",
    ],
    "oneuniverse.simulation.oufsim.native": [
        "ADAPTERS", "register_adapter", "get_adapter",
    ],
    "oneuniverse.twin": [
        "cross_correlation", "power_ratio", "recover_metrics", "RecoveryMetrics",
        "wiener_reconstruct", "constrained_realization", "run_mock_challenge",
    ],
}


@pytest.mark.parametrize("module,names", SURFACE.items())
def test_public_names_importable(module, names):
    mod = importlib.import_module(module)
    missing = [n for n in names if not hasattr(mod, n)]
    assert not missing, f"{module} lost public names: {missing}"
