"""Phase S2 T9 — SimulationRequest."""
import pytest

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.region import RegionSpec
from oneuniverse.simulation.request import SimulationRequest


def _region() -> RegionSpec:
    return RegionSpec(
        region_id="coma", kind="cluster",
        eulerian_bbox=(100.0, 110.0, 100.0, 110.0, 100.0, 110.0),
    )


def test_ok():
    req = SimulationRequest(
        request_id="req-001", parent_sim="AbacusSummit_base_c000_ph000",
        region=_region(), target_resolution=1e7,
        physics=("dm", "hydro"),
        cosmology=CosmologySpec(omega_m=0.31, sigma8=0.81, h=0.67),
        ic_strategy="zoom_from_parent_ic", code_hint="AREPO",
    )
    assert req.status == "pending"


def test_rejects_unknown_ic_strategy():
    with pytest.raises(ValueError, match="ic_strategy"):
        SimulationRequest(
            request_id="x", parent_sim=None, region=_region(),
            target_resolution=1.0, physics=("dm",),
            cosmology=CosmologySpec(), ic_strategy="teleport",
        )


def test_rejects_unknown_status():
    with pytest.raises(ValueError, match="status"):
        SimulationRequest(
            request_id="x", parent_sim=None, region=_region(),
            target_resolution=1.0, physics=("dm",),
            cosmology=CosmologySpec(), ic_strategy="fresh",
            status="exploded",
        )


def test_rejects_unknown_physics():
    with pytest.raises(ValueError, match="physics"):
        SimulationRequest(
            request_id="x", parent_sim=None, region=_region(),
            target_resolution=1.0, physics=("dm", "magic"),
            cosmology=CosmologySpec(), ic_strategy="fresh",
        )


def test_roundtrip():
    req = SimulationRequest(
        request_id="req-001", parent_sim="parent",
        region=_region(), target_resolution=1e7,
        physics=("dm", "hydro", "mhd"),
        cosmology=CosmologySpec(omega_m=0.31),
        ic_strategy="constrained_from_posterior", code_hint=None,
        status="dispatched", provenance={"submitted_by": "tester"},
    )
    assert SimulationRequest.from_dict(req.to_dict()) == req
