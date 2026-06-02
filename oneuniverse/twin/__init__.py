"""oneuniverse.twin — the data <-> simulation coupling layer.

This is the third layer the ADR mandates: it may import BOTH
``oneuniverse.simulation`` and ``oneuniverse.data`` (the coupling that
neither pillar may host). ``oneuniverse.simulation`` stays Rule-1 clean;
the no-Pillar-1-import guard scans ``simulation/`` only, not ``twin/``.

MVP = the mock challenge: truth field -> mock-observe (biased Poisson
tracers) -> constrain (Wiener filter) -> verify (cross-correlation r(k)).
All linear theory + FFT, deterministic. The synthetic "data" is replaced
by real Pillar-1 selection as the first data-side complexification.
"""
from oneuniverse.twin.engine import (
    ForwardEngine,
    Observation,
    ProductBundle,
    ReconstructionEngine,
    get_engine,
    register_engine,
    registered_engines,
)
from oneuniverse.twin.engines import LinearForwardEngine, WienerReconstruction
from oneuniverse.twin.mock_challenge import run_mock_challenge
from oneuniverse.twin.mock_observe import mock_tracer_field
from oneuniverse.twin.verify import cross_correlation, power_ratio
from oneuniverse.twin.wiener import wiener_reconstruct

__all__ = [
    "mock_tracer_field", "wiener_reconstruct",
    "cross_correlation", "power_ratio", "run_mock_challenge",
    "ReconstructionEngine", "ForwardEngine", "Observation", "ProductBundle",
    "register_engine", "get_engine", "registered_engines",
    "WienerReconstruction", "LinearForwardEngine",
]
