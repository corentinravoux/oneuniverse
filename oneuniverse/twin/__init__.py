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
