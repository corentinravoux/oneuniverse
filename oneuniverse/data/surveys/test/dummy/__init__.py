"""
oneuniverse.data.surveys.test.dummy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Deterministic in-memory dummy catalog for tests and small runs.

- 5 000 galaxies, full sky, z in [0.005, 0.15]
- Seed 42 — fully reproducible, no file I/O
- Provides core + spectroscopic + peculiar_velocity column groups
"""

from oneuniverse.data.surveys.test.dummy.loader import DummyLoader  # noqa: F401
