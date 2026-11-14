"""Reference physics plugin of the simulation substrate (``oneuniverse.reference``).

The single reference implementation of the toy physics used to exercise and
validate the simulation substrate and the twin contracts: linear theory and
Gaussian / Zel'dovich realisations (:mod:`.linear`) and the one P(k) / r(k)
estimator (:mod:`.fieldstats`). Later modules (particle-mesh, resimulation,
reconstruction, mocks, native fixture sims, engines) build on these
conventions.

Import the submodules explicitly, e.g. ``from oneuniverse.reference.linear
import gaussian_field``. ``reference`` depends on ``oneuniverse.simulation``
(specs), numpy and scipy only; ``simulation`` and ``twin.engine`` never import
it.
"""

__all__: list[str] = []
