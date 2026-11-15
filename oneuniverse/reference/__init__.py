"""Reference physics plugin of the simulation substrate (``oneuniverse.reference``).

The single reference implementation of the toy physics used to exercise and
validate the simulation substrate and the twin contracts: linear theory and
Gaussian / Zel'dovich realisations (:mod:`.linear`) and the one P(k) / r(k)
estimator (:mod:`.fieldstats`), particle-mesh and resimulation
(:mod:`.pm`, :mod:`.resim`), the mask-aware Wiener filter and seed-explicit
constrained realisation (:mod:`.reconstruct`) and lognormal tracer mocks
(:mod:`.mock`). Later modules (native fixture sims, engines) build on these
conventions.

Import the submodules explicitly, e.g. ``from oneuniverse.reference.linear
import gaussian_field``. ``reference`` depends on ``oneuniverse.simulation``
(specs), numpy and scipy only; ``simulation`` and ``twin.engine`` never import
it.
"""

__all__: list[str] = []
