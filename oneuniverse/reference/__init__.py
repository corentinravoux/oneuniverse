"""Reference physics plugin of the simulation substrate (``oneuniverse.reference``).

The single reference implementation of the toy physics used to exercise and
validate the simulation substrate and the twin contracts: linear theory and
Gaussian / Zel'dovich realisations (:mod:`.linear`) and the one P(k) / r(k)
estimator (:mod:`.fieldstats`), particle-mesh and resimulation
(:mod:`.pm`, :mod:`.resim`), the mask-aware Wiener filter and seed-explicit
constrained realisation (:mod:`.reconstruct`) and lognormal tracer mocks
(:mod:`.mock`), the linear fixture simulation on disk with its two converters
(:mod:`.native`) and the four reference engines on the twin contract
(:mod:`.engines`).

Plugins register on import: ``import oneuniverse.reference.engines`` registers
the engines with ``oneuniverse.twin``, ``import oneuniverse.reference.native``
registers the ``linear`` and ``packed_npy`` converters. Nothing is registered
by importing ``oneuniverse.reference`` itself.

Import the submodules explicitly, e.g. ``from oneuniverse.reference.linear
import gaussian_field``. ``reference`` depends on ``oneuniverse.simulation``
(specs, store), numpy and scipy only (only ``engines`` also imports
``oneuniverse.twin.engine``); ``simulation`` and ``twin.engine`` never import it.
"""

__all__: list[str] = []
