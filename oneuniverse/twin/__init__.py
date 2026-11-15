"""Digital twin: engine contracts and registry (physics lives in ``oneuniverse.reference``).

``observe_from_view`` is the data -> twin socket and ``run_engine_to_store`` the
engine <-> store boundary; the reference engines register themselves when
``oneuniverse.reference.engines`` is imported.
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
from oneuniverse.twin.observe import observe_from_view
from oneuniverse.twin.store_boundary import run_engine_to_store

__all__ = [
    "ForwardEngine",
    "Observation",
    "ProductBundle",
    "ReconstructionEngine",
    "get_engine",
    "observe_from_view",
    "register_engine",
    "registered_engines",
    "run_engine_to_store",
]
