"""oneuniverse.simulation — OUF-Sim: storage + orchestration of cosmological
simulations (Pillar 3, digital-twin substrate).

Standalone subpackage. **Must not import** from ``oneuniverse.data`` or
``oneuniverse.combine`` (enforced by test_sim_no_pillar1_imports).
"""
from oneuniverse.simulation._version import OUFSIM_FORMAT_VERSION

__all__ = ["OUFSIM_FORMAT_VERSION"]
