"""oneuniverse.simulation.pm — a fast particle-mesh mini-simulator.

CIC deposit → FFT Poisson force → KDK leapfrog. The second `ForwardEngine`
(after the linear engine) and the forward half of the resimulation loop
(S8). Validated against linear growth + Zel'dovich. Pure numpy.

Standalone (Rule 1): no imports from oneuniverse.data / combine.
"""
from oneuniverse.simulation.pm.deposit import deposit_cic, interpolate_cic
from oneuniverse.simulation.pm.poisson import pm_force
from oneuniverse.simulation.pm.run import run_pm, zeldovich_pm_ic

__all__ = ["deposit_cic", "interpolate_cic", "pm_force", "run_pm",
           "zeldovich_pm_ic"]
