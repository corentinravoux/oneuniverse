"""oneuniverse.simulation.linear — a pure-numpy linear-theory dummy
simulation: Eisenstein-Hu P(k) + Gaussian field + Zel'dovich particles
+ toy halos. The synthetic source used to finish the OUF-Sim machinery.

Standalone (Rule 1): no imports from oneuniverse.data / combine.
"""
from oneuniverse.simulation.linear.converter import LinearSimConverter
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.simulation.linear.generate import generate_linear_sim
from oneuniverse.simulation.linear.growth import growth_factor, growth_rate
from oneuniverse.simulation.linear.halos import find_peaks
from oneuniverse.simulation.linear.lightcone import build_lightcone_catalog
from oneuniverse.simulation.linear.power_spectrum import (
    linear_power,
    sigma_R,
    transfer_eh_nowiggle,
    unnormalised_power,
)
from oneuniverse.simulation.linear.zeldovich import zeldovich_particles

__all__ = [
    "transfer_eh_nowiggle", "unnormalised_power", "sigma_R", "linear_power",
    "growth_factor", "growth_rate",
    "generate_density_field", "zeldovich_particles", "find_peaks",
    "build_lightcone_catalog", "generate_linear_sim", "LinearSimConverter",
]
