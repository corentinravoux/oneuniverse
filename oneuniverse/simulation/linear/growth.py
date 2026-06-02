"""Linear growth factor + rate for flat LambdaCDM.

D(z) via the Carroll, Press & Turner (1992) fitting formula, normalised
to D(0) = 1. f(z) = Omega_m(z)^0.55 (Linder 2005 gamma). Flat universe
assumed: Omega_Lambda = 1 - Omega_m.
"""
from __future__ import annotations

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear._cosmo import require_cosmo


def _growth_unnorm(a: float, om: float) -> float:
    ol = 1.0 - om
    e2 = om * a ** -3 + ol
    om_a = om * a ** -3 / e2
    ol_a = ol / e2
    return (
        2.5 * a * om_a
        / (om_a ** (4.0 / 7.0) - ol_a + (1.0 + om_a / 2.0) * (1.0 + ol_a / 70.0))
    )


def growth_factor(z: float, cosmo: CosmologySpec) -> float:
    """Linear growth D(z), normalised so D(0) = 1."""
    c = require_cosmo(cosmo)
    a = 1.0 / (1.0 + z)
    return _growth_unnorm(a, c.omega_m) / _growth_unnorm(1.0, c.omega_m)


def growth_rate(z: float, cosmo: CosmologySpec) -> float:
    """Linear growth rate f(z) = Omega_m(z)^0.55."""
    c = require_cosmo(cosmo)
    a = 1.0 / (1.0 + z)
    e2 = c.omega_m * a ** -3 + (1.0 - c.omega_m)
    om_a = c.omega_m * a ** -3 / e2
    return float(om_a ** 0.55)
