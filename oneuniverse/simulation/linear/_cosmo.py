"""Validate + complete a CosmologySpec for linear-sim use.

The storage CosmologySpec has all-optional fields; the generator needs
concrete omega_m / omega_b / h / n_s / sigma8 and a CMB temperature
(defaulted to 2.7255 K). ``require_cosmo`` returns a completed spec or
raises with the name of the first missing field.
"""
from __future__ import annotations

from dataclasses import replace

from oneuniverse.simulation.cosmology import CosmologySpec

_REQUIRED = ("omega_m", "omega_b", "h", "n_s", "sigma8")
_DEFAULT_TCMB = 2.7255


def require_cosmo(spec: CosmologySpec) -> CosmologySpec:
    for name in _REQUIRED:
        if getattr(spec, name) is None:
            raise ValueError(
                f"linear sim requires CosmologySpec.{name} to be set"
            )
    if spec.t_cmb is None:
        return replace(spec, t_cmb=_DEFAULT_TCMB)
    return spec
