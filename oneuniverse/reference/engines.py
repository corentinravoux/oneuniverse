"""The four reference engines, registered on the twin contract at import.

``LinearForwardEngine`` ("linear"), ``PMForwardEngine`` ("fastpm"),
``WienerReconstruction`` ("wiener") and ``ConstrainedRealization``
("constrained"). Importing this module registers them; ``oneuniverse.twin``
never imports it.

Conventions: forward engines need a ``seed`` or an ``ic_field`` (a z = 0 linear
density contrast on the ``n_grid`` mesh; it wins over the seed, which is still
recorded). ``ProductBundle.provenance`` holds ``engine``, ``parameters``
(constructor arguments), ``seed`` and ``ic_field`` (bool). Reconstruction
engines need ``Observation.box`` to be set.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from oneuniverse._core.errors import ValidationError
from oneuniverse.reference.linear import gaussian_field, growth_factor
from oneuniverse.reference.pm import (
    run_pm,
    zeldovich_pm_ic,
    zeldovich_pm_ic_from_field,
)
from oneuniverse.reference.reconstruct import (
    constrained_realization,
    wiener_reconstruct,
)
from oneuniverse.twin.engine import (
    ForwardEngine,
    Observation,
    ProductBundle,
    ReconstructionEngine,
    register_engine,
)


def _check_ic(ic_field, n_grid: int) -> np.ndarray:
    ic = np.asarray(ic_field, dtype=np.float64)
    if ic.shape != (n_grid,) * 3:
        raise ValidationError(f"ic_field has shape {ic.shape}, engine needs {(n_grid,) * 3}")
    return ic


def _need_source(seed, ic_field) -> None:
    if seed is None and ic_field is None:
        raise ValidationError("forward needs a seed or an ic_field")


def _provenance(name: str, parameters: dict[str, Any], seed, ic_field) -> dict[str, Any]:
    return {"engine": name, "parameters": dict(parameters),
            "seed": None if seed is None else int(seed), "ic_field": ic_field is not None}


@register_engine
class LinearForwardEngine(ForwardEngine):
    """Linear theory: ``δ(z) = D(z)·δ(0)`` with δ(0) from ``seed`` or ``ic_field``."""

    name = "linear"

    def forward(self, *, cosmo, box, n_grid, z, seed=None, ic_field=None) -> ProductBundle:
        _need_source(seed, ic_field)
        if ic_field is not None:
            d0 = _check_ic(ic_field, n_grid)
        else:
            d0 = gaussian_field(cosmo, box=box, n_grid=n_grid, z=0.0, seed=seed)
        return ProductBundle({"delta": d0 * growth_factor(z, cosmo)}, float(z),
                             _provenance(self.name, {}, seed, ic_field))


@register_engine
class PMForwardEngine(ForwardEngine):
    """Particle-mesh run from Zel'dovich ICs at ``z_start`` down to ``z``; field ``delta`` = CIC δ.

    Constructor defaults: ``z_start=9.0``, ``n_steps=20`` (KDK steps from ``z_start`` to
    ``z``), ``force="periodic"``. ``ic_field`` (z = 0 linear δ) replaces the seeded draw.
    """

    name = "fastpm"

    def __init__(self, *, z_start: float = 9.0, n_steps: int = 20, force: str = "periodic"):
        self.z_start, self.n_steps, self.force = float(z_start), int(n_steps), force

    def forward(self, *, cosmo, box, n_grid, z, seed=None, ic_field=None) -> ProductBundle:
        _need_source(seed, ic_field)
        if ic_field is not None:
            pos, vel = zeldovich_pm_ic_from_field(cosmo, _check_ic(ic_field, n_grid),
                                                  box=box, z_start=self.z_start)
        else:
            pos, vel, _ = zeldovich_pm_ic(cosmo, box=box, n_grid=n_grid,
                                          z_start=self.z_start, seed=seed)
        out = run_pm(pos, vel, cosmo, box=box, n_grid=n_grid, z_start=self.z_start,
                     z_end=float(z), n_steps=self.n_steps, force=self.force, snapshots=(float(z),))
        params = {"z_start": self.z_start, "n_steps": self.n_steps, "force": self.force}
        return ProductBundle({"delta": out["snapshots"][float(z)]}, float(z),
                             _provenance(self.name, params, seed, ic_field))


def _need_box(obs: Observation) -> float:
    if not obs.box > 0:
        raise ValidationError("Observation.box must be set (> 0) to reconstruct")
    return float(obs.box)


@register_engine
class WienerReconstruction(ReconstructionEngine):
    """Mask-aware Wiener filter (:func:`~oneuniverse.reference.reconstruct.wiener_reconstruct`)."""

    name = "wiener"

    def reconstruct(self, observation: Observation, *, cosmo, z: float = 0.0) -> np.ndarray:
        return wiener_reconstruct(observation.delta_g, cosmo, box=_need_box(observation),
                                  nbar=observation.nbar, bias=observation.bias, z=z,
                                  mask=observation.mask)


@register_engine
class ConstrainedRealization(ReconstructionEngine):
    """Hoffman-Ribak constrained realisation; ``seed`` is required at construction."""

    name = "constrained"

    def __init__(self, *, seed: int):
        self.seed = int(seed)

    def reconstruct(self, observation: Observation, *, cosmo, z: float = 0.0) -> np.ndarray:
        return constrained_realization(observation.delta_g, cosmo, box=_need_box(observation),
                                       nbar=observation.nbar, bias=observation.bias, z=z,
                                       seed=self.seed, mask=observation.mask)
