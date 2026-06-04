"""Buffer-convergence benchmark — the reference for testing resim solutions.

`buffer_convergence(resim_fn, ...)` reports the inner-region agreement with the
full-box reference as a function of buffer size, for *any* resimulation
function `resim_fn(buffer) -> inner_field`. The uncoupled buffered run is the
baseline; a candidate sCOLA solution plugs in the same way and is judged by
whether it reaches a given accuracy at a *smaller* buffer.
"""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.resim.coupling import (
    full_target_slice,
    run_coupled,
    run_full_reference,
)


def reference_inner(cosmo: CosmologySpec, *, box: float, n_grid: int,
                    target_lo: float, target_side: float, seed: int,
                    z_start: float = 9.0, z_end: float = 0.0,
                    n_steps: int = 20) -> np.ndarray:
    """The full-box PM reference, restricted to the target cube."""
    full = run_full_reference(cosmo, box=box, n_grid=n_grid, z_start=z_start,
                              z_end=z_end, seed=seed, n_steps=n_steps)
    return full_target_slice(full, box=box, n_grid=n_grid, target_lo=target_lo,
                             target_side=target_side)


def buffer_convergence(resim_fn: Callable[[float], np.ndarray],
                       reference: np.ndarray,
                       buffers: Sequence[float]) -> List[Tuple[float, float]]:
    """[(buffer, corr-with-reference)] for a resim function `resim_fn(buffer)`."""
    out = []
    for b in buffers:
        inner = resim_fn(b)
        out.append((float(b),
                    float(np.corrcoef(inner.ravel(), reference.ravel())[0, 1])))
    return out


def uncoupled_resim_fn(cosmo: CosmologySpec, *, box: float, n_grid: int,
                       target_lo: float, target_side: float, seed: int,
                       z_start: float = 9.0, z_end: float = 0.0,
                       n_steps: int = 20) -> Callable[[float], np.ndarray]:
    """The baseline: plain buffered PM resimulation as a `resim_fn(buffer)`."""
    def fn(buffer: float) -> np.ndarray:
        return run_coupled(cosmo, box=box, n_grid=n_grid, target_lo=target_lo,
                           target_side=target_side, buffer=buffer,
                           z_start=z_start, z_end=z_end, seed=seed,
                           n_steps=n_steps)["inner"]
    return fn
