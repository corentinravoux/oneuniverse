"""Mock 'observation': sample biased tracers from a truth density field.

A stand-in for the Pillar-1 data side. Expected count per cell
λ = n̄_cell · max(0, 1 + b·δ); counts ~ Poisson(λ). Returns the counts and
the observed tracer overdensity δ_g = counts/⟨counts⟩ − 1. Linear-bias +
clip (simplest; lognormal/HOD are later complexifications).

**Caveat (measured):** the clip biases the *effective* tracer bias low once
the field is non-linear (σ_cell≳1): e.g. at σ_cell≈1.8 a requested b=1.5
samples as b_eff≈0.6. Fine for correlation diagnostics r(k) (normalisation-
independent), but absolute-power work (Wiener gain, constrained realization)
must either operate in the linear regime (higher z / coarser cells) or fit
b_eff + shot noise from the data. A lognormal/HOD mock that preserves bias
is the proper fix (future).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def mock_tracer_field(delta, *, box_size, nbar, bias=1.0, seed=0,
                      mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Poisson-sample biased tracers from ``delta``; return counts + δ_g.

    Parameters
    ----------
    delta : (n,n,n) truth matter overdensity.
    box_size : Mpc/h.
    nbar : mean tracer number density (Mpc/h)^-3.
    bias : linear tracer bias b.
    mask : optional (n,n,n) selection in [0,1].
    """
    d = np.asarray(delta, dtype=np.float64)
    n = d.shape[0]
    v_cell = (box_size / n) ** 3
    nbar_cell = nbar * v_cell
    rng = np.random.default_rng(seed)
    lam = nbar_cell * np.clip(1.0 + bias * d, 0.0, None)
    if mask is not None:
        lam = lam * np.asarray(mask, dtype=np.float64)
    counts = rng.poisson(lam).astype(np.float64)
    # Observed overdensity is defined relative to the *realised* mean count
    # (as real surveys estimate n̄ from the data), which makes ⟨δ_g⟩ = 0 by
    # construction and absorbs the small clip-induced mean shift.
    if mask is not None:
        sel = np.asarray(mask, dtype=np.float64) > 0
        mean_count = counts[sel].mean()
        delta_g = np.zeros_like(counts)
        delta_g[sel] = counts[sel] / mean_count - 1.0
    else:
        mean_count = counts.mean()
        delta_g = counts / mean_count - 1.0
    return {"counts": counts, "delta_g": delta_g,
            "nbar": float(nbar), "bias": float(bias),
            "nbar_cell": float(nbar_cell),
            "mean_count": float(mean_count)}
