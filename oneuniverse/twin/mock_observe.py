"""Mock 'observation': sample biased tracers from a truth density field.

A stand-in for the Pillar-1 data side. Two intensity models:

- ``"clip"`` (default): λ = n̄_cell · max(0, 1 + b·δ). Simplest; but the clip
  biases the *effective* bias **low** once the field is non-linear (σ_cell≳1):
  at σ_cell≈1.8 a requested b=1.5 samples with cross-bias ≈0.6.
- ``"lognormal"``: λ = n̄_cell · exp(b·δ − b²σ²/2). Always positive (no clip),
  realistic 1-point PDF, and **preserves the cross-bias** ⟨δ_g·δ⟩/⟨δ²⟩ ≈ b
  (what the Wiener filter / constrained realization use) even in the non-linear
  regime. Its *auto* power is enhanced by the non-linear transform (a genuine
  lognormal feature, not a bug). Use this for absolute-power work.

counts ~ Poisson(λ); δ_g = counts/⟨counts⟩ − 1. (HOD — populate halos — is the
next, more physical mock; not implemented here.)
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def mock_tracer_field(delta, *, box_size, nbar, bias=1.0, seed=0,
                      mask: Optional[np.ndarray] = None,
                      model: str = "clip") -> Dict[str, np.ndarray]:
    """Poisson-sample biased tracers from ``delta``; return counts + δ_g.

    Parameters
    ----------
    delta : (n,n,n) truth matter overdensity.
    box_size : Mpc/h.
    nbar : mean tracer number density (Mpc/h)^-3.
    bias : linear tracer bias b.
    mask : optional (n,n,n) selection in [0,1].
    model : ``"clip"`` (default) or ``"lognormal"`` (bias-preserving).
    """
    d = np.asarray(delta, dtype=np.float64)
    n = d.shape[0]
    v_cell = (box_size / n) ** 3
    nbar_cell = nbar * v_cell
    rng = np.random.default_rng(seed)
    if model == "lognormal":
        # λ = n̄·exp(bδ − b²σ²/2): mean n̄, always positive, cross-bias ≈ b.
        lam = nbar_cell * np.exp(bias * d - 0.5 * bias ** 2 * float(d.var()))
    elif model == "clip":
        lam = nbar_cell * np.clip(1.0 + bias * d, 0.0, None)
    else:
        raise ValueError(f"unknown model {model!r}; use 'clip' or 'lognormal'")
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
