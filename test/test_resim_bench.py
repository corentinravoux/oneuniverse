"""Buffer-convergence baseline — the reference a sCOLA solution must beat."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.resim.bench import (
    buffer_convergence,
    reference_inner,
    uncoupled_resim_fn,
)


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_buffer_baseline_monotone_and_nonbuffer_is_poor():
    c = _cosmo()
    kw = dict(box=200.0, n_grid=48, target_lo=75.0, target_side=50.0, seed=2,
              n_steps=15)
    ref = reference_inner(c, **kw)
    fn = uncoupled_resim_fn(c, **kw)
    curve = buffer_convergence(fn, ref, buffers=[0.0, 12.5, 37.5])
    corr = dict(curve)
    # buffering monotonically improves the inner-region agreement, and the
    # gain from the non-buffer (isolated target) case is substantial
    assert corr[0.0] < corr[12.5] < corr[37.5]
    assert corr[37.5] - corr[0.0] > 0.2    # a real buffer clearly helps
    assert corr[37.5] > 0.85               # well-buffered resim ~ full sim
