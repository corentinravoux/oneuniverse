"""TreePM-split resimulation — beats the uncoupled buffered baseline.

The working sCOLA-class solution: the same inner-region accuracy at a smaller
buffer, because the external tide is supplied by the full-box linear force.
"""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.simulation.resim.bench import reference_inner, uncoupled_resim_fn
from oneuniverse.simulation.resim.treepm import run_coupled_treepm


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


_KW = dict(box=256.0, n_grid=64, target_lo=96.0, target_side=64.0, seed=2,
           n_steps=18)


def _corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def test_treepm_beats_uncoupled_at_small_buffer():
    c = _cosmo()
    ref = reference_inner(c, **_KW)
    ic = generate_density_field(c, box_size=_KW["box"], n_grid=_KW["n_grid"],
                                z=0.0, seed=_KW["seed"])
    unc = uncoupled_resim_fn(c, **_KW)

    def treepm(buf):
        return run_coupled_treepm(c, ic, box=_KW["box"], n_grid=_KW["n_grid"],
                                  target_lo=_KW["target_lo"],
                                  target_side=_KW["target_side"], buffer=buf,
                                  z_start=9.0, z_end=0.0,
                                  n_steps=_KW["n_steps"])["inner"]

    u16 = _corr(unc(16.0), ref)
    t16 = _corr(treepm(16.0), ref)
    u32 = _corr(unc(32.0), ref)
    # TreePM beats the uncoupled run at the same buffer ...
    assert t16 > u16 + 0.1
    # ... and matches/exceeds the uncoupled run at TWICE the buffer
    assert t16 >= u32 - 0.02
