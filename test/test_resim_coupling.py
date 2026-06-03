"""Phase S8.4/S8.5 — buffer-region resimulation + Gate 2/3 convergence."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.resim.coupling import (
    full_target_slice,
    run_coupled,
    run_full_reference,
)
from oneuniverse.simulation.resim.verify import gate2_dynamical


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


BOX, N, TLO, TSIDE = 256.0, 64, 96.0, 64.0


def _ref(c):
    full = run_full_reference(c, box=BOX, n_grid=N, z_start=9.0, z_end=0.0,
                              seed=2, n_steps=15)
    return full_target_slice(full, box=BOX, n_grid=N, target_lo=TLO,
                             target_side=TSIDE)


def test_coupled_inner_shape():
    res = run_coupled(_cosmo(), box=BOX, n_grid=N, target_lo=TLO,
                      target_side=TSIDE, buffer=32.0, z_start=9.0, z_end=0.0,
                      seed=2, n_steps=15)
    nt = int(round(TSIDE / (BOX / N)))
    assert res["inner"].shape == (nt, nt, nt)


def test_gate2_passes_with_large_buffer():
    c = _cosmo()
    ref = _ref(c)
    res = run_coupled(c, box=BOX, n_grid=N, target_lo=TLO, target_side=TSIDE,
                      buffer=48.0, z_start=9.0, z_end=0.0, seed=2, n_steps=15)
    g2 = gate2_dynamical(res["inner"], ref, box_size=TSIDE)
    # large buffer -> inner region tracks the full-box reference on large scales
    assert g2["passed"] and g2["r_lowk"] > 0.8


def test_gate3_buffer_convergence():
    c = _cosmo()
    ref = _ref(c)
    small = run_coupled(c, box=BOX, n_grid=N, target_lo=TLO, target_side=TSIDE,
                        buffer=16.0, z_start=9.0, z_end=0.0, seed=2, n_steps=15)
    big = run_coupled(c, box=BOX, n_grid=N, target_lo=TLO, target_side=TSIDE,
                      buffer=48.0, z_start=9.0, z_end=0.0, seed=2, n_steps=15)
    cs = np.corrcoef(small["inner"].ravel(), ref.ravel())[0, 1]
    cb = np.corrcoef(big["inner"].ravel(), ref.ravel())[0, 1]
    assert cb > cs                      # bigger buffer -> better inner match
