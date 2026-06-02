"""Phase S5 T6 — phase_space, gr_fields, checkpoints products."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear import generate_linear_sim
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.simulation.linear.gr_fields import laplacian, potential_field
from oneuniverse.simulation.linear.phase_space import phase_space_sheet
from oneuniverse.simulation.oufsim import SimStore, write_oufsim_store


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_phase_space_columns_and_count():
    ps = phase_space_sheet(_cosmo(), box_size=200.0, n_grid=16, z=0.0, seed=1)
    assert set(ps) == {"qx", "qy", "qz", "x", "y", "z", "vx", "vy", "vz"}
    assert len(ps["qx"]) == 16 ** 3


def test_potential_solves_poisson():
    c = _cosmo()
    d = generate_density_field(c, box_size=200.0, n_grid=32, z=0.0, seed=2)
    phi = potential_field(d, box_size=200.0)
    lap = laplacian(phi, box_size=200.0)
    # ∇²φ recovers δ (zero-mean)
    np.testing.assert_allclose(lap, d - d.mean(), atol=1e-6)


def test_all_products_in_store(tmp_path):
    native = generate_linear_sim(tmp_path / "n", _cosmo(), box_size=200.0,
                                 n_grid=16, redshifts=(0.0, 1.0), seed=1)
    store = write_oufsim_store(native, tmp_path / "s", sim_name="d")
    s = SimStore(store)
    for p in ("snapshots", "fields", "halos", "lightcone", "tree",
              "phase_space", "gr_fields", "checkpoints"):
        assert p in s.products, f"missing product {p}"
