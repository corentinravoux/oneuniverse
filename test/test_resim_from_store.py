"""Phase S11 — resim consumes the parent IC from a SimStore (partial-access wiring)."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear import generate_linear_sim
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.simulation.oufsim import write_oufsim_store
from oneuniverse.simulation.resim.coupling import run_coupled, run_coupled_from_store


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_resim_from_store_matches_in_memory(tmp_path):
    c = _cosmo()
    box, n = 200.0, 48
    native = generate_linear_sim(tmp_path / "n", c, box_size=box, n_grid=n,
                                 redshifts=(0.0,), seed=2)
    store = write_oufsim_store(native, tmp_path / "s", sim_name="box")
    kw = dict(target_lo=75.0, target_side=50.0, buffer=25.0, z_start=9.0,
              z_end=0.0, n_steps=10)
    from_store = run_coupled_from_store(c, store, **kw)["inner"]
    in_mem = run_coupled(c, box=box, n_grid=n,
                         ic_field=generate_density_field(c, box_size=box,
                                                         n_grid=n, z=0.0,
                                                         seed=2), **kw)["inner"]
    # the resim reading its IC from the store reproduces the in-memory resim
    np.testing.assert_allclose(from_store, in_mem, atol=1e-6)
