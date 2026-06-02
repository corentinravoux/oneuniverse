"""Phase S5 T2 — parallel chunk writes match serial output."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear import generate_linear_sim
from oneuniverse.simulation.oufsim import SimStore, write_oufsim_store
from oneuniverse.simulation.oufsim._parallel import map_partitions
from oneuniverse.simulation.selectors import Cube


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_map_partitions_threaded_preserves_order():
    out = map_partitions(lambda x: x * x, list(range(10)), n_threads=4)
    assert out == [x * x for x in range(10)]


def test_threaded_write_matches_serial(tmp_path):
    native = generate_linear_sim(tmp_path / "n", _cosmo(), box_size=200.0,
                                 n_grid=32, redshifts=(0.0,), seed=5)
    serial = write_oufsim_store(native, tmp_path / "ser", sim_name="d")
    par = write_oufsim_store(native, tmp_path / "par", sim_name="d",
                             n_threads=4)
    cube = Cube(0, 60, 0, 60, 0, 60)
    a = SimStore(serial).read_box("snapshots", 0.0, cube)
    b = SimStore(par).read_box("snapshots", 0.0, cube)
    assert len(a["x"]) == len(b["x"])
    np.testing.assert_allclose(np.sort(a["x"]), np.sort(b["x"]))
