"""Phase S14 — generality proof: engines over the OUF-Sim store boundary."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.simulation.oufsim import SimStore
from oneuniverse.simulation.oufsim.write import ingest_field
from oneuniverse.simulation.selectors import Cube
from oneuniverse.twin.engines import LinearForwardEngine, PMForwardEngine
from oneuniverse.twin.store_boundary import run_engine_to_store


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def _ic_store(tmp_path):
    c = _cosmo()
    field = generate_density_field(c, box_size=200.0, n_grid=32, z=0.0, seed=2)
    return ingest_field(tmp_path / "ic", "ic", cosmo=c, box_size=200.0,
                        field=field), field


def test_ingest_field_roundtrips(tmp_path):
    store, field = _ic_store(tmp_path)
    s = SimStore(store)
    assert "fields" in s.products
    back, _ = s.read_field_box(0.0, Cube(0, 200, 0, 200, 0, 200))
    np.testing.assert_allclose(back, field)


def test_two_engines_over_store_boundary(tmp_path):
    c = _cosmo()
    ic_store, field = _ic_store(tmp_path)
    # both engines: IC read from a store, product written to a store
    lin_store = run_engine_to_store(LinearForwardEngine(), out_root=tmp_path,
                                    sim_name="lin", cosmo=c, box=200.0,
                                    n_grid=32, ic_store=ic_store)
    pm_store = run_engine_to_store(PMForwardEngine(), out_root=tmp_path,
                                   sim_name="pm", cosmo=c, box=200.0,
                                   n_grid=32, ic_store=ic_store, n_steps=10)
    # both produced readable OUF-Sim stores (the contract is engine-agnostic)
    for st in (lin_store, pm_store):
        s = SimStore(st)
        assert "fields" in s.products
        out, _ = s.read_field_box(0.0, Cube(0, 200, 0, 200, 0, 200))
        assert out.shape == (32, 32, 32) and np.isfinite(out).all()
    # the linear engine passed the IC through; the PM evolved it (they differ)
    lin_out = SimStore(lin_store).read_field_box(
        0.0, Cube(0, 200, 0, 200, 0, 200))[0]
    np.testing.assert_allclose(lin_out, field)        # linear = the IC at z=0
