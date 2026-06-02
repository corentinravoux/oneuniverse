"""Phase S5 T7 — SimDatasetView streaming reads."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.execution import ExecutionMode, ExecutionPlan
from oneuniverse.simulation.linear import generate_linear_sim
from oneuniverse.simulation.oufsim import (
    SimDatasetView,
    SimStore,
    write_oufsim_store,
)
from oneuniverse.simulation.selectors import Cube


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def _store(tmp_path):
    native = generate_linear_sim(tmp_path / "n", _cosmo(), box_size=200.0,
                                 n_grid=32, redshifts=(0.0,), seed=2)
    return write_oufsim_store(native, tmp_path / "s", sim_name="d",
                              particle_chunk_nside=4)


def test_iter_box_matches_read_box(tmp_path):
    store = _store(tmp_path)
    view = SimDatasetView(store)
    cube = Cube(0, 120, 0, 120, 0, 120)
    batches = list(view.iter_box("snapshots", 0.0, cube, batch_rows=1000))
    x = np.concatenate([b["x"] for b in batches]) if batches else np.array([])
    ref = SimStore(store).read_box("snapshots", 0.0, cube)
    assert len(x) == len(ref["x"])
    np.testing.assert_allclose(np.sort(x), np.sort(ref["x"]))


def test_batches_respect_batch_rows(tmp_path):
    store = _store(tmp_path)
    view = SimDatasetView(store)
    cube = Cube(0, 200, 0, 200, 0, 200)        # whole box
    batches = list(view.iter_box("snapshots", 0.0, cube, batch_rows=500))
    assert len(batches) > 1
    assert all(len(b["x"]) <= 500 for b in batches)


def test_plan_supplies_batch_rows(tmp_path):
    store = _store(tmp_path)
    view = SimDatasetView(store)
    plan = ExecutionPlan(mode=ExecutionMode.SEQUENTIAL,
                         memory_budget_bytes=8 * 1024 ** 2, batch_rows=300)
    cube = Cube(0, 200, 0, 200, 0, 200)
    batches = list(view.iter_box("snapshots", 0.0, cube, plan=plan))
    assert all(len(b["x"]) <= 300 for b in batches)
