"""S17 T6 — streaming a big read honours the memory budget."""
from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.execution import ExecutionMode, ExecutionPlan
from oneuniverse.simulation.linear import generate_linear_sim
from oneuniverse.simulation.oufsim import (
    SimDatasetView, SimStore, write_oufsim_store)
from oneuniverse.simulation.selectors import Cube


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_streamed_batches_respect_budget(tmp_path):
    lin = generate_linear_sim(tmp_path / "lin", _cosmo(), box_size=300.0,
                              n_grid=48, redshifts=(0.0,), seed=2,
                              with_lightcone=False)
    store = write_oufsim_store(lin, tmp_path / "s", sim_name="d",
                               particle_chunk_nside=2)
    plan = ExecutionPlan(mode=ExecutionMode.SEQUENTIAL,
                         memory_budget_bytes=48 * 4096)   # ~2048 rows/batch
    view = SimDatasetView(store)
    cube = Cube(0, 300, 0, 300, 0, 300)                   # whole box
    sizes = [len(b["x"]) for b in view.iter_box("snapshots", 0.0, cube,
                                                 plan=plan)]
    full = SimStore(store).read_box("snapshots", 0.0, cube)
    assert sum(sizes) == len(full["x"])
    assert max(sizes) <= 4096                             # each batch bounded
