"""Phase S5 T3 — ExecutionPlan mode enforcement (Rule 5)."""
import pytest

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.execution import ExecutionMode, ExecutionPlan
from oneuniverse.simulation.linear import generate_linear_sim
from oneuniverse.simulation.oufsim import write_oufsim_store


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_gpu_request_refused(tmp_path):
    native = generate_linear_sim(tmp_path / "n", _cosmo(), box_size=150.0,
                                 n_grid=16, redshifts=(0.0,), seed=1)
    plan = ExecutionPlan(mode=ExecutionMode.GPU,
                         memory_budget_bytes=64 * 1024 ** 2)
    with pytest.raises(ValueError, match="GPU"):
        write_oufsim_store(native, tmp_path / "s", sim_name="d", plan=plan)


def test_sequential_plan_runs(tmp_path):
    native = generate_linear_sim(tmp_path / "n", _cosmo(), box_size=150.0,
                                 n_grid=16, redshifts=(0.0,), seed=1)
    plan = ExecutionPlan(mode=ExecutionMode.SEQUENTIAL,
                         memory_budget_bytes=64 * 1024 ** 2)
    store = write_oufsim_store(native, tmp_path / "s", sim_name="d", plan=plan)
    assert (store / "manifest.json").is_file()
