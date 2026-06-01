"""Phase S2 T2 — ExecutionMode + ExecutionPlan."""
import pytest

from oneuniverse.simulation.execution import ExecutionMode, ExecutionPlan


def test_execution_modes():
    assert ExecutionMode.SEQUENTIAL.value == "sequential"
    assert ExecutionMode.MPI.value == "mpi"
    assert ExecutionMode.GPU.value == "gpu"


def test_plan_defaults():
    plan = ExecutionPlan(mode=ExecutionMode.SEQUENTIAL, memory_budget_bytes=4 * 1024**3)
    assert plan.batch_rows is None
    assert plan.device is None
    assert plan.n_chunks_estimate == 0


def test_plan_rejects_nonpositive_budget():
    with pytest.raises(ValueError, match="memory_budget_bytes"):
        ExecutionPlan(mode=ExecutionMode.SEQUENTIAL, memory_budget_bytes=0)


def test_plan_rejects_nonpositive_batch():
    with pytest.raises(ValueError, match="batch_rows"):
        ExecutionPlan(
            mode=ExecutionMode.GPU, memory_budget_bytes=1024, batch_rows=0,
        )


def test_plan_is_frozen():
    plan = ExecutionPlan(mode=ExecutionMode.MPI, memory_budget_bytes=1024)
    with pytest.raises(Exception):
        plan.mode = ExecutionMode.GPU  # type: ignore[misc]
