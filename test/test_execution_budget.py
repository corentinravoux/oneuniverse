"""S17 T6 — budget -> batch derivation."""
import pytest

from oneuniverse.simulation.execution import ExecutionMode, ExecutionPlan


def test_batch_for_derives_from_budget():
    plan = ExecutionPlan(mode=ExecutionMode.SEQUENTIAL,
                         memory_budget_bytes=1_000_000)
    # 6 float64 cols = 48 bytes/row; budget/48 caps the batch
    n = plan.batch_for(bytes_per_row=48)
    assert 0 < n <= 1_000_000 // 48


def test_explicit_batch_rows_wins():
    plan = ExecutionPlan(mode=ExecutionMode.SEQUENTIAL,
                         memory_budget_bytes=1_000_000, batch_rows=512)
    assert plan.batch_for(bytes_per_row=48) == 512


def test_bytes_per_row_must_be_positive():
    plan = ExecutionPlan(mode=ExecutionMode.SEQUENTIAL,
                         memory_budget_bytes=1_000_000)
    with pytest.raises(ValueError):
        plan.batch_for(bytes_per_row=0)
