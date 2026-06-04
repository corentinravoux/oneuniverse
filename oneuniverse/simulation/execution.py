"""Execution model for heavy OUF-Sim steps.

Optimisation is load-bearing (Pillar-3 Rule 5): every heavy-memory /
heavy-CPU-time step runs sequential-streamed (bounded memory),
MPI-collective, or GPU. An :class:`ExecutionPlan` declares the mode +
a hard memory budget; the chunk size derives from the budget, never
"the whole snapshot". The MPI communicator is intentionally NOT stored
on the (frozen, serialisable) plan — it is passed at call time.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ExecutionMode(str, Enum):
    SEQUENTIAL = "sequential"   # streamed, bounded working set
    MPI = "mpi"                 # collective, per-rank-local
    GPU = "gpu"                 # device-resident, GPUDirect where possible


@dataclass(frozen=True)
class ExecutionPlan:
    """How a heavy step will run + its memory budget.

    Parameters
    ----------
    mode
        One of :class:`ExecutionMode`.
    memory_budget_bytes
        Hard cap on the per-process working set. Must be > 0.
    batch_rows
        Chunk size for SEQUENTIAL / GPU streaming. ``None`` = derive
        from ``memory_budget_bytes`` at call time. If given, must be > 0.
    device
        e.g. ``"cuda:0"`` for GPU mode; ``None`` otherwise.
    n_chunks_estimate
        Estimated number of chunks (for progress / scheduling).
    """

    mode: ExecutionMode
    memory_budget_bytes: int
    batch_rows: Optional[int] = None
    device: Optional[str] = None
    n_chunks_estimate: int = 0

    def __post_init__(self) -> None:
        if self.memory_budget_bytes <= 0:
            raise ValueError(
                f"ExecutionPlan.memory_budget_bytes must be > 0, "
                f"got {self.memory_budget_bytes!r}"
            )
        if self.batch_rows is not None and self.batch_rows <= 0:
            raise ValueError(
                f"ExecutionPlan.batch_rows must be > 0 or None, "
                f"got {self.batch_rows!r}"
            )

    def batch_for(self, bytes_per_row: int, *, safety: float = 0.5) -> int:
        """Rows per streamed batch under the memory budget.

        ``safety`` reserves headroom for transient copies (concat, masks).
        An explicit ``batch_rows`` overrides the derivation.
        """
        if bytes_per_row <= 0:
            raise ValueError(
                f"bytes_per_row must be > 0, got {bytes_per_row!r}")
        if self.batch_rows is not None:
            return self.batch_rows
        n = int(self.memory_budget_bytes * safety // bytes_per_row)
        return max(1, n)
