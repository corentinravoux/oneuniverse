"""Execution plan: how much memory, how many threads, which parallel mode."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from oneuniverse._core.errors import ValidationError


class ExecutionMode(str, Enum):
    STREAMED = "streamed"
    MPI = "mpi"


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    mode: ExecutionMode = ExecutionMode.STREAMED
    memory_budget_bytes: int = 2**30
    mpi_comm: Any = None
    n_threads: int = 4

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "mode", ExecutionMode(self.mode))
        except ValueError:
            raise ValidationError(f"unknown execution mode {self.mode!r}") from None
        if int(self.memory_budget_bytes) != self.memory_budget_bytes or self.memory_budget_bytes <= 0:
            raise ValidationError(f"memory_budget_bytes must be a positive integer, got {self.memory_budget_bytes!r}")
        if int(self.n_threads) != self.n_threads or self.n_threads < 1:
            raise ValidationError(f"n_threads must be >= 1, got {self.n_threads!r}")
        if self.mode is ExecutionMode.MPI and self.mpi_comm is None:
            raise ValidationError("ExecutionMode.MPI requires mpi_comm")

    def batch_for(self, bytes_per_row: int, *, safety: float = 0.5) -> int:
        """Rows per batch fitting ``safety`` x the memory budget (at least 1)."""
        if bytes_per_row <= 0:
            raise ValidationError(f"bytes_per_row must be > 0, got {bytes_per_row}")
        if not 0 < safety <= 1:
            raise ValidationError(f"safety must be in (0, 1], got {safety}")
        return max(1, int(self.memory_budget_bytes * safety) // int(bytes_per_row))
