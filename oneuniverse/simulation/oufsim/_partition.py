"""Deterministic rank assignment for MPI-collective reads.

Each rank reads chunk i where ``i % size == rank`` — disjoint, complete, no
collective gather of bulk rows (Rule 3). Pure + unit-testable without mpi4py;
the MPI wiring (resolve rank/size from COMM_WORLD) lives in read.py behind an
import guard.
"""
from __future__ import annotations

from typing import List, Sequence


def partition_by_rank(items: Sequence, *, rank: int, size: int) -> List:
    if size <= 1:
        return list(items)
    return [it for i, it in enumerate(items) if i % size == rank]
