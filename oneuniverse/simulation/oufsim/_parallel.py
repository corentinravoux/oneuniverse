"""Run a per-partition callable across threads or MPI ranks.

Partition writes are embarrassingly parallel (one file each). ``mpi4py`` is
import-guarded: absent → threaded fallback. The MPI path assigns partition
i to rank (i % size) — each rank writes its own files (no collective gather
of bulk data, Rule 3) — then ``allgather``s the small per-partition index
rows so every rank can write a consistent manifest.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Sequence


def map_partitions(fn: Callable, items: Sequence, *, n_threads: int = 1,
                   use_mpi: bool = False) -> List:
    """Apply ``fn`` to each item; return results in input order."""
    items = list(items)
    if use_mpi:
        try:
            from mpi4py import MPI
        except ImportError:
            use_mpi = False
        else:
            comm = MPI.COMM_WORLD
            rank, size = comm.Get_rank(), comm.Get_size()
            local = [(i, fn(it)) for i, it in enumerate(items)
                     if i % size == rank]
            gathered = comm.allgather(local)
            flat = [pair for sub in gathered for pair in sub]
            flat.sort(key=lambda p: p[0])
            return [r for _, r in flat]
    if n_threads <= 1:
        return [fn(it) for it in items]
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        return list(ex.map(fn, items))
