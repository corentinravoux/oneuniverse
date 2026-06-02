"""Measure a read: wall time, peak memory, rows returned.

Reusable harness so every read-optimisation lever (projection, pushdown,
cache, parallel, Morton) is benchmarked + regression-tested, not asserted.
"""
from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ReadBenchmark:
    wall_s: float
    peak_bytes: int
    n_rows: int


def measure_read(fn: Callable) -> ReadBenchmark:
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn()
    wall = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    n = 0
    if isinstance(out, dict) and out:
        n = len(next(iter(out.values())))
    return ReadBenchmark(round(wall, 5), int(peak), int(n))
