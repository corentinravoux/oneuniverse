# Phase S6 — OUF-Sim read-path optimisation (benchmark + tests)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make OUF-Sim partial-access reads measurably faster and prove
each improvement with a reusable **benchmark harness** + regression tests
that assert the win (bytes read, row-groups scanned, wall time, index
re-reads). Builds on `SimStore` (S4) and `SimDatasetView` (S5); the demo
store at `…/oneuniverse_simulation/linsim_demo` is the benchmark fixture.

**Architecture:** Reads are already partition-pruned (S4 proved 64×). This
phase squeezes the *within-partition* and *I/O* costs: parquet column
projection + predicate pushdown (row-group skipping), an LRU index cache,
threaded parallel partition reads, optional space-filling-curve row order
for contiguous sub-cube ranges, and an import-guarded GPU-direct read hook.
Every lever ships with a benchmark and a test asserting it helps (or is at
worst neutral) and never changes results.

**Tech Stack:** pyarrow (`read_table(columns=, filters=)`, row-group
metadata), numpy, healpy; optional `cudf`/`kvikio` (import-guarded). No
real-sim deps. **Rule 1:** no `oneuniverse.data` / `combine` imports.

**Invariant for every task:** the optimised read returns **exactly the
same rows** as the S4/S5 baseline read. Correctness first, speed second.

---

## File Structure

- Create: `oufsim/bench.py` — `ReadBenchmark` dataclass + `measure_read()`.
- Modify: `oufsim/read.py` — `columns=`, predicate pushdown, index cache,
  parallel reads, GPU hook.
- Modify: `oufsim/view.py` (from S5) — propagate projection + parallelism.
- Modify: `oufsim/write.py` — optional `row_order="morton"` within chunks.
- Create: `scripts/bench_oufsim_reads.py` — benchmark driver + plots.
- Tests: `test/test_oufsim_read_*` (one per task) + `test/test_visual_oufsim_reads.py`.

## Pre-flight

- [ ] **Step 0: Baseline green.**

```bash
cd /home/ravoux/Documents/Python/Packages/oneuniverse
pytest test/test_oufsim_*.py -q 2>&1 | tail -3
```

Expected: all pass (S4 store + S5 view).

---

## Task 1: Read-benchmark harness

**Files:** Create `oufsim/bench.py`; Test `test/test_oufsim_read_bench.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_oufsim_read_bench.py
"""Phase S6 T1 — read benchmark harness."""
from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear import generate_linear_sim
from oneuniverse.simulation.oufsim import SimStore, write_oufsim_store
from oneuniverse.simulation.oufsim.bench import measure_read
from oneuniverse.simulation.selectors import Cube


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_measure_read_reports_fields(tmp_path):
    native = generate_linear_sim(tmp_path / "n", _cosmo(), box_size=200.0,
                                 n_grid=32, redshifts=(0.0,), seed=1)
    store = write_oufsim_store(native, tmp_path / "s", sim_name="d")
    s = SimStore(store)
    bm = measure_read(lambda: s.read_box("snapshots", 0.0,
                                         Cube(0, 50, 0, 50, 0, 50)))
    assert bm.wall_s >= 0.0
    assert bm.peak_bytes > 0
    assert bm.n_rows > 0
```

- [ ] **Step 2: Run — FAIL** (`bench` missing).

- [ ] **Step 3: Implement**

```python
# oneuniverse/simulation/oufsim/bench.py
"""Measure a read: wall time, peak memory, rows returned, partitions read."""
from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class ReadBenchmark:
    wall_s: float
    peak_bytes: int
    n_rows: int
    stats: Optional[dict] = None


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
```

- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `phaseS6/T1: read-benchmark harness (wall/peak/rows)`

---

## Task 2: Column projection

Read only requested columns → fewer bytes off disk + less memory.

**Files:** Modify `oufsim/read.py`; Test `test/test_oufsim_read_project.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_oufsim_read_project.py
"""Phase S6 T2 — column projection."""
import os

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear import generate_linear_sim
from oneuniverse.simulation.oufsim import SimStore, write_oufsim_store
from oneuniverse.simulation.selectors import Cube


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_projection_returns_only_requested(tmp_path):
    native = generate_linear_sim(tmp_path / "n", _cosmo(), box_size=200.0,
                                 n_grid=32, redshifts=(0.0,), seed=1)
    store = write_oufsim_store(native, tmp_path / "s", sim_name="d")
    s = SimStore(store)
    cube = Cube(0, 80, 0, 80, 0, 80)
    full = s.read_box("snapshots", 0.0, cube)
    proj = s.read_box("snapshots", 0.0, cube, columns=["x", "y", "z"])
    assert set(proj) == {"x", "y", "z"}
    # same rows, fewer columns
    assert len(proj["x"]) == len(full["x"])
```

- [ ] **Step 2: Run — FAIL** (`columns` unknown).
- [ ] **Step 3: Implement** — add `columns: Optional[Sequence[str]] = None`
  to `read_box`; pass to `pq.read_table(path, columns=...)` (always include
  x/y/z for the precise cube cut, then drop extras before returning if the
  caller didn't ask for them). Same for `read_cone` (keep lon/lat).
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `phaseS6/T2: column projection in read_box/read_cone`

---

## Task 3: Predicate pushdown (row-group skipping)

Pass the cube bounds as pyarrow `filters` so parquet row-group min/max
statistics prune rows *inside* a chunk before they hit Python.

**Files:** Modify `oufsim/write.py` (row-group sizing), `oufsim/read.py`;
Test `test/test_oufsim_read_pushdown.py`.

- [ ] **Step 1: Failing test** — write chunks with a modest
  `row_group_size`; assert that a small cube inside one chunk reads fewer
  parquet rows than the chunk holds (via
  `pq.ParquetFile(path).metadata` row-group selection) and that results
  equal the unfiltered-then-masked baseline.

```python
def test_pushdown_scans_fewer_rows(tmp_path):
    ...
    s = SimStore(store)
    small = s.read_box("snapshots", 0.0, Cube(0, 10, 0, 10, 0, 10))
    assert s.last_read_stats.get("rows_scanned", 0) \
        < s.last_read_stats["rows_in_touched_chunks"]
    # correctness vs brute force
    brute = s.read_box("snapshots", 0.0, Cube(0, 10, 0, 10, 0, 10),
                       pushdown=False)
    assert len(small["x"]) == len(brute["x"])
```

- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** — `pq.write_table(..., row_group_size=N)` in the
  writer; in `read_box`, build pyarrow `filters=[("x",">=",xlo),("x","<=",
  xhi),…]` and use `pq.read_table(path, filters=filters)` (or a
  `dataset.Scanner`). Record `rows_scanned` / `rows_in_touched_chunks` in
  `last_read_stats`. `pushdown=True` default; `pushdown=False` falls back to
  the brute path for the equality test.
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `phaseS6/T3: predicate pushdown — parquet row-group skipping inside chunks`

---

## Task 4: Index LRU cache

Avoid re-reading `_index.parquet` on every query.

**Files:** Modify `oufsim/read.py`; Test `test/test_oufsim_read_cache.py`.

- [ ] **Step 1: Failing test** — count physical reads of the index file
  (monkeypatch `pq.read_table` or wrap `_index_rows`); a second identical
  query reads the index **0** extra times.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** — memoise `_index_rows(rel_index)` per
  `SimStore` instance in a dict (or `functools.lru_cache` on a bound
  helper). Add `SimStore.clear_cache()`.
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `phaseS6/T4: per-store index cache (no repeated _index.parquet reads)`

---

## Task 5: Parallel partition reads

Overlapping chunks are independent I/O → read them on a thread pool.

**Files:** Modify `oufsim/read.py` (reuse `oufsim/_parallel.py` from S5);
Test `test/test_oufsim_read_parallel.py`.

- [ ] **Step 1: Failing test** — `read_box(..., n_threads=4)` returns
  identical rows (sorted compare) to `n_threads=1`.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** — gather the hit-chunk file list, read via
  `map_partitions(read_one, files, n_threads=...)` returning per-file
  tables, then concat. Deterministic order: sort results by chunk_id before
  concat so output is stable.
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `phaseS6/T5: threaded parallel partition reads (deterministic concat)`

---

## Task 6: Space-filling-curve row order (contiguous sub-cube ranges)

Order rows within each chunk by Morton (Z-order) key at write time so a
sub-cube maps to a near-contiguous row range → row-group pruning (T3) is
maximally effective. Write-side enabler, read-side benefit.

**Files:** Modify `oufsim/write.py`, add `oufsim/_morton.py`; Test
`test/test_oufsim_read_morton.py`.

- [ ] **Step 1: Failing test** — with `row_order="morton"`, a small sub-cube
  inside one chunk touches **fewer parquet row-groups** than with
  `row_order="none"` (compare `last_read_stats["row_groups_read"]`); rows
  identical either way.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** — `_morton.py` interleaves quantised x/y/z bits
  into a Z-order key; in `_write_chunked_catalog`, when
  `row_order="morton"`, sort each chunk's rows by the key before writing.
  Record `row_groups_read` in `last_read_stats`.
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `phaseS6/T6: Morton row order within chunks → fewer row-groups per sub-cube`

---

## Task 7: GPU-direct read hook (import-guarded)

Declare a GPU read path (cuDF / kvikio) behind the
`supports_gpu_direct` capability; absent libs → threaded fallback, never an
error.

**Files:** Modify `oufsim/read.py`; Test `test/test_oufsim_read_gpu.py`.

- [ ] **Step 1: Failing test** — `read_box(..., device="gpu")` returns the
  same rows as CPU on a machine without cuDF (fallback path), and sets
  `last_read_stats["device"]`.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** — `try: import cudf` guarded; if present and
  `device=="gpu"`, read into cuDF then `.to_pandas()/to_numpy()`; else log
  once + CPU path. Capability `supports_gpu_direct` stays False for the
  linear backend, so this only exercises the fallback in CI.
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit** `phaseS6/T7: import-guarded GPU-direct read hook with CPU fallback`

---

## Task 8: Benchmark suite + plots + close-out

**Files:** Create `scripts/bench_oufsim_reads.py`,
`test/test_visual_oufsim_reads.py`; Modify `CLAUDE.md`, `plans/README.md`,
memory.

- [ ] **Step 1:** `bench_oufsim_reads.py` runs the demo store through a
  query sweep and records, per optimisation, the `ReadBenchmark`:
  - cube-size sweep → partitions touched + wall time;
  - column subset → bytes/peak;
  - pushdown on/off → rows scanned;
  - threads sweep → wall time;
  - Morton on/off → row-groups read.
  Writes `READ_BENCHMARKS.md` + 3 plots (touched-vs-size,
  rows-scanned-vs-pushdown, speedup-vs-threads) into the demo `plots/`.
- [ ] **Step 2:** Run it; confirm each lever improves (or is neutral) and
  results match baseline.
- [ ] **Step 3:** Visual test asserts the new plots exist + are non-trivial.
- [ ] **Step 4:** Full suite green: `pytest -q 2>&1 | tail -3`.
- [ ] **Step 5:** Docs — `CLAUDE.md` (SimStore read options),
  `plans/README.md` (S6 → complete), memory append.
- [ ] **Step 6: Commit** `phaseS6/T8: read benchmark suite + plots + docs; read-path optimisation complete`

---

## Self-review checklist

- [ ] Every optimised read returns identical rows to the S4/S5 baseline.
- [ ] Column projection reduces bytes/peak; pushdown reduces rows scanned.
- [ ] Index cache eliminates repeated `_index.parquet` reads.
- [ ] Parallel reads are deterministic; Morton order cuts row-groups.
- [ ] GPU hook falls back cleanly with no GPU present.
- [ ] Benchmark suite records before/after numbers; plots non-trivial.
- [ ] Rule 1 guard green; optional deps import-guarded.

## Maps to pinned Pillar-3 rules

| Rule | Where |
|---|---|
| 2 — partial access load-bearing | pushdown + projection + Morton tighten it |
| 3 — MPI/GPU first-class | T5 threads, T7 GPU-direct hook |
| 5 — optimisation load-bearing | whole phase; every lever benchmarked + tested |
