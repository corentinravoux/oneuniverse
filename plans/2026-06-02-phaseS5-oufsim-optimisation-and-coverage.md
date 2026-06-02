# Phase S5 — OUF-Sim optimisation + full product coverage

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the landed Phase-S4 OUF-Sim prototype (JSON manifest +
parquet/`.npy`-tile products + sidecar `_index.parquet` + `SimStore`
partial access + `LinearSimConverter`) and (a) remove the convert-path
optimisation hotspots found empirically on `linsim_demo`, and (b) extend
the dummy linear sim to emit **every** OUF-Sim product kind so the store
format is exercised across all cases.

**Architecture:** Keep the exact on-disk format already proven (it mirrors
OUF: `manifest.json` + pyarrow parquet + HEALPix-NEST sky partitions +
memmap tiles). Change only *how* it is produced: a bounded-memory
streaming bucket-chunker driven by `ExecutionPlan`, parallel/MPI partition
writes gated by `BackendCapabilities`, a fused index pass, and an optional
**wrap-don't-re-encode** projection. Then add converter + generator support
for the remaining `PRODUCT_KINDS` (`tree`, `phase_space`, `gr_fields`,
`checkpoints`/`ic_posterior`) using trivial linear-theory models.

**Tech Stack:** numpy, pyarrow, healpy, pyyaml (all present); optional
`mpi4py` (import-guarded). No real-simulation deps. **Rule 1:** no
`oneuniverse.data` / `combine` imports (lint guard already scans
`oneuniverse/simulation/` recursively).

**Empirical driver:** [`research/2026-06-02-oufsim-optimization-findings.md`](../research/2026-06-02-oufsim-optimization-findings.md)
— convert was 4.5 s / 375 MB peak; parquet write = 52%; global
`argsort` + full-column gather doubles memory; store re-encodes (578 MB >
479 MB native). Partial-access reads already prune 64× and are **not** a
bottleneck.

---

## What already exists (Phase-S4 prototype — do not rebuild)

- `oneuniverse/simulation/oufsim/_io.py` — atomic JSON/bytes writers.
- `oneuniverse/simulation/oufsim/index.py` — Layer-1 toolkit:
  `cartesian_chunk_ids`, `chunk_coords`, `bbox_of`, `cube_overlaps_bbox`,
  `tile_specs`, `tile_overlaps_cube`, `healpix_partition_ids`,
  `cone_partition_pixels`, `skypatch_partition_pixels`.
- `oneuniverse/simulation/oufsim/write.py` — `write_oufsim_store` +
  per-product writers (`_write_chunked_catalog`, `_write_field_tiles`,
  `_write_lightcone`).
- `oneuniverse/simulation/oufsim/read.py` — `SimStore` (`read_box`,
  `read_field_box`, `read_cone`, `last_read_stats`).
- `oneuniverse/simulation/linear/converter.py` — `LinearSimConverter`
  (registered; real `convert()`).
- `oneuniverse/simulation/linear/lightcone.py` — toy lightcone product.
- `scripts/build_demo_oufsim.py` — demo + profiling + plots driver.

---

## File Structure (new / modified in S5)

- Modify: `oufsim/write.py` — streaming bucket-chunker, parallel writes,
  fused index, `ExecutionPlan` parameter, `projection` modes.
- Create: `oufsim/_parallel.py` — thread / MPI map helper (import-guarded).
- Create: `oufsim/view.py` — `SimDatasetView` typed streaming reads.
- Create: `linear/tree.py` — toy merger tree (`tree` product).
- Create: `linear/phase_space.py` — Lagrangian→Eulerian sheet
  (`phase_space` product).
- Create: `linear/gr_fields.py` — toy potential φ via ∇²φ = δ
  (`gr_fields` product) + IC checkpoint (`checkpoints`/`ic_posterior`).
- Modify: `linear/converter.py` + `oufsim/write.py` — declare + write the
  new products; declare `heavy_step_modes`.
- Tests under `test/`: one per task (below).

---

## Pre-flight

- [ ] **Step 0: Baseline green.**

```bash
cd /home/ravoux/Documents/Python/Packages/oneuniverse
pytest test/test_sim_*.py test/test_lin_*.py test/test_oufsim_*.py -q 2>&1 | tail -3
```

Expected: all pass (Phase S2+S3+S4 prototype).

---

## Task 1: Bounded-memory streaming bucket-chunker

Replace the global `argsort` + full-column copy with a counting/bucket
pass that writes one chunk at a time within a memory budget. Output store
must be **byte-identical** in content to the current writer.

**Files:** Modify `oufsim/write.py`; Test `test/test_oufsim_streaming.py`.

- [ ] **Step 1: Write the failing test**

```python
# test/test_oufsim_streaming.py
"""Phase S5 T1 — streaming bucket chunker matches the sorted writer."""
import tracemalloc

import numpy as np
import pyarrow.parquet as pq

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear import generate_linear_sim
from oneuniverse.simulation.oufsim import SimStore, write_oufsim_store
from oneuniverse.simulation.selectors import Cube


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def _rowcount(store, product, zt):
    import json
    info = json.load(open(store / "manifest.json"))["store_layout"]
    return info[product][zt]["n_rows"]


def test_streaming_preserves_rows_and_bbox(tmp_path):
    native = generate_linear_sim(tmp_path / "n", _cosmo(), box_size=200.0,
                                 n_grid=32, redshifts=(0.0,), seed=2)
    store = write_oufsim_store(native, tmp_path / "s", sim_name="d",
                               particle_chunk_nside=4)
    s = SimStore(store)
    cube = Cube(0, 50, 0, 50, 0, 50)
    sel = s.read_box("snapshots", 0.0, cube)
    # all points inside, and total rows preserved across all chunks
    assert sel["x"].max() <= 50.0
    assert _rowcount(store, "snapshots", "z0.000") == 32 ** 3
```

- [ ] **Step 2: Run to verify it fails / passes against current behaviour**

Run: `pytest test/test_oufsim_streaming.py -v` — should pass with the
current writer (it is a characterisation test). Then add the memory
assertion below that *fails* until the streaming writer lands.

```python
def test_peak_memory_is_bounded(tmp_path):
    native = generate_linear_sim(tmp_path / "n", _cosmo(), box_size=300.0,
                                 n_grid=64, redshifts=(0.0,), seed=2)
    tracemalloc.start()
    write_oufsim_store(native, tmp_path / "s", sim_name="d",
                       particle_chunk_nside=4, batch_rows=200_000)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # 64^3 particles * 6 float64 = ~12.5 MB/col-set; sorted copy would
    # ~2x the whole snapshot. Streaming must stay well under the full
    # sorted-copy ceiling.
    assert peak < 80 * 1e6
```

Run: `pytest test/test_oufsim_streaming.py::test_peak_memory_is_bounded -v`
Expected: FAIL (current writer copies all columns; `batch_rows` unknown).

- [ ] **Step 3: Implement the streaming chunker**

In `oufsim/write.py`, add a `batch_rows: Optional[int] = None` parameter to
`write_oufsim_store` and `_write_chunked_catalog`. Replace the global
`argsort` + dictcomp with a two-pass bucket:

```python
def _write_chunked_catalog(prod_dir, columns, pos, box_size, n_side,
                           batch_rows=None):
    prod_dir.mkdir(parents=True, exist_ok=True)
    n = len(pos)
    chunk_ids = cartesian_chunk_ids(pos, box_size, n_side)
    # Pass 1: counts + fused bbox per chunk (single min/max accumulation).
    n_chunks = n_side ** 3
    counts = np.bincount(chunk_ids, minlength=n_chunks)
    lo = np.full((n_chunks, 3), np.inf)
    hi = np.full((n_chunks, 3), -np.inf)
    np.minimum.at(lo, chunk_ids, pos)
    np.maximum.at(hi, chunk_ids, pos)
    # Pass 2: per-chunk write; gather rows for one chunk at a time.
    rows = []
    for cid in np.nonzero(counts)[0]:
        mask = chunk_ids == cid
        table = pa.table({k: v[mask] for k, v in columns.items()})
        fname = f"part_{int(cid):04d}.parquet"
        pq.write_table(prod_dir / fname, ...)  # see existing call
        cx, cy, cz = chunk_coords(int(cid), n_side)
        rows.append({"chunk_id": int(cid), "cx": cx, "cy": cy, "cz": cz,
                     "xlo": lo[cid, 0], "xhi": hi[cid, 0],
                     "ylo": lo[cid, 1], "yhi": hi[cid, 1],
                     "zlo": lo[cid, 2], "zhi": hi[cid, 2],
                     "n_rows": int(counts[cid]), "file": fname})
    _write_index(prod_dir / INDEX_FILE, rows)
    return {"partition": "cartesian_chunk", "n_side": int(n_side),
            "n_chunks": int(np.count_nonzero(counts)), "n_rows": int(n)}
```

This removes the full sorted copy (no `argsort`, no `{k: v[order]}` over the
whole array); `np.minimum.at`/`np.maximum.at` fuse the bbox into one pass
(findings hotspots #3, #4, #5). `batch_rows` reserved for the MPI path
(Task 2) to cap the per-rank working set.

- [ ] **Step 4: Run tests**

Run: `pytest test/test_oufsim_streaming.py -v`
Expected: both pass; peak memory under the ceiling.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/simulation/oufsim/write.py test/test_oufsim_streaming.py
git commit -m "phaseS5/T1: streaming bucket chunker — drop global argsort+copy, fuse bbox; bounded memory"
```

---

## Task 2: Parallel / MPI partition writes (the 52% hotspot)

Partition writes are embarrassingly parallel per chunk. Add a map helper
that runs the per-chunk write across a thread pool (default) or MPI ranks
(when `mpi4py` present and the `ExecutionPlan` asks for it), gated by
`BackendCapabilities`.

**Files:** Create `oufsim/_parallel.py`; Modify `oufsim/write.py`,
`linear/converter.py`; Test `test/test_oufsim_parallel.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_oufsim_parallel.py
"""Phase S5 T2 — parallel chunk writes match serial output."""
from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.execution import ExecutionMode, ExecutionPlan
from oneuniverse.simulation.linear import generate_linear_sim
from oneuniverse.simulation.oufsim import SimStore, write_oufsim_store
from oneuniverse.simulation.selectors import Cube


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_threaded_write_matches_serial(tmp_path):
    native = generate_linear_sim(tmp_path / "n", _cosmo(), box_size=200.0,
                                 n_grid=32, redshifts=(0.0,), seed=5)
    serial = write_oufsim_store(native, tmp_path / "ser", sim_name="d")
    plan = ExecutionPlan(mode=ExecutionMode.SEQUENTIAL,
                         memory_budget_bytes=256 * 1024**2)
    par = write_oufsim_store(native, tmp_path / "par", sim_name="d",
                             plan=plan, n_threads=4)
    cube = Cube(0, 50, 0, 50, 0, 50)
    a = SimStore(serial).read_box("snapshots", 0.0, cube)
    b = SimStore(par).read_box("snapshots", 0.0, cube)
    assert len(a["x"]) == len(b["x"])
```

- [ ] **Step 2: Run — FAIL** (`plan` / `n_threads` unknown).

- [ ] **Step 3: Implement `_parallel.py` + wire it**

```python
# oneuniverse/simulation/oufsim/_parallel.py
"""Run a per-partition callable across threads or MPI ranks.

mpi4py is import-guarded: absent -> threaded fallback. The MPI path
assigns partition i to rank (i % size); each rank writes its own files
(no collective gather of bulk data), matching Rule 3 (MPI-collective /
GPU-direct reads first-class) on the write side.
"""
from concurrent.futures import ThreadPoolExecutor


def map_partitions(fn, items, *, n_threads=1, use_mpi=False):
    if use_mpi:
        try:
            from mpi4py import MPI
        except ImportError:
            use_mpi = False
        else:
            comm = MPI.COMM_WORLD
            rank, size = comm.Get_rank(), comm.Get_size()
            for i, it in enumerate(items):
                if i % size == rank:
                    fn(it)
            comm.Barrier()
            return
    if n_threads <= 1:
        for it in items:
            fn(it)
    else:
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            list(ex.map(fn, items))
```

In `write.py`, refactor the per-chunk body into a closure `write_one(cid)`
and dispatch via `map_partitions(...)`, choosing `use_mpi` from
`plan.mode == ExecutionMode.MPI` **and** `capabilities.supports(
"particle_chunking", ExecutionMode.MPI)`. Collect index rows thread-safely
(append under a lock, or return rows and merge after the map). Update
`LinearSimConverter.capabilities.heavy_step_modes` to declare
`particle_chunking`/`parquet_write`/`field_tiling` as
`(SEQUENTIAL, MPI)`.

- [ ] **Step 4: Run** `pytest test/test_oufsim_parallel.py -v` — PASS.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/simulation/oufsim/_parallel.py oneuniverse/simulation/oufsim/write.py oneuniverse/simulation/linear/converter.py test/test_oufsim_parallel.py
git commit -m "phaseS5/T2: parallel/MPI partition writes gated by BackendCapabilities (52% hotspot)"
```

---

## Task 3: ExecutionPlan enforcement (refuse unbounded fallback)

`write_oufsim_store` must accept an `ExecutionPlan` and, per Rule 5, refuse
a requested mode the backend cannot honour rather than silently degrading.

**Files:** Modify `oufsim/write.py`; Test `test/test_oufsim_execplan.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_oufsim_execplan.py
"""Phase S5 T3 — ExecutionPlan mode enforcement."""
import pytest

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.execution import ExecutionMode, ExecutionPlan
from oneuniverse.simulation.linear import generate_linear_sim
from oneuniverse.simulation.oufsim import write_oufsim_store


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_gpu_request_refused_when_unsupported(tmp_path):
    native = generate_linear_sim(tmp_path / "n", _cosmo(), box_size=150.0,
                                 n_grid=16, redshifts=(0.0,), seed=1)
    plan = ExecutionPlan(mode=ExecutionMode.GPU,
                         memory_budget_bytes=64 * 1024**2)
    with pytest.raises(ValueError, match="GPU"):
        write_oufsim_store(native, tmp_path / "s", sim_name="d", plan=plan)
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement** the capability check at the top of
`write_oufsim_store`: for each heavy step, if `plan.mode` not in
`LinearSimConverter.capabilities.modes_for(step)`, raise
`ValueError(f"{step}: mode {plan.mode} not supported by backend")`.

- [ ] **Step 4: Run — PASS.**

- [ ] **Step 5: Commit**
`git commit -m "phaseS5/T3: ExecutionPlan mode enforcement — refuse unsupported modes (Rule 5)"`

---

## Task 4: Wrap-don't-re-encode projection

Add `projection="reference"` that writes only the sidecar index + manifest
pointing at the **native** files (no parquet/tile copy), for backends with
random access. Demonstrate that the resulting store is ~index-sized and
reads resolve through to native arrays.

**Files:** Modify `oufsim/write.py`, `oufsim/read.py`; Test
`test/test_oufsim_reference.py`.

- [ ] **Step 1: Failing test** — assert `store_size < 0.2 * native_size`
  when `projection="reference"`, and that `read_box` still returns rows
  inside the cube (reader follows the index's `native_file` + slice).
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement.** In `reference` mode, the index rows carry
  `{native_file, row_start, row_stop}` (for the linear backend, a
  contiguous Lagrangian-grid slice per chunk after a one-time native
  re-order, or a per-chunk row-id list stored once). The manifest records
  `projection: "reference"`. `SimStore` branches on it to memmap the native
  `.npy`/parquet instead of a copied partition. Document that real backends
  (AbacusSummit/Gadget) use their native KD-tree/Hilbert ranges here.
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit**
`git commit -m "phaseS5/T4: wrap-don't-re-encode projection (reference) — index-only store over native files"`

---

## Task 5: Merger-tree product (`tree`)

Link halos across adjacent redshift snapshots by nearest-neighbour in
comoving position → a progenitor/descendant edge list. Covers
`PRODUCT_KINDS = "tree"`.

**Files:** Create `linear/tree.py`; Modify `linear/generate.py`,
`oufsim/write.py`, `linear/converter.py`; Test `test/test_lin_tree.py`,
extend `test/test_oufsim_store.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_lin_tree.py
"""Phase S5 T5 — toy merger tree."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear.gaussian_field import generate_density_field
from oneuniverse.simulation.linear.halos import find_peaks
from oneuniverse.simulation.linear.tree import build_merger_tree


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81)


def test_edges_link_adjacent_snapshots():
    c = _cosmo()
    halos = {}
    for z in (0.0, 1.0):
        d = generate_density_field(c, box_size=200.0, n_grid=32, z=z, seed=4)
        halos[z] = find_peaks(d, box_size=200.0, threshold=1.0)
    tree = build_merger_tree(halos, box_size=200.0)
    assert {"descendant_id", "progenitor_id", "z_desc", "z_prog"} \
        <= set(tree)
    assert np.all(tree["z_prog"] > tree["z_desc"])  # progenitor at higher z
```

- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** `build_merger_tree(halos_by_z, box_size)`:
  for each adjacent (z_desc < z_prog) pair, for every descendant halo find
  the nearest progenitor (periodic KD-tree via `scipy.spatial.cKDTree` or a
  numpy brute force for small N) → edge list dict. Add a `tree.parquet`
  native output in `generate_linear_sim`; add a `_write_tree` (single
  parquet + trivial index) and a `"tree"` product decl in the writer +
  converter.
- [ ] **Step 4: Run** `pytest test/test_lin_tree.py test/test_oufsim_store.py -v` — PASS.
- [ ] **Step 5: Commit**
`git commit -m "phaseS5/T5: toy merger-tree product (tree) — nearest-neighbour progenitor edges"`

---

## Task 6: Phase-space, GR-field, and checkpoint products

Cover the remaining `PRODUCT_KINDS` with trivial linear-theory models:

- **`phase_space`** — the Zel'dovich Lagrangian→Eulerian sheet: store
  `(q_x,q_y,q_z, x,y,z, vx,vy,vz)` (already computed in `zeldovich`),
  partitioned in Lagrangian cube-chunks. New `linear/phase_space.py`.
- **`gr_fields`** — toy peculiar potential φ from ∇²φ = δ solved in
  Fourier space (`φ_k = -δ_k / k²`); stored as a field/tile product tagged
  `gr_fields`. New `linear/gr_fields.py`.
- **`checkpoints` / `ic_posterior`** — a JSON "checkpoint" recording the
  generator seed + cosmology + box/grid (the reproducible IC), written as a
  small sidecar; demonstrates the differentiable-checkpoint / IC-posterior
  slot without running any sampler (Rule 4).

**Files:** Create `linear/phase_space.py`, `linear/gr_fields.py`; Modify
`linear/generate.py`, `oufsim/write.py`, `linear/converter.py`; Tests
`test/test_lin_phase_space.py`, `test/test_lin_gr_fields.py`, extend
`test/test_oufsim_store.py`.

- [ ] **Step 1–4 (per product):** failing test → run-fail → implement
  generator + native output + writer branch + product decl → run-pass.
  Each test asserts shape/columns + store round-trip + (φ) that
  `∇²φ ≈ δ` to FFT tolerance.
- [ ] **Step 5: Commit**
`git commit -m "phaseS5/T6: phase_space + gr_fields + checkpoint products — all PRODUCT_KINDS covered"`

---

## Task 7: `SimDatasetView` — typed streaming reads

Formalise `SimStore` into a `SimDatasetView` that returns **typed,
batched** iterators honouring `ExecutionPlan` (batch size from
`batch_rows`; MPI/GPU read hooks declared, threaded fallback). This task
delivers a **correct** streaming reader only — read-path *optimisation*
and benchmarking (column projection, predicate pushdown, index cache,
parallel reads, Morton order) are Phase S6
([`2026-06-02-phaseS6-oufsim-read-optimisation.md`](2026-06-02-phaseS6-oufsim-read-optimisation.md)).

**Files:** Create `oufsim/view.py`; Test `test/test_oufsim_view.py`.

- [ ] **Step 1: Failing test** — `SimDatasetView(store).iter_box(
  "snapshots", z, cube, plan)` yields batches whose concatenation equals
  `SimStore.read_box(...)`, and each batch ≤ `batch_rows`.
- [ ] **Step 2: Run — FAIL.**
- [ ] **Step 3: Implement** `SimDatasetView` over `SimStore`: resolve
  overlapping partitions from the index, then yield row batches capped at
  `plan.batch_rows`, reading partitions lazily (one parquet/tile at a
  time). Declare GPU/MPI read hooks (import-guarded) but default threaded.
- [ ] **Step 4: Run — PASS.**
- [ ] **Step 5: Commit**
`git commit -m "phaseS5/T7: SimDatasetView — typed ExecutionPlan-batched streaming reads"`

---

## Task 8: Re-run demo + plots + close-out

- [ ] **Step 1:** Extend `scripts/build_demo_oufsim.py` to (a) use the
  optimised streaming/parallel writer, (b) emit the new products, (c) add
  two plots: **convert peak-memory vs `batch_rows`** and **write wall-time
  vs `n_threads`** (the optimisation payoff). Regenerate the store at
  `/home/ravoux/Documents/Science/Cosmography/oneuniverse_simulation`.
- [ ] **Step 2:** Run the demo; confirm `convert_peak_mb` drops vs the
  S4 baseline (375 MB) and parallel write speeds up.
- [ ] **Step 3:** Update `RUN_SUMMARY.json` + `OPTIMIZATION_FINDINGS.md`
  with before/after numbers.
- [ ] **Step 4:** Visual diagnostic test (`test/test_visual_oufsim.py`):
  assert the new plots exist and are non-trivial.
- [ ] **Step 5:** Full suite green.

```bash
pytest -q 2>&1 | tail -3
```

- [ ] **Step 6:** Docs — update `CLAUDE.md` (oufsim bullet: optimisation +
  all products) + `plans/README.md` (S5 → complete) + append to memory
  `project_oneuniverse_stabilisation.md`.
- [ ] **Step 7: Commit**
`git commit -m "phaseS5/T8: optimised demo + before/after profiling + docs; OUF-Sim optimisation & full coverage complete"`

---

## Self-review checklist

- [ ] Convert peak memory bounded by `ExecutionPlan.memory_budget_bytes` /
      `batch_rows` (no global sorted copy).
- [ ] Partition writes run threaded/MPI; output identical to serial.
- [ ] Unsupported `ExecutionMode` is refused, never silently degraded.
- [ ] `projection="reference"` store is index-sized over native files.
- [ ] All `PRODUCT_KINDS` exercised by the linear backend
      (snapshots, fields, halos, lightcone, tree, phase_space, gr_fields,
      checkpoints/ic_posterior).
- [ ] `SimDatasetView` streams typed batches honouring the plan.
- [ ] Rule 1 guard green; no real-sim deps added (mpi4py import-guarded).
- [ ] Demo regenerated; before/after numbers recorded; plots non-trivial.

## Maps to pinned Pillar-3 rules

| Rule | Where satisfied |
|---|---|
| 1 — minimal coupling | guard stays green; `oufsim/` uses only numpy/pyarrow/healpy |
| 2 — partial access load-bearing | already proven (S4); S7 view streams it |
| 3 — MPI/GPU first-class | T2 MPI writes, T7 MPI/GPU read hooks |
| 4 — no mini-sim runs | checkpoint stores IC metadata only; no sampler |
| 5 — optimisation load-bearing | T1 memory bound, T2 parallel, T3 enforcement |
