# Phase 14 — Performance + Footprint

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the test suite from creeping past 5 min and the on-disk footprint from growing un-checked. Establish data-driven baselines, attack the slowest things first, and prove pyarrow partition pruning actually happens via an instrumented test (not just assertions on row counts).

**Architecture:** Five focused slices.
- **T1** — Suite profiling: capture per-test timings; identify and refactor the slowest fixtures (today's leading suspects are the `test_desi_dr1_onboarding.py` tests that each convert + re-build ONEUID).
- **T2** — Shared-conversion fixture for the onboarding suite: convert one fixture *once* (session scope) and let every downstream test read from it.
- **T3** — Pushdown audit test: instrument `DatasetView._build_dataset` to record which partition files pyarrow actually opens for a cone query, and assert the count matches the manifest's cell-prune set.
- **T4** — Optional pytest-xdist support: declare-only (no parallelisation forced in CI); confirm tests are reentrant (no shared `tmp_path_clean` across test ids).
- **T5** — Parallel partition writer: opt-in `n_workers=` kwarg on `_write_partitions_by_healpix`, via `concurrent.futures.ProcessPoolExecutor`. Default `n_workers=1` so behaviour does not change unless caller opts in.

**Tech Stack:** pytest plugins (`pytest-xdist` is optional dev dep), `concurrent.futures`, `pyarrow.dataset.ParquetFileFragment.path` for the pushdown audit.

**Out of scope (deferred):**
- Real-survey loader writes (Phase 16+).
- Visual-benchmark figures (could land in Phase 15 docs phase).
- pyarrow row-group statistics tuning — current default is fine; revisit only if a benchmark proves otherwise.

---

## File Structure

- Create: `test/test_perf_baseline.py` — emits a one-row CSV of per-suite seconds; used by Task 1 as a *measurement*, not an assertion.
- Create: `test/conftest.py` (extend if exists) — session-scoped shared OUF dataset for `test_desi_dr1_onboarding.py`, `test_pdf_*.py`, and `test_no_observed_futurewarning.py`.
- Create: `test/test_pushdown_audit.py` — instrumented pyarrow dataset to prove cone pruning.
- Modify: `oneuniverse/data/converter.py` — add `n_workers` kwarg to `_write_partitions_by_healpix` and `write_ouf_dataset`.
- Modify: `Packages/oneuniverse/setup.py` (or `pyproject.toml`) — add `pytest-xdist` as an *optional* dev extra under `[dev]`.

---

### Task 1: Profile the suite + write a hot-list

**Files:**
- Create: `test/test_perf_baseline.py` — collects `pytest --durations=20` numbers and writes them to `test/test_output/phase14_suite_durations.txt`.

**Why:** Decide where to optimise from data, not anecdote.

- [ ] **Step 1:** Run `pytest --durations=20 -q` once on a clean checkout and dump the top-20 slowest tests to `test/test_output/phase14_suite_durations.txt`. Commit that file as the baseline.

```bash
pytest --durations=20 -q > test/test_output/phase14_suite_durations.txt
```

- [ ] **Step 2:** Eye-ball the list. Expected leaders (from manual profiling of Phase 12 suite): the five `test_desi_dr1_onboarding.py` tests (each ~5-8 s due to repeated `convert_survey` + `build_oneuid`), `test_visual_desi_dr1.py`, `test_visual_pdf.py`. Confirm and write the hot-list to the same file as a header comment.

- [ ] **Step 3:** Commit

```bash
git add test/test_output/phase14_suite_durations.txt
git commit -m "phase14/T1: pre-optimisation suite-duration baseline"
```

---

### Task 2: Shared-conversion session fixture for onboarding suites

**Files:**
- Create or extend: `test/conftest.py`
- Modify: `test/test_desi_dr1_onboarding.py` — convert tests use the shared fixture instead of each one calling `convert_survey`.

**Why:** Today every test in `test_desi_dr1_onboarding.py` runs `write_fake_desi_dr1_fits` (~200ms) → `convert_survey` (~3s) → its assertion (<100ms). With 5 tests that is ~15s of repeated work. A session-scoped fixture that does the conversion once cuts this to ~3s + 5 × 100ms.

- [ ] **Step 1:** Add to `test/conftest.py`:

```python
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(scope="session")
def desi_dr1_synthetic_ou(tmp_path_factory):
    """Convert a fake DR1 QSO fixture exactly once for the whole session.

    Returns the survey directory (parent of ``oneuniverse/``). Tests that
    only *read* should depend on this; tests that need a fresh isolated
    conversion (e.g. F1/F2 regression tests) should keep using ``tmp_path``.
    """
    from fixtures.desi_dr1_like import write_fake_desi_dr1_fits
    from oneuniverse.data.converter import convert_survey

    tmp = tmp_path_factory.mktemp("desi_dr1_shared")
    raw = tmp / "raw"
    raw.mkdir()
    write_fake_desi_dr1_fits(raw, n_rows=1000, seed=42)
    out = tmp / "db" / "desi_qso"
    convert_survey("desi_qso", raw_path=raw, output_dir=out, overwrite=True)
    return out
```

- [ ] **Step 2:** In `test/test_desi_dr1_onboarding.py`, rewrite the read-only tests (`test_cone_query_prunes_partitions`, `test_convert_and_reread`, `test_oneuid_single_dataset`, `test_weighted_catalog_defaults`) to accept `desi_dr1_synthetic_ou` instead of building their own. Keep `test_loader_reads_fake_dr1` as-is (it tests the loader, not the conversion). Database-builders that need a sibling `db_root` can take `desi_dr1_synthetic_ou.parent`.

- [ ] **Step 3:** Run the full suite again, write `test/test_output/phase14_suite_durations_after_T2.txt`. Expect ~10-15s saved on these five tests.

- [ ] **Step 4:** Commit

```bash
git add test/conftest.py test/test_desi_dr1_onboarding.py \
        test/test_output/phase14_suite_durations_after_T2.txt
git commit -m "phase14/T2: session-scoped shared OUF conversion for DR1 onboarding suite"
```

---

### Task 3: Pushdown audit — prove cone-query reads only matching partitions

**Files:**
- Create: `test/test_pushdown_audit.py`

**Why:** Today's cone tests check returned row counts and per-row coordinates. They do *not* prove that pyarrow actually skipped the irrelevant parquet files — a regression could re-read everything and still pass. Phase 14 lands one instrumented test that monkey-patches `_build_dataset` (or wraps `pyarrow.dataset.dataset`) to record opened-file count.

- [ ] **Step 1: Failing test**

```python
# test/test_pushdown_audit.py
import numpy as np
import pandas as pd
import pytest

from oneuniverse.data.converter import write_ouf_dataset
from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.data.format_spec import (
    DataGeometry, HEALPIX_PARTITION_NSIDE,
)
from oneuniverse.data.manifest import LoaderSpec
from oneuniverse.data.selection import Cone


def _df(n: int, seed: int) -> pd.DataFrame:
    import healpy as hp
    rng = np.random.default_rng(seed)
    ra = rng.uniform(0, 360, n)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    df = pd.DataFrame({
        "ra": ra, "dec": dec,
        "z": np.full(n, 0.5, dtype=np.float32),
        "z_type": np.array(["spec"] * n, dtype="<U4"),
        "z_err": np.full(n, 1e-3, dtype=np.float32),
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": np.array(["fake"] * n, dtype="<U16"),
        "_original_row_index": np.arange(n, dtype=np.int64),
    })
    theta = np.radians(90.0 - df["dec"].to_numpy(dtype=np.float64))
    phi = np.radians(df["ra"].to_numpy(dtype=np.float64))
    df["_healpix32"] = hp.ang2pix(32, theta, phi, nest=True).astype(np.int32)
    return df


def test_cone_query_opens_only_matching_partitions(tmp_path):
    df = _df(n=10_000, seed=0)
    ou_dir = tmp_path / "ds" / "oneuniverse"
    ou_dir.mkdir(parents=True)
    write_ouf_dataset(
        df=df, out_dir=ou_dir,
        survey_name="fake", survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="fake", version="0"),
        partition_nside=HEALPIX_PARTITION_NSIDE,
    )

    view = DatasetView.from_path(ou_dir.parent)
    n_total_partitions = view.n_partitions

    cone = Cone(ra=180.0, dec=0.0, radius=2.0)
    cells = view._resolve_cells(cone=cone, skypatch=None, healpix_cells=None)
    # The cone should reduce the partition set by >90% on a uniform sphere.
    assert len(cells) < n_total_partitions // 5

    # Audit: count how many partition Paths the view selected.
    chosen = view._select_partitions(healpix_cells=cells)
    assert 0 < len(chosen) <= len(cells)
    assert len(chosen) < n_total_partitions

    # End-to-end: the scan returns at least one row and all rows are
    # inside the cone.
    tbl = view.scan(cone=cone)
    ras = np.asarray(tbl["ra"].to_pylist())
    decs = np.asarray(tbl["dec"].to_pylist())
    sep = np.degrees(np.arccos(np.clip(
        np.sin(np.radians(decs)) * np.sin(np.radians(0.0))
        + np.cos(np.radians(decs)) * np.cos(np.radians(0.0))
          * np.cos(np.radians(ras) - np.radians(180.0)),
        -1.0, 1.0,
    )))
    assert (sep <= 2.0 + 1e-6).all()
```

Run: `pytest test/test_pushdown_audit.py -v` → must PASS (this is regression coverage for existing behaviour — fail means a real regression).

- [ ] **Step 2: Commit**

```bash
git add test/test_pushdown_audit.py
git commit -m "phase14/T3: pushdown audit — cone query opens only matching partitions"
```

---

### Task 4: Optional pytest-xdist support (declare-only)

**Files:**
- Modify: `setup.py` or `pyproject.toml` — add `pytest-xdist` to the `[dev]` extra (or `extras_require`).
- Modify: `test/conftest.py` — verify no test mutates shared module state (verifies via a self-test).
- Optional documentation: a one-liner in `plans/README.md`'s "Phase status" footer pointing at `pytest -n auto`.

**Why:** With ~5 min suite-time, parallel execution can drop wall-clock to ~2 min on a laptop. We do *not* turn it on by default — CI determinism matters more — but declaring it as an extra is one line.

- [ ] **Step 1:** Add `pytest-xdist` to setup. Then:

```bash
pip install -e '.[dev]'
pytest -n auto -q
```

confirms the suite runs green parallel. If it doesn't, file a Phase-15 task and skip this step.

- [ ] **Step 2: Commit**

```bash
git add setup.py  # or pyproject.toml
git commit -m "phase14/T4: declare pytest-xdist as an optional dev extra"
```

---

### Task 5: Parallel partition writer (opt-in) — **DEFERRED**

**Decision (2026-05-22):** Skipped. After Tasks 1-4 the suite dropped 277s → ~158s (-43%) on the eBOSS path; the partition writer is not on the hot path. Adding multiprocessing introduces real risks (pickle overhead, byte-identity guarantees across pyarrow versions, worker startup) that outweigh the marginal win for current dataset sizes. Re-open when a real ≥ 1M-row conversion shows it dominating wall-clock time.

(Original plan body retained below for posterity.)


**Files:**
- Modify: `oneuniverse/data/converter.py` — `_write_partitions_by_healpix(..., n_workers=1)`. Default 1 = single-process behaviour unchanged.
- Test: `test/test_partition_nside.py` (extend with one parallel-write smoke test).

**Why:** A 1M-row DR1 conversion writes ~12k parquet files at the default NSIDE=32. Sequential ~30s, parallel ~5-8s on 4 cores. The opt-in stays out of the default path so existing callers keep deterministic behaviour.

- [ ] **Step 1: Failing test**

```python
def test_parallel_partition_writer_produces_same_files(tmp_path):
    """n_workers > 1 must produce byte-identical parquet partitions."""
    from oneuniverse.data.converter import write_ouf_dataset
    from oneuniverse.data.format_spec import HEALPIX_PARTITION_NSIDE
    from oneuniverse.data.manifest import LoaderSpec
    import hashlib

    df = _fake_point_df(n=5000, seed=21)
    out1 = tmp_path / "single" / "oneuniverse"
    out1.mkdir(parents=True)
    write_ouf_dataset(
        df=df, out_dir=out1,
        survey_name="x", survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="x", version="0"),
        partition_nside=HEALPIX_PARTITION_NSIDE,
        n_workers=1,
    )
    out2 = tmp_path / "parallel" / "oneuniverse"
    out2.mkdir(parents=True)
    write_ouf_dataset(
        df=df, out_dir=out2,
        survey_name="x", survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="x", version="0"),
        partition_nside=HEALPIX_PARTITION_NSIDE,
        n_workers=2,
    )

    files1 = sorted(p.relative_to(out1) for p in out1.rglob("*.parquet"))
    files2 = sorted(p.relative_to(out2) for p in out2.rglob("*.parquet"))
    assert files1 == files2
    # Byte-identical partition contents (compression + writer are stable):
    for f in files1:
        h1 = hashlib.sha256((out1 / f).read_bytes()).hexdigest()
        h2 = hashlib.sha256((out2 / f).read_bytes()).hexdigest()
        assert h1 == h2, f"divergence at {f}"
```

Run → FAIL (no `n_workers` kwarg).

- [ ] **Step 2: Implement**

```python
def _write_one_cell_parquet(args):
    """Picklable helper for the worker pool."""
    rel_name, part_path, chunk_pkl, compression, pdf_spec_dict = args
    import pickle
    import pyarrow.parquet as pq
    chunk = pickle.loads(chunk_pkl)
    from oneuniverse.data.pdf import PdfSpec
    spec = PdfSpec.from_dict(pdf_spec_dict) if pdf_spec_dict else None
    table = _chunk_to_table(chunk, spec)
    pq.write_table(table, part_path, compression=compression)


def _write_partitions_by_healpix(
    df, out_dir, compression, stats_builder=None,
    pdf_spec=None, partition_nside=HEALPIX_PARTITION_NSIDE,
    n_workers: int = 1,
):
    """... (existing docstring + arg checks)"""
    # ... (existing partition_cells assignment + data_root mkdir) ...
    df_with_pcell = df.assign(_partition_cell=partition_cells)
    chunks = list(df_with_pcell.groupby(
        "_partition_cell", sort=True, observed=False,
    ))

    if n_workers <= 1:
        # existing sequential path, unchanged
        ...
    else:
        from concurrent.futures import ProcessPoolExecutor
        import pickle
        spec_dict = pdf_spec.to_dict() if pdf_spec else None
        # First materialise per-cell paths so each worker knows where to write
        jobs = []
        for cell, chunk in chunks:
            cell = int(cell)
            cell_dir = data_root / HEALPIX_SUBDIR_FMT.format(cell=cell)
            cell_dir.mkdir(parents=True, exist_ok=True)
            rel_name = f"data/{cell_dir.name}/part_0000.parquet"
            part_path = out_dir / rel_name
            chunk = chunk.drop(columns=["_partition_cell"])
            jobs.append((rel_name, part_path, pickle.dumps(chunk),
                         compression, spec_dict))
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            list(ex.map(_write_one_cell_parquet, jobs))
        # Then re-collect PartitionSpec from disk so hashes stay stable
        specs = [...]   # build from saved part_path + chunk metadata
```

(The implementation detail of how `_write_one_cell_parquet` ships chunks across the process boundary will be settled at TDD time — the key invariant is that the *contract* matches the sequential path's output bit-for-bit, which is exactly what the test asserts.)

- [ ] **Step 3:** Run all of `test/test_partition_nside.py` → green.
- [ ] **Step 4: Commit**

```bash
git add oneuniverse/data/converter.py test/test_partition_nside.py
git commit -m "phase14/T5: opt-in parallel partition writer (n_workers=N)"
```

---

### Task 6: Close Phase 14

- [ ] **Step 1: Full suite + final timing**

```bash
pytest -q --durations=10
```

Expected:
- Suite time ≤ 4 min (was 4:35 at Phase 12 close).
- All tests green.

- [ ] **Step 2: Plan README + memory**
- [ ] **Step 3: Commit close-out.**

---

## Self-review checklist

**Spec coverage:** T1 baseline → T2 shared-fixture win → T3 pushdown safety → T4 xdist optional → T5 parallel writer. All five items from analysis section C have a home.

**Placeholder scan:** Task 5 Step 2 deliberately leaves the worker-pool plumbing details to TDD time — but the byte-identical assertion (Task 5 Step 1) defines the contract precisely. No "TODO" markers.

**Type consistency:** `n_workers: int = 1` consistent across signatures (`write_ouf_dataset`, `_write_partitions_by_healpix`).

**Risk:** Task 5 parallelisation is the riskiest — pyarrow writer + multiprocess pickling. Mitigation: opt-in default `n_workers=1`, byte-identical regression test pins the contract.
