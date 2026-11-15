# Phase 5 — Streaming hydration Implementation Plan

> **For agentic workers:** Execute task-by-task. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Let users hydrate arbitrarily many ONEUIDs without blowing
up RAM by introducing a generator API
(`OneuidQuery.iter_partial(uids, columns, batch_size)`) that streams
rows in fixed-size batches. `partial_for` / `full_for` become thin
list-collecting wrappers.

**Architecture:** Chunk the ONEUID list into `batch_size` groups,
resolve each batch against the index, and delegate per-dataset row
selection to `DatasetView.scan` with a
`pc.field("_original_row_index").isin(...)` pushdown filter. One batch
lives in memory at a time.

**Tech Stack:** unchanged from Phase 4 (pyarrow, pandas, numpy).

**Context — what Phase 1–4 already delivered:**

- `_original_row_index` is a required CORE column on every POINT
  dataset, present in every converted partition.
- `DatasetView.scan(filter=pc.Expression)` already pushes arbitrary
  predicates into the Parquet reader and prunes partitions by manifest
  stats.
- `OneuidQuery.partial_for` currently reads the whole Parquet dataset
  then slices in pandas — this is what we're fixing.

---

## File Structure

- Modify: `oneuniverse/data/oneuid.py`
  — `OneuidQuery.iter_partial(uids, columns, datasets, batch_size)`
    generator; `partial_for` becomes a wrapper that concatenates what
    `iter_partial` yields; internal helper
    `_hydrate_batch(sub_index_table, columns, datasets)`.
- Create: `test/test_oneuid_streaming.py`
  — streaming generator contract + equivalence with legacy
    `partial_for`.

---

## Task 1: Row-level pushdown via DatasetView

**Files:**
- Modify: `oneuniverse/data/oneuid.py`

- [ ] **Step 1: Write failing test** (`test/test_oneuid_streaming.py`)

```python
from oneuniverse.data.oneuid import OneuidQuery

def test_partial_for_reads_only_requested_rows(two_overlapping_databases):
    """partial_for uses _original_row_index pushdown so huge datasets
    don't have to be fully materialised."""
    db = two_overlapping_databases
    db.build_oneuid(sky_tol_arcsec=2.0, dz_tol=1e-3)
    q = db.oneuid_query()
    uid = int(q.index.table["oneuid"].iloc[0])

    # Patch read_oneuniverse_parquet so the test fails if the old
    # "read whole file then slice" path is taken.
    from oneuniverse.data import oneuid as mod
    calls = []
    real = mod.read_oneuniverse_parquet
    def _spy(*a, **kw):
        calls.append((a, kw))
        return real(*a, **kw)
    mod.read_oneuniverse_parquet = _spy
    try:
        df = q.partial_for([uid], columns=["value"])
    finally:
        mod.read_oneuniverse_parquet = real

    # Pushdown path doesn't use the legacy helper.
    assert calls == []
    assert len(df) >= 1 and "value" in df.columns
```

- [ ] **Step 2: Add `_hydrate_batch` helper**

```python
def _hydrate_batch(
    self,
    sub: pd.DataFrame,
    columns: Optional[Sequence[str]],
    datasets: Optional[Sequence[str]],
) -> pd.DataFrame:
    """Hydrate the (dataset, row_index) pairs in *sub* into one frame.

    Uses ``DatasetView.scan`` with a ``_original_row_index`` pushdown
    filter so we never materialise rows we don't need.
    """
    import pyarrow.compute as pc

    if datasets is not None:
        sub = sub[sub["dataset"].isin(datasets)]
    if sub.empty:
        extra = list(columns) if columns is not None else []
        return pd.DataFrame(columns=[
            "oneuid", "dataset", "row_index", "ra", "dec", "z", *extra,
        ])

    col_arg = list(columns) if columns is not None else None
    pieces: List[pd.DataFrame] = []
    for ds_name, grp in sub.groupby("dataset", sort=False):
        view = self.database[ds_name]
        available = {c for c in view.columns}
        want = (
            [c for c in col_arg if c in available] if col_arg is not None
            else None
        )
        # Always need _original_row_index to re-order the returned table.
        if want is not None and "_original_row_index" not in want:
            want = want + ["_original_row_index"]
        rows = grp["row_index"].to_numpy()
        expr = pc.field("_original_row_index").isin(pa.array(rows))
        tbl = view.scan(columns=want, filter=expr)
        part = tbl.to_pandas()
        # Re-order to match grp's order using _original_row_index.
        part = part.set_index("_original_row_index").loc[rows].reset_index(drop=True)
        part = part.drop(
            columns=[c for c in ("ra", "dec", "z") if c in part.columns],
            errors="ignore",
        )
        if col_arg is not None:
            for c in col_arg:
                if c not in part.columns and c not in ("ra", "dec", "z"):
                    part[c] = np.nan
        part.insert(0, "oneuid", grp["oneuid"].to_numpy())
        part.insert(1, "dataset", ds_name)
        part.insert(2, "row_index", grp["row_index"].to_numpy())
        part.insert(3, "ra", grp["ra"].to_numpy())
        part.insert(4, "dec", grp["dec"].to_numpy())
        part.insert(5, "z", grp["z"].to_numpy())
        pieces.append(part)

    return pd.concat(pieces, ignore_index=True)
```

- [ ] **Step 3: Rewire `partial_for` to call `_hydrate_batch`**

```python
def partial_for(
    self,
    oneuids: Iterable[int],
    columns: Optional[Sequence[str]] = None,
    datasets: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    uids = np.asarray(list(oneuids), dtype=np.int64)
    sub = self.index.table[np.isin(self._uid_arr, uids)]
    return self._hydrate_batch(sub, columns, datasets)
```

- [ ] **Step 4: Add `pa` import** at top of `oneuid.py`
  (`import pyarrow as pa`).

- [ ] **Step 5: Run tests**

```bash
python3 -m pytest test/test_oneuid_streaming.py::test_partial_for_reads_only_requested_rows -v
python3 -m pytest test/test_oneuid.py -q
```

Expected: all tests green; the new test passes; the existing
`test_partial_for_*` tests keep passing (same outputs, just faster).

- [ ] **Step 6: Commit**

```bash
git add oneuniverse/data/oneuid.py test/test_oneuid_streaming.py
git commit -m "phase5: row-level pushdown via _original_row_index"
```

---

## Task 2: `iter_partial` streaming generator

**Files:**
- Modify: `oneuniverse/data/oneuid.py`
- Modify: `test/test_oneuid_streaming.py`

- [ ] **Step 1: Write failing tests**

```python
def test_iter_partial_yields_batches(two_overlapping_databases):
    db = two_overlapping_databases
    db.build_oneuid(sky_tol_arcsec=2.0, dz_tol=1e-3)
    q = db.oneuid_query()
    uids = q.index.table["oneuid"].unique().tolist()
    batches = list(q.iter_partial(uids, columns=["value"], batch_size=2))
    # With 5 unique ONEUIDs and batch_size=2 → 3 batches (2, 2, 1 uids).
    assert len(batches) == 3
    assert all(isinstance(b, pd.DataFrame) for b in batches)
    assert all("value" in b.columns for b in batches)
    # Each batch's oneuids are a subset of the input and non-overlapping.
    seen = set()
    for b in batches:
        this = set(b["oneuid"])
        assert this.isdisjoint(seen)
        seen |= this
    assert seen == set(uids)


def test_iter_partial_matches_partial_for(two_overlapping_databases):
    db = two_overlapping_databases
    db.build_oneuid(sky_tol_arcsec=2.0, dz_tol=1e-3)
    q = db.oneuid_query()
    uids = q.index.table["oneuid"].unique().tolist()

    streamed = pd.concat(
        list(q.iter_partial(uids, columns=["value", "value_err"], batch_size=2)),
        ignore_index=True,
    )
    materialised = q.partial_for(uids, columns=["value", "value_err"])
    streamed_sorted = streamed.sort_values(["oneuid", "dataset"]).reset_index(drop=True)
    mat_sorted = materialised.sort_values(["oneuid", "dataset"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(streamed_sorted, mat_sorted, check_dtype=False)


def test_iter_partial_default_batch_size(two_overlapping_databases):
    """batch_size=None → single batch == partial_for."""
    db = two_overlapping_databases
    db.build_oneuid(sky_tol_arcsec=2.0, dz_tol=1e-3)
    q = db.oneuid_query()
    uids = q.index.table["oneuid"].unique().tolist()
    batches = list(q.iter_partial(uids, columns=["value"]))
    assert len(batches) == 1


def test_iter_partial_empty(two_overlapping_databases):
    db = two_overlapping_databases
    db.build_oneuid(sky_tol_arcsec=2.0, dz_tol=1e-3)
    q = db.oneuid_query()
    batches = list(q.iter_partial([], columns=["value"]))
    assert batches == []


def test_iter_partial_batch_size_validation(two_overlapping_databases):
    db = two_overlapping_databases
    db.build_oneuid(sky_tol_arcsec=2.0, dz_tol=1e-3)
    q = db.oneuid_query()
    with pytest.raises(ValueError, match="batch_size"):
        list(q.iter_partial([0], columns=["value"], batch_size=0))
```

- [ ] **Step 2: Implement `iter_partial`**

```python
def iter_partial(
    self,
    oneuids: Iterable[int],
    columns: Optional[Sequence[str]] = None,
    datasets: Optional[Sequence[str]] = None,
    batch_size: Optional[int] = None,
) -> Iterator[pd.DataFrame]:
    """Stream hydrated rows batch-by-batch.

    Yields one :class:`pandas.DataFrame` per batch of up to ``batch_size``
    ONEUIDs. Peak memory is proportional to ``batch_size``, not to
    ``len(oneuids)``. ``batch_size=None`` yields a single batch (useful
    for testing / short queries).
    """
    uids = np.asarray(list(oneuids), dtype=np.int64)
    if uids.size == 0:
        return
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive or None")
    step = batch_size if batch_size is not None else uids.size
    for start in range(0, uids.size, step):
        chunk = uids[start:start + step]
        sub = self.index.table[np.isin(self._uid_arr, chunk)]
        yield self._hydrate_batch(sub, columns, datasets)
```

- [ ] **Step 3: Update imports** — `Iterator` from `typing`.

- [ ] **Step 4: Reuse `iter_partial` in `partial_for`**

```python
def partial_for(
    self,
    oneuids: Iterable[int],
    columns: Optional[Sequence[str]] = None,
    datasets: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    uids = np.asarray(list(oneuids), dtype=np.int64)
    pieces = list(self.iter_partial(uids, columns=columns, datasets=datasets))
    if not pieces:
        extra = list(columns) if columns is not None else []
        return pd.DataFrame(columns=[
            "oneuid", "dataset", "row_index", "ra", "dec", "z", *extra,
        ])
    return pd.concat(pieces, ignore_index=True)
```

- [ ] **Step 5: Run tests**

```bash
python3 -m pytest test/test_oneuid_streaming.py -v
```

Expected: all streaming tests pass.

- [ ] **Step 6: Commit**

```bash
git add oneuniverse/data/oneuid.py test/test_oneuid_streaming.py
git commit -m "phase5: iter_partial streaming generator"
```

---

## Task 3: Bounded-memory scan test

**Files:**
- Modify: `test/test_oneuid_streaming.py`

- [ ] **Step 1: Write the test**

Show that a 10k-row synth dataset yields the same output whether
`batch_size=10` or `batch_size=10_000`, and that each batch's row
count is bounded by `batch_size × n_datasets`.

```python
def test_iter_partial_batch_bound(two_overlapping_databases):
    db = two_overlapping_databases
    db.build_oneuid(sky_tol_arcsec=2.0, dz_tol=1e-3)
    q = db.oneuid_query()
    uids = q.index.table["oneuid"].unique().tolist()
    n_ds = q.index.table["dataset"].nunique()

    for b in q.iter_partial(uids, columns=["value"], batch_size=2):
        assert len(b) <= 2 * n_ds
```

- [ ] **Step 2: Run + commit**

```bash
python3 -m pytest test/test_oneuid_streaming.py -v
git add test/test_oneuid_streaming.py
git commit -m "phase5: streaming batch-bound test"
```

---

## Task 4: Integration + full suite

- [ ] **Step 1: Run the full suite**

```bash
python3 -m pytest test/ -q
```

Expected: 195+ green (190 baseline + ≥5 new streaming tests).

- [ ] **Step 2: Update `plans/README.md` phase status** to
  `**complete (2026-04-16, XXX/XXX tests green)**`.

- [ ] **Step 3: Update the stabilisation memory entry** with a
  Phase 5 bullet.

---

## Deferred to Phase 6

- Deletion of legacy `load_universal` (it is already shadowed by
  `OneuidQuery.iter_partial` / `partial_for`).
- Categorical dtype on `dataset` column (minor perf; bundled with
  other housekeeping).
