# Phase 3 — HEALPix Spatial Partitioning — Implementation Plan

> **For agentic workers:** Execute task-by-task. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Partition POINT datasets on disk by HEALPix cell (NSIDE=32,
NESTED) so `Cone` / `SkyPatch` queries prune to the covering cells —
10–100× speedup on spatial queries.

**Architecture:** POINT converter writes one Parquet file per
`_healpix32` cell under `{ouf}/data/healpix32={cell}/part_*.parquet`;
Manifest records HEALPix metadata on each `PartitionSpec`;
`DatasetView` translates `Cone`/`SkyPatch` to a list of covering cells
via `healpy` and filters partitions before opening. SIGHTLINE /
HEALPIX geometries keep their row-based partitioning.

**Tech Stack:** `healpy.pixelfunc` (ang2pix, query_disc, query_polygon),
`pyarrow.dataset`, existing `PartitionSpec` / `PartitionStats`.

**Context — what Phase 1+2 already delivered:**

- `_healpix32` is a required CORE column on every POINT row.
- `PartitionSpec` already has `stats.ra/dec/z min/max` for range
  pruning — we extend with an optional HEALPix cell id.
- `DatasetView._select_partitions()` does range pruning today; we
  add HEALPix-cell pruning orthogonally.

---

## File Structure

- Modify: `oneuniverse/data/format_spec.py`
  — add HEALPix layout constants (subdir name, NSIDE=32, NEST=True).
- Modify: `oneuniverse/data/manifest.py`
  — add `healpix_cell: Optional[int]` to `PartitionSpec`.
- Modify: `oneuniverse/data/converter.py`
  — new `_write_partitions_by_healpix()` path; `write_ouf_dataset()`
    dispatches on geometry + uses HEALPix layout for POINT.
- Modify: `oneuniverse/data/selection.py`
  — add `Cone.healpix_cells(nside, nest)` and
    `SkyPatch.healpix_cells(nside, nest)` returning `np.ndarray[int]`.
- Modify: `oneuniverse/data/dataset_view.py`
  — extend `_select_partitions()` with `healpix_cells` argument;
    `scan()` accepts `cone=`, `skypatch=` kwargs and translates.
- Create: `test/test_healpix_partitioning.py`
  — converter layout, Cone→cells, SkyPatch→cells, scan pruning.

---

## Task 1: HEALPix constants + PartitionSpec field

**Files:**
- Modify: `oneuniverse/data/format_spec.py`
- Modify: `oneuniverse/data/manifest.py`

- [ ] **Step 1: Add constants to `format_spec.py`**

```python
# HEALPix on-disk layout ---------------------------------------------
HEALPIX_PARTITION_NSIDE: int = 32
HEALPIX_PARTITION_NEST: bool = True
HEALPIX_SUBDIR_FMT: str = "healpix32={cell:05d}"  # under data/
```

- [ ] **Step 2: Extend `PartitionSpec` with optional cell id**

In `oneuniverse/data/manifest.py`, add field:

```python
@dataclass(frozen=True)
class PartitionSpec:
    name: str
    n_rows: int
    sha256: str
    size_bytes: int
    stats: PartitionStats = field(default_factory=PartitionStats)
    healpix_cell: Optional[int] = None  # NEW
```

Update `PartitionSpec.to_dict` / `from_dict` to round-trip the field
(emit only when not None to keep old manifests readable).

- [ ] **Step 3: Run full suite — should still be 145 green**

`python3 -m pytest test/ -q`

---

## Task 2: POINT converter writes per-HEALPix-cell partitions

**Files:**
- Modify: `oneuniverse/data/converter.py`

- [ ] **Step 1: Write failing test**

In `test/test_healpix_partitioning.py`:

```python
def test_point_converter_writes_healpix_layout(tmp_path_clean):
    df = _make_point_df(500)  # spans many cells
    ou_dir = _write(tmp_path_clean, df)
    # data/healpix32=NNNNN/part_0000.parquet structure
    data_dir = ou_dir / "data"
    assert data_dir.is_dir()
    cell_dirs = sorted(data_dir.glob("healpix32=*"))
    assert len(cell_dirs) > 1
    # each matches exactly the unique cells present in df
    observed = {int(d.name.split("=")[1]) for d in cell_dirs}
    expected = set(int(c) for c in df["_healpix32"].unique())
    assert observed == expected
```

- [ ] **Step 2: Add `_write_partitions_by_healpix()` helper**

```python
def _write_partitions_by_healpix(
    df: pd.DataFrame, out_dir: Path, compression: str,
    stats_builder=None,
) -> List[PartitionSpec]:
    from oneuniverse.data.format_spec import (
        HEALPIX_PARTITION_NSIDE, HEALPIX_SUBDIR_FMT,
    )
    if "_healpix32" not in df.columns:
        raise ValueError("POINT df missing required _healpix32 column")
    data_root = out_dir / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    specs: List[PartitionSpec] = []
    # group by cell; stable order for reproducible manifests
    for cell, chunk in df.groupby("_healpix32", sort=True):
        cell = int(cell)
        cell_dir = data_root / HEALPIX_SUBDIR_FMT.format(cell=cell)
        cell_dir.mkdir(parents=True, exist_ok=True)
        part_name = f"data/{cell_dir.name}/part_0000.parquet"
        part_path = out_dir / part_name
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        pq.write_table(table, part_path, compression=compression)
        stats = stats_builder(chunk) if stats_builder else PartitionStats()
        specs.append(PartitionSpec(
            name=part_name, n_rows=len(chunk),
            sha256=hash_file(part_path),
            size_bytes=part_path.stat().st_size,
            stats=stats, healpix_cell=cell,
        ))
    return specs
```

- [ ] **Step 3: Dispatch in `write_ouf_dataset()`**

Replace the single `_write_partitions()` call with:

```python
if geometry is DataGeometry.POINT:
    partitions = _write_partitions_by_healpix(
        df, out_dir, compression, stats_builder,
    )
else:
    partitions = _write_partitions(
        df, out_dir, partition_rows, compression, stats_builder,
    )
```

- [ ] **Step 4: Run test — should pass**

- [ ] **Step 5: Run full suite — fix any breakage**

Likely breakage: old `partition_rows` kwarg on POINT now ignored;
update any test that asserted row-based partition counts on POINT data.

---

## Task 3: `Cone.healpix_cells()` and `SkyPatch.healpix_cells()`

**Files:**
- Modify: `oneuniverse/data/selection.py`
- Modify: `test/test_healpix_partitioning.py`

- [ ] **Step 1: Write failing tests**

```python
def test_cone_healpix_cells_covers_centre():
    c = Cone(ra=180.0, dec=0.0, radius=1.0)
    cells = c.healpix_cells(nside=32, nest=True)
    # centre cell must be included
    centre = hp.ang2pix(32, np.radians(90.0), np.radians(180.0), nest=True)
    assert int(centre) in set(cells.tolist())

def test_skypatch_healpix_cells_nonempty():
    s = SkyPatch(ra_min=0, ra_max=10, dec_min=-5, dec_max=5)
    cells = s.healpix_cells(nside=32, nest=True)
    assert len(cells) > 0
```

- [ ] **Step 2: Implement `Cone.healpix_cells`**

```python
def healpix_cells(self, nside: int, nest: bool = True) -> np.ndarray:
    import healpy as hp
    vec = hp.ang2vec(np.radians(90.0 - self.dec), np.radians(self.ra))
    return hp.query_disc(
        nside, vec, radius=np.radians(self.radius),
        inclusive=True, nest=nest,
    ).astype(np.int64)
```

- [ ] **Step 3: Implement `SkyPatch.healpix_cells`**

Use `hp.query_polygon` on the four corners (handling wrap-around by
splitting into two polygons when `ra_min > ra_max`):

```python
def healpix_cells(self, nside: int, nest: bool = True) -> np.ndarray:
    import healpy as hp
    def polygon(ra_lo, ra_hi):
        corners = [
            (ra_lo, self.dec_min), (ra_hi, self.dec_min),
            (ra_hi, self.dec_max), (ra_lo, self.dec_max),
        ]
        vecs = np.array([
            hp.ang2vec(np.radians(90.0 - d), np.radians(r))
            for r, d in corners
        ])
        return hp.query_polygon(
            nside, vecs, inclusive=True, nest=nest,
        )
    if self.ra_min <= self.ra_max:
        cells = polygon(self.ra_min, self.ra_max)
    else:
        cells = np.concatenate([
            polygon(self.ra_min, 360.0),
            polygon(0.0, self.ra_max),
        ])
    return np.unique(cells).astype(np.int64)
```

- [ ] **Step 4: Run tests — should pass**

---

## Task 4: DatasetView prunes on HEALPix cells

**Files:**
- Modify: `oneuniverse/data/dataset_view.py`
- Modify: `test/test_healpix_partitioning.py`

- [ ] **Step 1: Write failing test**

```python
def test_dataset_view_scan_with_cone_prunes(tmp_path_clean):
    df = _make_point_df(2000, seed=42)
    ou_dir = _write(tmp_path_clean, df)
    view = DatasetView.from_ou_dir(ou_dir)
    # Cone around first row, 1 deg
    r0 = df.iloc[0]
    cone = Cone(ra=float(r0["ra"]), dec=float(r0["dec"]), radius=1.0)
    tbl = view.scan(cone=cone)
    # result is subset of cells covering the cone
    cells = set(cone.healpix_cells(
        HEALPIX_PARTITION_NSIDE, nest=HEALPIX_PARTITION_NEST,
    ).tolist())
    got_cells = {
        int(c) for c in tbl["_healpix32"].to_pylist()
    }
    assert got_cells.issubset(cells)
    assert tbl.num_rows > 0  # cone contains at least row 0
```

- [ ] **Step 2: Extend `_select_partitions`**

```python
def _select_partitions(
    self, *, ra_range=None, dec_range=None, z_range=None,
    healpix_cells: Optional[Iterable[int]] = None,
) -> List[PartitionSpec]:
    cell_filter = set(int(c) for c in healpix_cells) \
        if healpix_cells is not None else None
    keep = []
    for part in self.manifest.partitions:
        if cell_filter is not None and part.healpix_cell is not None:
            if part.healpix_cell not in cell_filter:
                continue
        if not _range_overlaps(ra_range, part.stats.ra_min, part.stats.ra_max):
            continue
        if not _range_overlaps(dec_range, part.stats.dec_min, part.stats.dec_max):
            continue
        if not _range_overlaps(z_range, part.stats.z_min, part.stats.z_max):
            continue
        keep.append(part)
    return keep
```

- [ ] **Step 3: `scan()` accepts `cone` / `skypatch` kwargs**

```python
def scan(
    self, columns=None, filter=None, *,
    ra_range=None, dec_range=None, z_range=None,
    cone: Optional["Cone"] = None,
    skypatch: Optional["SkyPatch"] = None,
    healpix_cells: Optional[Iterable[int]] = None,
) -> pa.Table:
    cells = self._resolve_cells(cone, skypatch, healpix_cells)
    partitions = self._select_partitions(
        ra_range=ra_range, dec_range=dec_range, z_range=z_range,
        healpix_cells=cells,
    )
    ...
```

`_resolve_cells` unions user-supplied `healpix_cells` with cells
derived from `cone` / `skypatch`, applying the pruning NSIDE from
manifest (`HEALPIX_PARTITION_NSIDE` for now — Phase 3 assumes all
POINT datasets use NSIDE=32).

Also extend the expression builder: if `cone` or `skypatch` supplied,
build a pyarrow filter on `ra`/`dec` so rows inside the cell boundary
but outside the exact cone get dropped (cell pruning is
cover-not-exact).

- [ ] **Step 4: Run tests — should pass**

---

## Task 5: Integration + full suite

- [ ] **Step 1: Update existing DatasetView tests**

The converter now groups by HEALPix, so `test_multi_partition_layout`
test (which asserted `n_partitions == 10` for a sorted-by-z split)
no longer holds. Rewrite to force a SIGHTLINE / HEALPIX geometry
test, or drop the row-based partition test and replace with a
healpix-cell-count test.

- [ ] **Step 2: Run full suite**

`python3 -m pytest test/ -q` — expect 150+ green.

- [ ] **Step 3: Update memory + plan status**

- `project_oneuniverse_stabilisation.md`: add Phase 3 complete entry.
- `plans/README.md`: Phase 3 status → complete.

---

## Deferred / out of scope for Phase 3

- HEALPix partitioning for SIGHTLINE / HEALPIX geometries (row-based
  remains fine — they are not spatial-query dominated).
- Storing pruning NSIDE in the Manifest `partitioning` spec (will
  land with Phase 6 housekeeping when `PartitioningSpec` gets filled
  in properly).
- Streaming scan (`iter_partial` over cells) — Phase 5.
