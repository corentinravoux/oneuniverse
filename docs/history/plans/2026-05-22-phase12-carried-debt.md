# Phase 12 — Carried-over Debt Cleanup

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land four concrete debt items carried over from Phases 6–11 + one latent partition-NSIDE bug surfaced during planning, so Pillar-1 enters the post-stabilisation era with no known correctness gaps in the core data path.

**Architecture:** Four orthogonal slices, each TDD-driven, each committable on its own.
- **F3** — Adaptive partition NSIDE in `write_ouf_dataset`: auto-coarsen for small catalogs to reduce file count + parquet-header overhead. Store the chosen NSIDE on `manifest.partitioning.extra["nside"]`.
- **D5** — `DatasetView._resolve_cells` must read NSIDE from the manifest, not from the global `HEALPIX_PARTITION_NSIDE` constant. Latent bug, becomes load-bearing once F3 lands.
- **D1** — Delete module-level `_data_root` state from `oneuniverse.data._config`. Internal callers (`database.build`, `convert_survey`) migrate to `env_data_root()` + explicit kwarg threading. Public `get_data_root` / `set_data_root` exports removed.
- **D2** — Pass `observed=False` to the one `groupby` on a Categorical (`oneuid.py:653`) to silence the pandas `FutureWarning` and pin behaviour against the upcoming default flip.
- **D3** — `convert_survey(loader=<instance>, ...)` overload so test/one-off conversions don't need `@register` or registry lookup. Existing `survey_name=` path unchanged.

**Tech Stack:** No new dependencies. Pure refactor + small API addition within `oneuniverse/data/`.

---

## File Structure

- Modify: `oneuniverse/data/converter.py` — F3 partition logic, D3 loader= overload, D1 removal of `set_data_root` import.
- Modify: `oneuniverse/data/dataset_view.py` — D5 manifest-driven NSIDE.
- Modify: `oneuniverse/data/_config.py` — D1 drop module state + deprecated wrappers.
- Modify: `oneuniverse/data/__init__.py`, `oneuniverse/__init__.py` — D1 remove public exports.
- Modify: `oneuniverse/data/database.py` — D1 migrate `build` away from `set_data_root`.
- Modify: `oneuniverse/data/oneuid.py` — D2 one-liner.
- Test: `test/test_partition_nside.py` (F3, D5).
- Test: `test/test_convert_survey_loader_kwarg.py` (D3).
- Test: `test/test_data_root_removed.py` (D1 — ensures import fails).
- Modify: `test/` — touch any tests that called `set_data_root` to switch to the kwarg path. Inventory listed in Task 4.

---

### Task 1: D2 — pandas `observed=False` one-liner

**Files:**
- Modify: `oneuniverse/data/oneuid.py:653`
- Test: existing `test/test_oneuid_streaming.py` already exercises this path; assertion is "zero `FutureWarning` from this line".

**Why:** Smallest, most isolated win. Lands the discipline of `-W error::FutureWarning` for at least one warning. The remaining suite-wide warning audit lives in Phase 15.

- [ ] **Step 1: Write the failing test**

```python
# test/test_no_observed_futurewarning.py  (new)
import warnings

def test_iter_partial_no_observed_futurewarning():
    """OneuidQuery.iter_partial must not emit pandas observed=… FutureWarning."""
    # Tiny end-to-end: convert a 2-row fixture, build oneuid, iter_partial,
    # filter warnings for the specific message.
    import numpy as np
    import pandas as pd
    from oneuniverse.data.converter import write_ouf_dataset
    from oneuniverse.data.format_spec import DataGeometry
    from oneuniverse.data.manifest import LoaderSpec
    from oneuniverse.data.database import OneuniverseDatabase
    from oneuniverse.data.oneuid_rules import CrossMatchRules

    # ... build a minimal POINT dataset with 4 rows, build oneuid, then:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # exercise the iter_partial / partial_for path
        ...
    bad = [w for w in caught
           if issubclass(w.category, FutureWarning)
           and "observed=" in str(w.message)]
    assert bad == [], [str(b.message) for b in bad]
```

(Full test body is filled in during Step 2 once we settle the minimum reproduction; do not commit a placeholder body.)

- [ ] **Step 2: Run** `pytest test/test_no_observed_futurewarning.py -v` → FAIL.

- [ ] **Step 3: Fix**

```python
# oneuniverse/data/oneuid.py:653
for ds_name, grp in sub.groupby("dataset", sort=False, observed=False):
```

`observed=False` is the **current** semantic — we are pinning behaviour, not changing it. The default will flip to `True` in a future pandas; revisit then.

- [ ] **Step 4: Run** `pytest test/test_no_observed_futurewarning.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/oneuid.py test/test_no_observed_futurewarning.py
git commit -m "phase12/D2: pin observed=False on oneuid groupby (silences pandas FW)"
```

---

### Task 2: D5 — Cone/SkyPatch resolution reads NSIDE from manifest

**Files:**
- Modify: `oneuniverse/data/dataset_view.py` — `_resolve_cells`
- Test: `test/test_partition_nside.py` (first half — D5 only; F3 lands in Task 3)

**Why:** `_resolve_cells` currently hardcodes `HEALPIX_PARTITION_NSIDE = 32`. Phase 12 F3 lets `write_ouf_dataset` pick a different NSIDE per dataset; if cone-query still uses 32, it pulls the wrong cell ids and silently misses partitions. Fix before F3 lands so we never ship a partial-coverage cone read.

- [ ] **Step 1: Write the failing test**

```python
# test/test_partition_nside.py  (new — D5 first)
import numpy as np
import pandas as pd
from pathlib import Path

from oneuniverse.data.converter import write_ouf_dataset
from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.manifest import LoaderSpec, PartitioningSpec
from oneuniverse.data.selection import Cone


def _materialise_at_nside(df, nside):
    """Stamp a `_healpix32` column with a non-default NSIDE encoded value."""
    import healpy as hp
    theta = np.radians(90.0 - df["dec"].to_numpy(dtype=np.float64))
    phi = np.radians(df["ra"].to_numpy(dtype=np.float64))
    df = df.copy()
    df["_healpix32"] = hp.ang2pix(nside, theta, phi, nest=True).astype(np.int32)
    return df


def test_cone_query_uses_manifest_nside(tmp_path):
    """Conversion at NSIDE=8 must still produce a correct cone result."""
    nside = 8                    # not the default 32
    rng = np.random.default_rng(0)
    n = 3000
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
    df = _materialise_at_nside(df, nside)

    ou_dir = tmp_path / "ds" / "oneuniverse"
    ou_dir.mkdir(parents=True)
    write_ouf_dataset(
        df=df, out_dir=ou_dir,
        survey_name="fake", survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="fake", version="0"),
        partitioning=PartitioningSpec(
            scheme="healpix", column="_healpix32",
            extra={"nside": nside, "nest": True},
        ),
    )

    view = DatasetView.from_path(ou_dir.parent)
    cone = Cone(ra=180.0, dec=0.0, radius=10.0)
    got = view.read(cone=cone, columns=["ra", "dec"])
    # Sanity: every returned row sits inside the cone (haversine).
    sep = _haversine_deg(got["ra"], got["dec"], 180.0, 0.0)
    assert (sep <= 10.0 + 1e-6).all()
    # And there is at least *one* row — proves we didn't read zero
    # partitions because we used the wrong NSIDE.
    assert len(got) > 0


def _haversine_deg(ra, dec, ra0, dec0):
    ra1 = np.radians(np.asarray(ra)); dec1 = np.radians(np.asarray(dec))
    ra2 = np.radians(ra0); dec2 = np.radians(dec0)
    return np.degrees(np.arccos(np.clip(
        np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2),
        -1.0, 1.0,
    )))
```

Run: `pytest test/test_partition_nside.py::test_cone_query_uses_manifest_nside -v` → FAIL (returns 0 rows because cone-cell ids use NSIDE=32, partitions stored at NSIDE=8).

- [ ] **Step 2: Fix `_resolve_cells`**

```python
# oneuniverse/data/dataset_view.py — replace the body of _resolve_cells
def _resolve_cells(
    self,
    cone: Optional[Cone],
    skypatch: Optional[SkyPatch],
    healpix_cells: Optional[Iterable[int]],
) -> Optional[List[int]]:
    if cone is None and skypatch is None and healpix_cells is None:
        return None

    # Read partitioning NSIDE/ordering from the manifest. Fall back to the
    # global default for legacy manifests that didn't record it.
    nside = HEALPIX_PARTITION_NSIDE
    nest = HEALPIX_PARTITION_NEST
    if self.manifest.partitioning is not None:
        nside = int(self.manifest.partitioning.extra.get("nside", nside))
        nest = bool(self.manifest.partitioning.extra.get("nest", nest))

    acc: set = set()
    if healpix_cells is not None:
        acc.update(int(c) for c in healpix_cells)
    if cone is not None:
        acc.update(int(c) for c in cone.healpix_cells(nside, nest=nest))
    if skypatch is not None:
        acc.update(int(c) for c in skypatch.healpix_cells(nside, nest=nest))
    return sorted(acc)
```

- [ ] **Step 3: Run** `pytest test/test_partition_nside.py::test_cone_query_uses_manifest_nside -v` → PASS.

- [ ] **Step 4: Commit**

```bash
git add oneuniverse/data/dataset_view.py test/test_partition_nside.py
git commit -m "phase12/D5: DatasetView reads partition NSIDE from manifest (was global)"
```

---

### Task 3: F3 — Adaptive partition NSIDE in `write_ouf_dataset`

**Files:**
- Modify: `oneuniverse/data/converter.py` — `write_ouf_dataset` + `_write_partitions_by_healpix` + `convert_survey`.
- Modify: `oneuniverse/data/format_spec.py` — add `MIN_ROWS_PER_PARTITION` constant.
- Test: `test/test_partition_nside.py` (extend).

**Why:** Current behaviour writes one parquet file per populated NSIDE=32 cell regardless of population. A 17 618-row catalog produces 2 551 files at 0.1× compression because parquet header dominates per-file. Fix: pick the *coarsest* NSIDE for which the mean rows-per-cell exceeds a target threshold (default 5 000), capped at NSIDE=32 (the canonical fine scale we always store the column at). The chosen NSIDE goes on `manifest.partitioning.extra["nside"]` — Task 2's D5 fix already routes cone queries through it.

**Design point:** The `_healpix32` column stays NSIDE=32 (column name + dtype unchanged → no schema break). The *partition key* is `_healpix32 >> (2 * level_diff)` for NEST ordering, computed on the fly when grouping. We do not rewrite the column.

- [ ] **Step 1: Failing test**

```python
# test/test_partition_nside.py — append
def test_small_catalog_uses_coarser_nside(tmp_path):
    """A 1k-row catalog must not produce hundreds of partition files."""
    import healpy as hp
    rng = np.random.default_rng(7)
    n = 1000
    df = pd.DataFrame({
        "ra": rng.uniform(0, 360, n),
        "dec": np.degrees(np.arcsin(rng.uniform(-1, 1, n))),
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

    ou_dir = tmp_path / "small" / "oneuniverse"
    ou_dir.mkdir(parents=True)
    from oneuniverse.data.manifest import LoaderSpec
    from oneuniverse.data.format_spec import DataGeometry
    from oneuniverse.data.converter import write_ouf_dataset

    manifest = write_ouf_dataset(
        df=df, out_dir=ou_dir,
        survey_name="small", survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="small", version="0"),
    )

    # Auto-picked NSIDE must coarsen far below 32 for 1k rows.
    chosen_nside = int(manifest.partitioning.extra["nside"])
    assert chosen_nside <= 4, f"got nside={chosen_nside}, expected coarsening"

    # File count must be small (≤ ~30, not 2500).
    parquet_files = list(ou_dir.rglob("*.parquet"))
    assert len(parquet_files) <= 32
    # And every row is recoverable via the view.
    from oneuniverse.data.dataset_view import DatasetView
    got = DatasetView.from_path(ou_dir.parent).read()
    assert len(got) == n


def test_large_catalog_keeps_nside_32(tmp_path):
    """At 1M rows the default NSIDE=32 (~80 rows/cell) should still be picked."""
    import healpy as hp
    rng = np.random.default_rng(11)
    n = 1_000_000
    df = pd.DataFrame({
        "ra": rng.uniform(0, 360, n),
        "dec": np.degrees(np.arcsin(rng.uniform(-1, 1, n))),
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

    ou_dir = tmp_path / "big" / "oneuniverse"
    ou_dir.mkdir(parents=True)
    from oneuniverse.data.manifest import LoaderSpec
    from oneuniverse.data.format_spec import DataGeometry
    from oneuniverse.data.converter import write_ouf_dataset

    manifest = write_ouf_dataset(
        df=df, out_dir=ou_dir,
        survey_name="big", survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="big", version="0"),
    )
    assert int(manifest.partitioning.extra["nside"]) == 32


def test_partition_nside_can_be_forced(tmp_path):
    """Caller can pin partition_nside=4 even for a large catalog."""
    import healpy as hp
    rng = np.random.default_rng(13)
    n = 10_000
    df = pd.DataFrame({
        "ra": rng.uniform(0, 360, n),
        "dec": np.degrees(np.arcsin(rng.uniform(-1, 1, n))),
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

    ou_dir = tmp_path / "forced" / "oneuniverse"
    ou_dir.mkdir(parents=True)
    from oneuniverse.data.manifest import LoaderSpec
    from oneuniverse.data.format_spec import DataGeometry
    from oneuniverse.data.converter import write_ouf_dataset

    manifest = write_ouf_dataset(
        df=df, out_dir=ou_dir,
        survey_name="forced", survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="forced", version="0"),
        partition_nside=4,
    )
    assert int(manifest.partitioning.extra["nside"]) == 4
```

Run: `pytest test/test_partition_nside.py -v` → 3 new FAIL.

- [ ] **Step 2: Add `MIN_ROWS_PER_PARTITION` constant**

```python
# oneuniverse/data/format_spec.py — add near HEALPIX_PARTITION_NSIDE
MIN_ROWS_PER_PARTITION: int = 5_000
```

- [ ] **Step 3: Add `partition_nside` kwarg + auto-pick helper**

```python
# oneuniverse/data/converter.py — add helper above _write_partitions_by_healpix
def _auto_partition_nside(n_rows: int, min_rows: int = MIN_ROWS_PER_PARTITION) -> int:
    """Return the largest valid NSIDE (power of 2, ≤ HEALPIX_PARTITION_NSIDE) for
    which the *mean* rows-per-cell ≥ min_rows. Floors at 1.
    """
    nside = HEALPIX_PARTITION_NSIDE
    while nside > 1:
        npix = 12 * nside * nside
        if n_rows >= min_rows * npix:
            return nside
        nside //= 2
    return 1
```

Thread a new kwarg through `write_ouf_dataset`:

```python
def write_ouf_dataset(
    ...,
    partition_nside: Optional[int] = None,
    ...,
) -> Manifest:
    ...
    if geometry is DataGeometry.POINT:
        chosen_nside = (
            int(partition_nside) if partition_nside is not None
            else _auto_partition_nside(len(df))
        )
        partitions = _write_partitions_by_healpix(
            df, out_dir, compression, stats_builder, pdf_spec,
            partition_nside=chosen_nside,
        )
        if partitioning is None:
            partitioning = PartitioningSpec(
                scheme="healpix",
                column="_healpix32",
                extra={"nside": chosen_nside, "nest": True},
            )
```

- [ ] **Step 4: Coarsen in `_write_partitions_by_healpix`**

```python
def _write_partitions_by_healpix(
    df: pd.DataFrame,
    out_dir: Path,
    compression: str,
    stats_builder=None,
    pdf_spec: Optional[PdfSpec] = None,
    partition_nside: int = HEALPIX_PARTITION_NSIDE,
) -> List[PartitionSpec]:
    import pyarrow.parquet as pq

    if "_healpix32" not in df.columns:
        raise ValueError("POINT df missing required _healpix32 column")

    # Coarsen the partition key in NEST ordering by right-shifting.
    # NEST nside_fine = 32, nside_coarse = 2^k.  bits_to_drop = 2 * (log2(32/k)).
    fine = HEALPIX_PARTITION_NSIDE
    bits_to_drop = 2 * int(np.log2(fine // partition_nside))
    if bits_to_drop > 0:
        partition_cells = df["_healpix32"].to_numpy() >> bits_to_drop
    else:
        partition_cells = df["_healpix32"].to_numpy()

    data_root = out_dir / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    specs: List[PartitionSpec] = []
    # Group the DataFrame by the coarse cell.
    df = df.assign(_partition_cell=partition_cells)
    for cell, chunk in df.groupby("_partition_cell", sort=True, observed=False):
        cell = int(cell)
        cell_dir = data_root / HEALPIX_SUBDIR_FMT.format(cell=cell)
        cell_dir.mkdir(parents=True, exist_ok=True)
        rel_name = f"data/{cell_dir.name}/part_0000.parquet"
        part_path = out_dir / rel_name
        chunk = chunk.drop(columns=["_partition_cell"])
        table = _chunk_to_table(chunk, pdf_spec)
        pq.write_table(table, part_path, compression=compression)

        stats = stats_builder(chunk) if stats_builder else PartitionStats()
        specs.append(PartitionSpec(
            name=rel_name,
            n_rows=len(chunk),
            sha256=hash_file(part_path),
            size_bytes=part_path.stat().st_size,
            stats=stats,
            healpix_cell=cell,
        ))
        logger.info(
            "  %s: %d rows (%.1f MB)",
            rel_name, len(chunk), part_path.stat().st_size / 1e6,
        )
    return specs
```

- [ ] **Step 5: Run tests**

Run: `pytest test/test_partition_nside.py -v` → 4 PASS.
Run: `pytest test/test_desi_dr1_onboarding.py -v` → still all PASS (regression on existing onboarding suite — different NSIDE will be chosen at 1000-row fixture size, but cone-query works thanks to D5).

- [ ] **Step 6: Commit**

```bash
git add oneuniverse/data/converter.py oneuniverse/data/format_spec.py test/test_partition_nside.py
git commit -m "phase12/F3: adaptive partition NSIDE for small catalogs"
```

---

### Task 4: D1 — Remove module-level `_data_root` state

**Files:**
- Modify: `oneuniverse/data/_config.py` — delete `_data_root`, `get_data_root`, `set_data_root`. Keep only `env_data_root` + `resolve_survey_path` (the latter now reads only env + explicit arg).
- Modify: `oneuniverse/data/__init__.py` + `oneuniverse/__init__.py` — drop the two exports.
- Modify: `oneuniverse/data/database.py` (`build` classmethod) and `oneuniverse/data/converter.py` (`convert_survey`) — pass `data_root` explicitly through call args instead of mutating module state.
- Test: `test/test_data_root_removed.py` (new — asserts import fails).
- Audit: any test or notebook calling `set_data_root` — rewrite to pass `data_root=` to the consuming function.

**Why:** Process-global mutable state is the worst kind: tests bleed config into each other, multi-database use is impossible, and the Phase 6 closeout note (per-database `data_root` kwarg added) was supposed to retire this. Carrying the wrappers forever is the cost.

**Inventory (audit before coding):**

```bash
grep -rn 'set_data_root\|get_data_root' --include='*.py' \
    oneuniverse/ test/ scripts/
```

Expected hits (per the analysis pass): `oneuniverse/data/__init__.py`, `oneuniverse/__init__.py`, `oneuniverse/data/database.py:282,292`, `oneuniverse/data/converter.py:23 (docstring), 242, 246`, `oneuniverse/data/_config.py` (the definitions themselves). Tests/notebooks should be zero — if grep returns any, fold them into Step 4.

- [ ] **Step 1: Write failing test**

```python
# test/test_data_root_removed.py
import pytest


def test_set_data_root_no_longer_importable():
    with pytest.raises(ImportError):
        from oneuniverse.data import set_data_root  # noqa: F401


def test_get_data_root_no_longer_importable():
    with pytest.raises(ImportError):
        from oneuniverse.data import get_data_root  # noqa: F401


def test_env_data_root_still_available():
    """env_data_root is the surviving canonical accessor."""
    from oneuniverse.data._config import env_data_root  # must still import
    # Returns None or a Path — both fine; we only assert importability.
    assert env_data_root() is None or hasattr(env_data_root(), "exists")
```

Run: `pytest test/test_data_root_removed.py -v` → FAIL (functions still importable).

- [ ] **Step 2: Migrate internal callers**

In `oneuniverse/data/database.py` — `build` classmethod sets `set_data_root(raw_root)`. Replace by threading `data_root=raw_root` into the underlying constructor:

```python
# database.py around L282 — was:
#   from oneuniverse.data._config import set_data_root
#   set_data_root(raw_root)
# becomes:
db = cls(root=ouf_root, data_root=raw_root, ...)
```

(`OneuniverseDatabase.__init__` already accepts `data_root=`.)

In `oneuniverse/data/converter.py` — `convert_survey` sets `set_data_root(data_root)` then calls `resolve_survey_path()`. Switch to passing `data_root` directly to `resolve_survey_path` (extend that helper's signature):

```python
# _config.py — extend resolve_survey_path
def resolve_survey_path(
    survey_type: str,
    survey_name: str,
    data_subpath: str = "",
    *,
    data_root: Optional[Path] = None,
) -> Optional[Path]:
    """Return the survey data directory, using the explicit data_root or env fallback."""
    root = data_root if data_root is not None else env_data_root()
    if root is None:
        return None
    if data_subpath:
        return Path(root) / data_subpath
    return Path(root) / survey_type / survey_name
```

```python
# converter.py — convert_survey
def convert_survey(
    survey_name: str,
    data_root: Optional[str | Path] = None,
    ...
) -> Path:
    ...
    if raw_path is not None:
        ...
    else:
        survey_path = resolve_survey_path(
            config.survey_type, config.name, config.data_subpath,
            data_root=Path(data_root) if data_root else None,
        )
```

- [ ] **Step 3: Delete the wrappers**

```python
# oneuniverse/data/_config.py — final file (after edit):
"""Per-call data-root resolution. No module state.
... (rewritten module docstring)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_ENV_VAR = "ONEUNIVERSE_DATA_ROOT"


def env_data_root() -> Optional[Path]:
    env = os.environ.get(_ENV_VAR)
    return Path(env) if env else None


def resolve_survey_path(
    survey_type: str,
    survey_name: str,
    data_subpath: str = "",
    *,
    data_root: Optional[Path] = None,
) -> Optional[Path]:
    root = data_root if data_root is not None else env_data_root()
    if root is None:
        return None
    if data_subpath:
        return Path(root) / data_subpath
    return Path(root) / survey_type / survey_name
```

- [ ] **Step 4: Drop the exports**

```python
# oneuniverse/data/__init__.py — remove the get_data_root / set_data_root lines
# (from imports list and __all__).

# oneuniverse/__init__.py — same.
```

Update the `convert_survey` module-docstring example (`>>> set_data_root("/data/surveys")` becomes `>>> convert_survey("eboss_qso", data_root="/data/surveys")`).

- [ ] **Step 5: Run full test suite**

```bash
pytest -q
```

Expected: green. Any test that imports `set_data_root` must be rewritten as part of this commit (audit returned zero hits per the inventory, but the suite will tell us).

- [ ] **Step 6: Commit**

```bash
git add oneuniverse/data/_config.py oneuniverse/data/__init__.py \
        oneuniverse/__init__.py oneuniverse/data/database.py \
        oneuniverse/data/converter.py test/test_data_root_removed.py
git commit -m "phase12/D1: remove module-level data-root state and deprecated wrappers"
```

---

### Task 5: D3 — `convert_survey(loader=<instance>, ...)` overload

**Files:**
- Modify: `oneuniverse/data/converter.py` — `convert_survey`.
- Test: `test/test_convert_survey_loader_kwarg.py`.

**Why:** Today `convert_survey(survey_name, ...)` pulls the loader class from the `@register` registry. One-off conversions and tests are forced through `write_ouf_dataset` directly (Phase 10 deviation). Add an opt-in `loader=` instance overload; behaviour unchanged when omitted.

- [ ] **Step 1: Failing test**

```python
# test/test_convert_survey_loader_kwarg.py
import numpy as np
import pandas as pd
from pathlib import Path

from oneuniverse.data._base_loader import BaseSurveyLoader, SurveyConfig
from oneuniverse.data.converter import convert_survey
from oneuniverse.data.dataset_view import DatasetView


class _InlineLoader(BaseSurveyLoader):
    """A loader instance built on-the-fly with no @register decorator."""
    config = SurveyConfig(
        name="inline_fake",
        survey_type="spectroscopic",
        description="inline-built loader, not registered",
        column_groups=("core",),
    )

    def __init__(self, df):
        self._df = df

    def _load_raw(self, data_path=None, **kwargs):
        return self._df.copy()


def test_convert_survey_accepts_loader_instance(tmp_path):
    n = 50
    df = pd.DataFrame({
        "ra": np.linspace(0, 90, n, dtype=np.float64),
        "dec": np.linspace(-10, 10, n, dtype=np.float64),
        "z": np.full(n, 0.5, dtype=np.float32),
        "z_type": np.array(["spec"] * n, dtype="<U4"),
        "z_spec_err": np.full(n, 1e-3, dtype=np.float32),
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": np.array(["inline_fake"] * n, dtype="<U16"),
    })
    loader = _InlineLoader(df)

    out = tmp_path / "inline_fake"
    convert_survey(
        loader=loader,
        output_dir=out,
        overwrite=True,
    )

    view = DatasetView.from_path(out)
    got = view.read(columns=["ra", "dec", "z", "z_type"])
    assert len(got) == n
    assert set(got["z_type"].unique()) <= {"spec"}


def test_convert_survey_loader_instance_takes_precedence_over_name(tmp_path):
    """If both are given, the loader instance wins (registry is bypassed)."""
    ...  # similar smoke
```

Run: `pytest test/test_convert_survey_loader_kwarg.py -v` → FAIL (unrecognised kwarg).

- [ ] **Step 2: Extend `convert_survey` signature**

```python
# oneuniverse/data/converter.py
def convert_survey(
    survey_name: Optional[str] = None,
    *,
    loader=None,                           # BaseSurveyLoader instance, overrides survey_name
    data_root: Optional[str | Path] = None,
    partition_rows: Optional[int] = None,
    compression: str = COMPRESSION,
    overwrite: bool = False,
    output_dir: Optional[str | Path] = None,
    raw_path: Optional[str | Path] = None,
    partition_nside: Optional[int] = None,     # F3 surface here too
    **loader_kwargs: Any,
) -> Path:
    if loader is None and survey_name is None:
        raise TypeError(
            "convert_survey requires either a registered `survey_name=` or "
            "an explicit `loader=` instance"
        )
    if loader is None:
        from oneuniverse.data._registry import get_loader
        loader = get_loader(survey_name)
    config = loader.config
    survey_name = survey_name or config.name
    ...
```

The rest of the function body is unchanged. Keep the existing branch that resolves `survey_path` from `raw_path` / data_root.

- [ ] **Step 3: Run tests**

```bash
pytest test/test_convert_survey_loader_kwarg.py -v   # new tests PASS
pytest test/test_desi_dr1_onboarding.py -v           # regression: still PASS
```

- [ ] **Step 4: Commit**

```bash
git add oneuniverse/data/converter.py test/test_convert_survey_loader_kwarg.py
git commit -m "phase12/D3: convert_survey accepts a loader=<instance> overload"
```

---

### Task 6: Close Phase 12

- [ ] **Step 1: Full suite**

```bash
pytest -q
```

Record count. Expected: 345 (Phase 11 baseline) + ~9 new tests = ~354, all green.

- [ ] **Step 2: Update `plans/README.md`**

Add row:

```
| 12 | Carried-over debt: adaptive partition NSIDE (F3), manifest-NSIDE cone (D5), drop _data_root state (D1), pandas observed (D2), convert_survey loader= overload (D3) | **complete (YYYY-MM-DD, N/N tests green)** |
```

- [ ] **Step 3: Update memory `project_oneuniverse_stabilisation.md`**

Block summarising the five items + the constant/API surface changes (`MIN_ROWS_PER_PARTITION`, `partition_nside`, removed exports).

- [ ] **Step 4: Final commit**

```bash
git add plans/README.md
git commit -m "phase12: close-out — carried-over debt cleanup"
```

---

## Self-review checklist (run before execution)

**1. Spec coverage:** F3, D5, D1, D2, D3 each have a dedicated task — confirmed.
**2. Placeholders:** Step bodies contain real code, no TODO markers. Step 1 of Task 1 deliberately defers the test body to Step 2 — flagged as a known exception (the minimum-repro requires a working dataset; we want the implementer to see the actual warning before pinning it).
**3. Type consistency:** `partition_nside` (Task 3) is used by `convert_survey` (Task 5) and `_write_partitions_by_healpix` (Task 3) — same name and `Optional[int]` semantics everywhere. `data_root=` (Task 4) consistently a `Path` keyword.

**4. Out of scope (deferred to later phases):**
- D4 (`spec_boss_like` registry default) — depends on a real survey adopting `boss_total_weight`; tracked for Phase 13+.
- Suite-wide warning audit, golden-image regression — Phase 15.
- SYSNet-map bitemporal versioning — Phase 14/15 cross-cut.
