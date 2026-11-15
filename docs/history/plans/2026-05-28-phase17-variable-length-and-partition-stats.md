# Phase 17 — Variable-Length Columns + Generic Partition Stats Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift the OUF fixed-width column assumption (so per-row variable-length payloads such as Lyα δ pixels, ZTF lightcurves, GAIA XP spectra, DESI BITWEIGHTS, and ragged multi-filter photometry can be written natively) and replace `PartitionStats`'s four hard-coded axes (ra/dec/z/t) with a generic per-column min/max dict so pushdown works on S/N, EBV, magnitude, or any other survey-specific column.

**Architecture:** Two narrowly-scoped extensions of the existing writer / manifest / reader layers. (a) `_chunk_to_table` learns a tiny dtype mini-language (`list<f4>`, `f4[N]`, `i8[N]`, `large_list<f4>`) and routes per-column conversions accordingly; `write_ouf_dataset(column_dtypes=...)` forwards the per-column type map. (b) `PartitionStats` gains `extra_ranges: Dict[str, (lo, hi)]`; `_default_stats_builder` accepts an `extra_stats_columns` list; `DatasetView._select_partitions` + `scan` accept an `extra_filters: Mapping[str, Range]` kwarg that drives both partition pruning and pyarrow row-level pushdown. OUF bumps 2.2.0 → 2.3.0. 2.0/2.1/2.2 manifests still parse.

**Tech Stack:** Python 3.9+, pyarrow ≥ 13 (FixedSizeList + List + LargeList already supported), pandas, dataclasses, pytest. No new runtime dependencies.

---

## File Structure

**New files:**
- `oneuniverse/data/dtype_lang.py` — parses the dtype mini-language and builds the matching pyarrow types.
- `test/test_dtype_lang.py` — mini-language parser tests.
- `test/test_variable_length_columns.py` — converter + reader round-trip for `list<f4>` / `f4[N]` / `i8[N]` / `large_list<f4>`.
- `test/test_partition_stats_extra_ranges.py` — `PartitionStats.extra_ranges` serialisation + builder integration.
- `test/test_dataset_view_extra_filters.py` — pushdown via `extra_filters`.
- `test/test_visual_phase17.py` — diagnostic figure (Lyα-style payload + extra-range pushdown).

**Modified files:**
- `oneuniverse/data/manifest.py` — bump `FORMAT_VERSION` / `SCHEMA_VERSION` to `2.3.0`; extend version-compat check; extend `PartitionStats` + (de)serialisation.
- `oneuniverse/data/format_spec.py` — bump duplicate `FORMAT_VERSION` / `SCHEMA_VERSION` to `2.3.0`.
- `oneuniverse/data/converter.py:80-97` — `write_ouf_dataset` gains `column_dtypes` + `extra_stats_columns`.
- `oneuniverse/data/converter.py:560-574` — `_default_stats_builder` accepts `extra_columns` and populates `extra_ranges`.
- `oneuniverse/data/converter.py:711-743` — `_chunk_to_table` consumes `column_dtypes`.
- `oneuniverse/data/dataset_view.py:111-143` — `_select_partitions` honours `extra_ranges` for pruning.
- `oneuniverse/data/dataset_view.py:156-260` — `scan` / `read` accept `extra_filters` + push down to pyarrow.
- `oneuniverse/CLAUDE.md` — note variable-length + extra-stats hooks, OUF 2.3.
- `plans/README.md` — mark Phase 17 complete after close-out.
- `research/schema_generalisation_audit.md` — cross-ref Phase 17 plan.

---

## Pre-flight

- [ ] **Step 0a: Confirm baseline is green.**

```bash
cd /home/ravoux/Documents/Python/Packages/oneuniverse
pytest -q 2>&1 | tail -3
```

Expected: `406 passed, 1 skipped` (the post-Phase-16 baseline).

---

## Task 1: Dtype mini-language parser

**Files:**
- Create: `oneuniverse/data/dtype_lang.py`
- Create: `test/test_dtype_lang.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_dtype_lang.py
"""Phase 17 T1 — dtype mini-language."""
import pyarrow as pa
import pytest

from oneuniverse.data.dtype_lang import parse_dtype, is_variable_length


def test_scalar_f4():
    t = parse_dtype("f4")
    assert t.equals(pa.float32())


def test_scalar_i8():
    t = parse_dtype("i8")
    assert t.equals(pa.int64())


def test_fixed_size_list_f4_64():
    t = parse_dtype("f4[64]")
    assert isinstance(t, pa.FixedSizeListType)
    assert t.list_size == 64
    assert t.value_type.equals(pa.float32())


def test_fixed_size_list_i8_64():
    t = parse_dtype("i8[64]")
    assert isinstance(t, pa.FixedSizeListType)
    assert t.list_size == 64
    assert t.value_type.equals(pa.int64())


def test_variable_length_list_f4():
    t = parse_dtype("list<f4>")
    assert isinstance(t, pa.ListType)
    assert t.value_type.equals(pa.float32())


def test_large_list_f4():
    t = parse_dtype("large_list<f4>")
    assert isinstance(t, pa.LargeListType)
    assert t.value_type.equals(pa.float32())


def test_rejects_unknown_syntax():
    with pytest.raises(ValueError, match="dtype"):
        parse_dtype("array of floats")


def test_is_variable_length_classifies():
    assert is_variable_length("list<f4>")
    assert is_variable_length("large_list<f4>")
    assert not is_variable_length("f4[64]")
    assert not is_variable_length("f4")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_dtype_lang.py -v
```

Expected: `ImportError: No module named 'oneuniverse.data.dtype_lang'`.

- [ ] **Step 3: Implement the module**

```python
# oneuniverse/data/dtype_lang.py
"""Tiny dtype mini-language for OUF 2.3 variable-length payloads.

Grammar:

    f4               -> pa.float32()
    f8               -> pa.float64()
    i1 i2 i4 i8      -> pa.int8 / int16 / int32 / int64
    u1 u2 u4 u8      -> pa.uint8 / uint16 / uint32 / uint64
    U<N>             -> pa.string()   (string columns; N kept for the loader)
    <scalar>[N]      -> pa.FixedSizeList(scalar, N)
    list<scalar>     -> pa.list_(scalar)
    large_list<scalar> -> pa.large_list(scalar)

Whitespace is rejected. Used by ``_chunk_to_table`` to coerce list /
fixed-size list / variable-length list columns to the right pyarrow
type at write time.
"""
from __future__ import annotations

import re
from typing import Dict

import pyarrow as pa

_SCALAR_MAP: Dict[str, pa.DataType] = {
    "f4": pa.float32(),
    "f8": pa.float64(),
    "i1": pa.int8(),
    "i2": pa.int16(),
    "i4": pa.int32(),
    "i8": pa.int64(),
    "u1": pa.uint8(),
    "u2": pa.uint16(),
    "u4": pa.uint32(),
    "u8": pa.uint64(),
}

_FIXED_RE = re.compile(r"^([a-z]\d)\[(\d+)\]$")
_LIST_RE = re.compile(r"^list<([a-z]\d)>$")
_LARGE_LIST_RE = re.compile(r"^large_list<([a-z]\d)>$")


def parse_dtype(spec: str) -> pa.DataType:
    """Parse a dtype mini-language string into a pyarrow type."""
    if not isinstance(spec, str) or " " in spec:
        raise ValueError(f"invalid dtype string {spec!r} (no whitespace)")
    if spec in _SCALAR_MAP:
        return _SCALAR_MAP[spec]
    if spec.startswith("U") and spec[1:].isdigit():
        return pa.string()
    m = _FIXED_RE.match(spec)
    if m:
        scalar, n = m.group(1), int(m.group(2))
        if scalar not in _SCALAR_MAP:
            raise ValueError(f"unknown scalar {scalar!r} in dtype {spec!r}")
        return pa.list_(_SCALAR_MAP[scalar], n)
    m = _LIST_RE.match(spec)
    if m:
        return pa.list_(_SCALAR_MAP[m.group(1)])
    m = _LARGE_LIST_RE.match(spec)
    if m:
        return pa.large_list(_SCALAR_MAP[m.group(1)])
    raise ValueError(
        f"unsupported dtype {spec!r}; allowed forms: f4 / i8 / U32 / "
        f"f4[N] / list<f4> / large_list<f4>"
    )


def is_variable_length(spec: str) -> bool:
    """Return True iff ``spec`` produces a variable-length pyarrow type."""
    return spec.startswith("list<") or spec.startswith("large_list<")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/test_dtype_lang.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/dtype_lang.py test/test_dtype_lang.py
git commit -m "phase17/T1: dtype mini-language (f4/i8/list<f4>/f4[N]/large_list<f4>)"
```

---

## Task 2: `_chunk_to_table` consumes `column_dtypes`

**Files:**
- Modify: `oneuniverse/data/converter.py:711-743` (`_chunk_to_table`)
- Create: `test/test_chunk_to_table_dtypes.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_chunk_to_table_dtypes.py
"""Phase 17 T2 — _chunk_to_table accepts per-column dtype overrides."""
import numpy as np
import pandas as pd
import pyarrow as pa

from oneuniverse.data.converter import _chunk_to_table


def test_fixed_size_list_column():
    df = pd.DataFrame({
        "id": np.arange(3, dtype="i8"),
        "vals": [np.arange(4, dtype="f4"),
                 np.arange(4, dtype="f4") * 2,
                 np.arange(4, dtype="f4") * 3],
    })
    table = _chunk_to_table(df, pdf_spec=None, column_dtypes={"vals": "f4[4]"})
    assert isinstance(table.schema.field("vals").type, pa.FixedSizeListType)
    assert table.schema.field("vals").type.list_size == 4


def test_int_bitweight_column():
    df = pd.DataFrame({
        "id": np.arange(2, dtype="i8"),
        "BITWEIGHTS": [np.zeros(64, dtype="i8"), np.ones(64, dtype="i8")],
    })
    table = _chunk_to_table(
        df, pdf_spec=None, column_dtypes={"BITWEIGHTS": "i8[64]"},
    )
    t = table.schema.field("BITWEIGHTS").type
    assert isinstance(t, pa.FixedSizeListType)
    assert t.list_size == 64


def test_variable_length_list_column():
    df = pd.DataFrame({
        "id": np.arange(3, dtype="i8"),
        "delta": [np.arange(3, dtype="f4"),
                  np.arange(5, dtype="f4"),
                  np.arange(7, dtype="f4")],
    })
    table = _chunk_to_table(
        df, pdf_spec=None, column_dtypes={"delta": "list<f4>"},
    )
    assert isinstance(table.schema.field("delta").type, pa.ListType)
    py = table.column("delta").to_pylist()
    assert [len(x) for x in py] == [3, 5, 7]


def test_large_list_column():
    df = pd.DataFrame({
        "id": np.arange(2, dtype="i8"),
        "lc": [np.arange(10, dtype="f4"), np.arange(20, dtype="f4")],
    })
    table = _chunk_to_table(
        df, pdf_spec=None, column_dtypes={"lc": "large_list<f4>"},
    )
    assert isinstance(table.schema.field("lc").type, pa.LargeListType)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_chunk_to_table_dtypes.py -v
```

Expected: TypeError on `column_dtypes` keyword (not yet accepted).

- [ ] **Step 3: Extend `_chunk_to_table`**

Replace the function in `oneuniverse/data/converter.py` (lines 711-743) with:

```python
def _chunk_to_table(
    chunk: pd.DataFrame,
    pdf_spec: Optional[PdfSpec],
    *,
    column_dtypes: Optional[Dict[str, str]] = None,
):
    """Convert a DataFrame chunk to a pyarrow Table.

    Routing
    -------
    * Columns listed in ``column_dtypes`` are coerced according to the
      dtype mini-language in :mod:`oneuniverse.data.dtype_lang`
      (``f4[N]`` / ``i8[N]`` / ``list<f4>`` / ``large_list<f4>``).
    * PDF columns implied by ``pdf_spec`` are cast to
      ``FixedSizeList[float32, n_components]`` (Phase 10 behaviour).
    * Remaining columns fall through to :func:`pa.Table.from_pandas`.
    """
    import pyarrow as pa

    from oneuniverse.data.dtype_lang import is_variable_length, parse_dtype

    column_dtypes = dict(column_dtypes or {})

    # Resolve PDF list columns first so they appear in ``list_cols`` like
    # any other variable-length payload.
    pdf_cols = []
    if pdf_spec is not None:
        n = int(pdf_spec.n_components)
        pdf_cols = ["z_pdf_values"]
        if pdf_spec.parameterisation == "mixmod":
            pdf_cols += ["z_pdf_sigma", "z_pdf_weights"]
        pdf_cols = [c for c in pdf_cols if c in chunk.columns]
        for c in pdf_cols:
            column_dtypes.setdefault(c, f"f4[{n}]")

    list_cols = [c for c in column_dtypes if c in chunk.columns]
    scalar = chunk.drop(columns=list_cols)
    table = pa.Table.from_pandas(scalar, preserve_index=False)

    for col in list_cols:
        spec = column_dtypes[col]
        target = parse_dtype(spec)
        if isinstance(target, pa.FixedSizeListType):
            n_target = target.list_size
            arr = np.stack(
                [
                    np.asarray(r, dtype=target.value_type.to_pandas_dtype())
                    for r in chunk[col].to_numpy()
                ]
            )
            if arr.shape[1] != n_target:
                raise ValueError(
                    f"column {col!r}: expected {n_target} components, "
                    f"got {arr.shape[1]}"
                )
            flat = pa.array(arr.reshape(-1), type=target.value_type)
            built = pa.FixedSizeListArray.from_arrays(flat, n_target)
        elif isinstance(target, (pa.ListType, pa.LargeListType)):
            built = pa.array(
                [list(r) for r in chunk[col].to_numpy()],
                type=target,
            )
        else:
            built = pa.array(chunk[col].to_numpy(), type=target)
        table = table.append_column(col, built)

    return table
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/test_chunk_to_table_dtypes.py test/test_pdf_converter.py -v
```

Expected: 4 new tests pass; PDF converter test still green.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/converter.py test/test_chunk_to_table_dtypes.py
git commit -m "phase17/T2: _chunk_to_table routes per-column dtype overrides via mini-language"
```

---

## Task 3: `write_ouf_dataset(column_dtypes=...)` + reader round-trip

**Files:**
- Modify: `oneuniverse/data/converter.py:80-97` (signature)
- Modify: `oneuniverse/data/converter.py:586,664,692` (every `_chunk_to_table` call site)
- Create: `test/test_variable_length_columns.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_variable_length_columns.py
"""Phase 17 T3 — write_ouf_dataset round-trips variable-length payloads."""
import healpy as hp
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from oneuniverse.data.converter import write_ouf_dataset
from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.manifest import LoaderSpec


def _base_core(n: int) -> pd.DataFrame:
    ra = np.linspace(0.0, 10.0, n).astype("f8")
    dec = np.linspace(-5.0, 5.0, n).astype("f8")
    return pd.DataFrame({
        "ra": ra,
        "dec": dec,
        "z": np.full(n, 0.5, dtype="f4"),
        "z_type": np.array(["spec"] * n, dtype=object),
        "z_err": np.full(n, 0.01, dtype="f4"),
        "galaxy_id": np.arange(n, dtype="i8"),
        "survey_id": np.array(["fix"] * n, dtype=object),
        "_original_row_index": np.arange(n, dtype="i8"),
        "_healpix32": hp.ang2pix(32, ra, dec, nest=True, lonlat=True).astype("i4"),
    })


def test_writer_emits_variable_length_list(tmp_path):
    df = _base_core(4)
    df["delta"] = [np.arange(k + 3, dtype="f4") for k in range(4)]
    out = tmp_path / "x" / "oneuniverse"
    out.mkdir(parents=True)
    write_ouf_dataset(
        df=df, out_dir=out,
        survey_name="x", survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="x", version="0"),
        column_dtypes={"delta": "list<f4>"},
    )
    view = DatasetView.from_path(out.parent)
    out_df = view.read()
    lengths = [len(x) for x in out_df["delta"]]
    assert sorted(lengths) == [3, 4, 5, 6]


def test_writer_emits_fixedsize_bitweights(tmp_path):
    df = _base_core(3)
    df["BITWEIGHTS"] = [np.arange(64, dtype="i8")] * 3
    out = tmp_path / "x" / "oneuniverse"
    out.mkdir(parents=True)
    write_ouf_dataset(
        df=df, out_dir=out,
        survey_name="x", survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="x", version="0"),
        column_dtypes={"BITWEIGHTS": "i8[64]"},
    )
    paths = sorted((out / "data").rglob("*.parquet"))
    assert paths, "no partition written"
    table = pq.read_table(paths[0])
    t = table.schema.field("BITWEIGHTS").type
    assert t.list_size == 64
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_variable_length_columns.py -v
```

Expected: TypeError on `column_dtypes` (write_ouf_dataset doesn't accept it yet).

- [ ] **Step 3: Add the kwarg + thread through call sites**

In `oneuniverse/data/converter.py`, extend the `write_ouf_dataset` signature to include:

```python
def write_ouf_dataset(
    ...,
    coordinate: Optional["CoordinateSpec"] = None,
    spectrum: Optional["SpectrumSpec"] = None,
    column_dtypes: Optional[Dict[str, str]] = None,
) -> Manifest:
```

At every `_chunk_to_table(chunk, pdf_spec)` call, append the kwarg:

```python
table = _chunk_to_table(chunk, pdf_spec, column_dtypes=column_dtypes)
```

(There are three call sites at lines ~586, ~664, ~692; the parent
helpers `_write_partitions` and `_write_partitions_by_healpix` need
the kwarg threaded through as well.)

Update `_write_partitions` and `_write_partitions_by_healpix`
signatures so they accept and forward `column_dtypes`. Each receives
a default of `None`.

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/test_variable_length_columns.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/converter.py test/test_variable_length_columns.py
git commit -m "phase17/T3: write_ouf_dataset(column_dtypes=...) + round-trip through DatasetView"
```

---

## Task 4: `PartitionStats.extra_ranges` + manifest (de)serialisation

**Files:**
- Modify: `oneuniverse/data/manifest.py:62-72` (PartitionStats)
- Modify: `oneuniverse/data/manifest.py:218-233` (`_from_dict` partition block)
- Create: `test/test_partition_stats_extra_ranges.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_partition_stats_extra_ranges.py
"""Phase 17 T4 — PartitionStats.extra_ranges round-trips."""
import json

import pytest

from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.manifest import (
    FORMAT_VERSION,
    LoaderSpec,
    Manifest,
    OriginalFileSpec,
    PartitionSpec,
    PartitionStats,
    read_manifest,
    write_manifest,
)


def _minimal(partitions):
    return Manifest(
        oneuniverse_format_version=FORMAT_VERSION,
        oneuniverse_schema_version=FORMAT_VERSION,
        geometry=DataGeometry.POINT,
        survey_name="fixture",
        survey_type="spectroscopic",
        created_utc="2026-05-28T00:00:00+00:00",
        original_files=[
            OriginalFileSpec(
                path="raw.fits", sha256="0123456789abcdef",
                n_rows=1, size_bytes=100, format="fits",
            ),
        ],
        partitions=partitions,
        partitioning=None,
        schema=[],
        conversion_kwargs={},
        loader=LoaderSpec(name="fixture_loader", version="0.0"),
    )


def test_extra_ranges_default_empty():
    stats = PartitionStats()
    assert stats.extra_ranges == {}


def test_extra_ranges_in_manifest_roundtrip(tmp_path):
    parts = [
        PartitionSpec(
            name="data/part_0000.parquet",
            n_rows=1, sha256="fedcba9876543210", size_bytes=50,
            stats=PartitionStats(
                ra_min=0.0, ra_max=1.0,
                dec_min=-1.0, dec_max=1.0,
                z_min=0.1, z_max=0.5,
                extra_ranges={"snr": (10.0, 100.0), "ebv": (0.0, 0.05)},
            ),
        ),
    ]
    m = _minimal(parts)
    path = tmp_path / "manifest.json"
    write_manifest(path, m)
    read = read_manifest(path)
    s = read.partitions[0].stats
    assert s.extra_ranges == {
        "snr": (10.0, 100.0), "ebv": (0.0, 0.05),
    }


def test_old_manifest_without_extra_ranges_parses(tmp_path):
    payload = {
        "oneuniverse_format_version": "2.2.0",
        "oneuniverse_schema_version": "2.2.0",
        "geometry": "point",
        "survey_name": "legacy", "survey_type": "spectroscopic",
        "created_utc": "2026-05-28T00:00:00+00:00",
        "original_files": [{
            "path": "raw.fits", "sha256": "0123456789abcdef",
            "n_rows": 1, "size_bytes": 100, "format": "fits",
        }],
        "partitions": [{
            "name": "data/part_0000.parquet", "n_rows": 1,
            "sha256": "fedcba9876543210", "size_bytes": 50,
            "stats": {"ra_min": 0.0, "ra_max": 1.0},
        }],
        "partitioning": None, "schema": [], "conversion_kwargs": {},
        "loader": {"name": "legacy_loader", "version": "0.0"},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    read = read_manifest(path)
    assert read.partitions[0].stats.extra_ranges == {}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_partition_stats_extra_ranges.py -v
```

Expected: `TypeError: __init__() got an unexpected keyword argument 'extra_ranges'`.

- [ ] **Step 3: Extend `PartitionStats` + (de)serialisation**

In `oneuniverse/data/manifest.py`, replace the `PartitionStats`
dataclass (lines 62-72) with:

```python
@dataclass(frozen=True)
class PartitionStats:
    ra_min: Optional[float] = None
    ra_max: Optional[float] = None
    dec_min: Optional[float] = None
    dec_max: Optional[float] = None
    z_min: Optional[float] = None
    z_max: Optional[float] = None
    t_min: Optional[float] = None
    t_max: Optional[float] = None
    # Phase 17: generic per-column min/max for arbitrary axes
    # (S/N, EBV, magnitude, ...). Empty by default for forward-compat.
    extra_ranges: Dict[str, tuple] = field(default_factory=dict)
```

Update the `_from_dict` partition-stats block (around line 225) so
``stats=PartitionStats(**p.get("stats", {}))`` first normalises the
``extra_ranges`` dict (lists → tuples). Replace the existing line:

```python
stats=PartitionStats(**p.get("stats", {})),
```

with:

```python
stats=_load_partition_stats(p.get("stats", {})),
```

And add the helper just above `_from_dict`:

```python
def _load_partition_stats(raw: Dict[str, Any]) -> PartitionStats:
    er = {
        k: (float(v[0]), float(v[1]))
        for k, v in raw.get("extra_ranges", {}).items()
    }
    return PartitionStats(
        ra_min=raw.get("ra_min"), ra_max=raw.get("ra_max"),
        dec_min=raw.get("dec_min"), dec_max=raw.get("dec_max"),
        z_min=raw.get("z_min"), z_max=raw.get("z_max"),
        t_min=raw.get("t_min"), t_max=raw.get("t_max"),
        extra_ranges=er,
    )
```

`_to_dict` already calls `asdict(m)` which serialises the dict
member; tuples are emitted as JSON arrays, which the loader then
normalises back to tuples.

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/test_partition_stats_extra_ranges.py test/test_manifest.py test/test_manifest_phase16.py -v 2>&1 | tail -8
```

Expected: 3 new + existing manifest tests green.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/manifest.py test/test_partition_stats_extra_ranges.py
git commit -m "phase17/T4: PartitionStats.extra_ranges + back-compat loader"
```

---

## Task 5: `_default_stats_builder` populates `extra_ranges` + `write_ouf_dataset(extra_stats_columns=...)`

**Files:**
- Modify: `oneuniverse/data/converter.py:80-97` (signature)
- Modify: `oneuniverse/data/converter.py:166-174` (passing builder)
- Modify: `oneuniverse/data/converter.py:560-574` (`_default_stats_builder`)
- Create: `test/test_extra_stats_columns.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_extra_stats_columns.py
"""Phase 17 T5 — writer populates extra_ranges per partition."""
import healpy as hp
import numpy as np
import pandas as pd

from oneuniverse.data.converter import write_ouf_dataset
from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.manifest import LoaderSpec, read_manifest


def _base_core(n: int) -> pd.DataFrame:
    ra = np.linspace(0.0, 30.0, n).astype("f8")
    dec = np.linspace(-5.0, 5.0, n).astype("f8")
    return pd.DataFrame({
        "ra": ra, "dec": dec,
        "z": np.linspace(0.1, 0.9, n).astype("f4"),
        "z_type": np.array(["spec"] * n, dtype=object),
        "z_err": np.full(n, 0.01, dtype="f4"),
        "galaxy_id": np.arange(n, dtype="i8"),
        "survey_id": np.array(["fix"] * n, dtype=object),
        "_original_row_index": np.arange(n, dtype="i8"),
        "_healpix32": hp.ang2pix(32, ra, dec, nest=True, lonlat=True).astype("i4"),
    })


def test_extra_ranges_present_per_partition(tmp_path):
    n = 200
    df = _base_core(n)
    df["snr"] = np.linspace(5.0, 95.0, n).astype("f4")
    df["ebv"] = np.linspace(0.0, 0.1, n).astype("f4")
    out = tmp_path / "x" / "oneuniverse"
    out.mkdir(parents=True)
    write_ouf_dataset(
        df=df, out_dir=out,
        survey_name="x", survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="x", version="0"),
        extra_stats_columns=["snr", "ebv"],
    )
    m = read_manifest(out / "manifest.json")
    have_snr = False
    have_ebv = False
    for p in m.partitions:
        er = p.stats.extra_ranges
        if "snr" in er:
            lo, hi = er["snr"]
            assert lo <= hi
            have_snr = True
        if "ebv" in er:
            have_ebv = True
    assert have_snr and have_ebv
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_extra_stats_columns.py -v
```

Expected: TypeError on `extra_stats_columns`.

- [ ] **Step 3: Extend converter**

Add to `write_ouf_dataset` signature:

```python
column_dtypes: Optional[Dict[str, str]] = None,
extra_stats_columns: Optional[Sequence[str]] = None,
```

When constructing `stats_builder` (around line 172), if the caller
passed `None` but `extra_stats_columns` is non-empty, wrap the
default builder with the extra columns:

```python
if stats_builder is None:
    extra_cols = tuple(extra_stats_columns or ())
    if extra_cols:
        def _builder(chunk: pd.DataFrame, _extra=extra_cols) -> PartitionStats:
            return _default_stats_builder(chunk, extra_columns=_extra)
        stats_builder = _builder
    else:
        stats_builder = _default_stats_builder
```

Replace `_default_stats_builder` (lines 560-574) with:

```python
def _default_stats_builder(
    chunk: pd.DataFrame,
    *,
    extra_columns: tuple = (),
) -> PartitionStats:
    def _minmax(col: str):
        if col not in chunk.columns:
            return None, None
        return float(chunk[col].min()), float(chunk[col].max())

    ra_lo, ra_hi = _minmax("ra")
    dec_lo, dec_hi = _minmax("dec")
    z_lo, z_hi = _minmax("z")
    t_lo, t_hi = _minmax("t_obs")
    er = {}
    for col in extra_columns:
        lo, hi = _minmax(col)
        if lo is not None:
            er[col] = (lo, hi)
    return PartitionStats(
        ra_min=ra_lo, ra_max=ra_hi,
        dec_min=dec_lo, dec_max=dec_hi,
        z_min=z_lo, z_max=z_hi,
        t_min=t_lo, t_max=t_hi,
        extra_ranges=er,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/test_extra_stats_columns.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/converter.py test/test_extra_stats_columns.py
git commit -m "phase17/T5: writer populates PartitionStats.extra_ranges from extra_stats_columns"
```

---

## Task 6: `DatasetView` honours `extra_filters` for partition pruning + row-level pushdown

**Files:**
- Modify: `oneuniverse/data/dataset_view.py:111-143` (`_select_partitions`)
- Modify: `oneuniverse/data/dataset_view.py:156-260` (`scan` / `read`)
- Create: `test/test_dataset_view_extra_filters.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_dataset_view_extra_filters.py
"""Phase 17 T6 — DatasetView prunes + pushes down extra_filters."""
import healpy as hp
import numpy as np
import pandas as pd

from oneuniverse.data.converter import write_ouf_dataset
from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.manifest import LoaderSpec


def _make(tmp_path):
    rng = np.random.default_rng(0)
    n = 5000
    ra = rng.uniform(0, 360, n).astype("f8")
    dec = rng.uniform(-60, 60, n).astype("f8")
    df = pd.DataFrame({
        "ra": ra, "dec": dec,
        "z": rng.uniform(0.0, 1.0, n).astype("f4"),
        "z_type": np.array(["spec"] * n, dtype=object),
        "z_err": np.full(n, 0.01, dtype="f4"),
        "galaxy_id": np.arange(n, dtype="i8"),
        "survey_id": np.array(["fix"] * n, dtype=object),
        "_original_row_index": np.arange(n, dtype="i8"),
        "_healpix32": hp.ang2pix(32, ra, dec, nest=True, lonlat=True).astype("i4"),
        "snr": rng.uniform(1.0, 200.0, n).astype("f4"),
    })
    out = tmp_path / "x" / "oneuniverse"
    out.mkdir(parents=True)
    write_ouf_dataset(
        df=df, out_dir=out,
        survey_name="x", survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="x", version="0"),
        extra_stats_columns=["snr"],
    )
    return DatasetView.from_path(out.parent), df


def test_extra_filters_push_down_to_rows(tmp_path):
    view, df = _make(tmp_path)
    out = view.read(extra_filters={"snr": (50.0, None)})
    assert (out["snr"] >= 50.0).all()


def test_extra_filters_upper_bound(tmp_path):
    view, df = _make(tmp_path)
    out = view.read(extra_filters={"snr": (None, 20.0)})
    assert (out["snr"] <= 20.0).all()


def test_extra_filters_prune_partitions(tmp_path):
    """Partition pruning excludes partitions whose stats cannot overlap."""
    view, _ = _make(tmp_path)
    full = view._select_partitions()
    pruned = view._select_partitions(extra_filters={"snr": (180.0, None)})
    assert len(pruned) < len(full)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_dataset_view_extra_filters.py -v
```

Expected: TypeError on `extra_filters` keyword.

- [ ] **Step 3: Extend `_select_partitions`**

Replace the signature and body of `_select_partitions` (lines 111-143)
with:

```python
def _select_partitions(
    self,
    *,
    ra_range: Optional[Range] = None,
    dec_range: Optional[Range] = None,
    z_range: Optional[Range] = None,
    t_range: Optional[Range] = None,
    healpix_cells: Optional[Iterable[int]] = None,
    extra_filters: Optional[Mapping[str, Range]] = None,
) -> List[PartitionSpec]:
    """Return partitions whose stats may overlap the given filters."""
    cell_filter = (
        {int(c) for c in healpix_cells}
        if healpix_cells is not None else None
    )
    extras = dict(extra_filters or {})
    keep: List[PartitionSpec] = []
    for part in self.manifest.partitions:
        if (
            cell_filter is not None
            and part.healpix_cell is not None
            and part.healpix_cell not in cell_filter
        ):
            continue
        if not _range_overlaps(ra_range, part.stats.ra_min, part.stats.ra_max):
            continue
        if not _range_overlaps(dec_range, part.stats.dec_min, part.stats.dec_max):
            continue
        if not _range_overlaps(z_range, part.stats.z_min, part.stats.z_max):
            continue
        if not _range_overlaps(t_range, part.stats.t_min, part.stats.t_max):
            continue
        ok = True
        for col, rng in extras.items():
            er = part.stats.extra_ranges.get(col)
            if er is None:
                # No partition stats for the requested column: cannot prune
                # — keep partition; row-level pushdown will filter.
                continue
            if not _range_overlaps(rng, er[0], er[1]):
                ok = False
                break
        if ok:
            keep.append(part)
    return keep
```

- [ ] **Step 4: Add `extra_filters` to `scan` + push down to pyarrow**

In `scan` (~line 156) extend the signature:

```python
extra_filters: Optional[Mapping[str, Range]] = None,
```

In the body, just before `_build_dataset`, after the existing filter
expression assembly, augment the row-level pushdown filter with:

```python
extras = dict(extra_filters or {})
extra_expr = None
for col, (lo, hi) in extras.items():
    e = None
    if lo is not None:
        e = pc.field(col) >= lo
    if hi is not None:
        upper = pc.field(col) <= hi
        e = upper if e is None else (e & upper)
    if e is not None:
        extra_expr = e if extra_expr is None else (extra_expr & e)
if extra_expr is not None:
    filter = extra_expr if filter is None else (filter & extra_expr)
```

Pass `extra_filters=extras` to `_select_partitions`.

In `read` (~line 197) forward the kwarg too:

```python
extra_filters: Optional[Mapping[str, Range]] = None,
```

and pass it on to `scan(...)`.

(`Range = Tuple[Optional[float], Optional[float]]` is the existing
alias; add `Mapping` to the imports if needed.)

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest test/test_dataset_view_extra_filters.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add oneuniverse/data/dataset_view.py test/test_dataset_view_extra_filters.py
git commit -m "phase17/T6: DatasetView._select_partitions + scan/read honour extra_filters"
```

---

## Task 7: OUF 2.3.0 bump + back-compat

**Files:**
- Modify: `oneuniverse/data/manifest.py:33-34` (`FORMAT_VERSION`, `SCHEMA_VERSION`)
- Modify: `oneuniverse/data/manifest.py:200-211` (version check)
- Modify: `oneuniverse/data/format_spec.py:102-108`
- Modify: `test/test_lightcurve_geometry.py:42-44`
- Modify: `test/test_manifest_phase16.py:60` (assertion)

- [ ] **Step 1: Bump constants**

`oneuniverse/data/manifest.py`:

```python
FORMAT_VERSION: str = "2.3.0"
SCHEMA_VERSION: str = "2.3.0"
```

`oneuniverse/data/format_spec.py`:

```python
FORMAT_VERSION: str = "2.3.0"
SCHEMA_VERSION: str = "2.3.0"
```

- [ ] **Step 2: Extend version compat check**

In `_from_dict` (lines 200-211), replace the version-accept clause:

```python
fmt = raw["oneuniverse_format_version"]
if not (
    isinstance(fmt, str)
    and (
        fmt.startswith("2.0")
        or fmt.startswith("2.1")
        or fmt.startswith("2.2")
        or fmt.startswith("2.3")
    )
):
    raise ManifestValidationError(
        f"{path}: oneuniverse_format_version={fmt!r} is not compatible "
        f"with this library (expected 2.0.x / 2.1.x / 2.2.x / 2.3.x)."
    )
```

- [ ] **Step 3: Update the bumped-version tests**

`test/test_lightcurve_geometry.py`:

```python
def test_format_version_is_2_3_0():
    assert FORMAT_VERSION == "2.3.0"
    assert SCHEMA_VERSION == "2.3.0"
```

`test/test_manifest_phase16.py`:

```python
def test_version_constants_bumped():
    assert FORMAT_VERSION == "2.3.0"
```

- [ ] **Step 4: Run the manifest+version suites**

```bash
pytest test/test_manifest.py test/test_manifest_phase16.py test/test_lightcurve_geometry.py test/test_partition_stats_extra_ranges.py -q
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/manifest.py oneuniverse/data/format_spec.py \
        test/test_lightcurve_geometry.py test/test_manifest_phase16.py
git commit -m "phase17/T7: bump OUF format/schema to 2.3.0; 2.0/2.1/2.2 still parse"
```

---

## Task 8: Visual diagnostic

**Files:**
- Create: `test/test_visual_phase17.py`

- [ ] **Step 1: Write the test**

```python
# test/test_visual_phase17.py
"""Phase 17 visual diagnostic — variable-length payload + extra-range pushdown."""
from __future__ import annotations

from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from oneuniverse.data.converter import write_ouf_dataset  # noqa: E402
from oneuniverse.data.dataset_view import DatasetView  # noqa: E402
from oneuniverse.data.format_spec import DataGeometry  # noqa: E402
from oneuniverse.data.manifest import LoaderSpec  # noqa: E402

OUT = Path(__file__).parent / "test_output"
OUT.mkdir(exist_ok=True)


def test_phase17_visual(tmp_path):
    rng = np.random.default_rng(0)
    n = 600
    ra = rng.uniform(0, 360, n).astype("f8")
    dec = rng.uniform(-30, 30, n).astype("f8")
    snr = rng.uniform(1.0, 200.0, n).astype("f4")
    deltas = [
        rng.normal(0.0, 0.1, size=rng.integers(20, 60)).astype("f4")
        for _ in range(n)
    ]
    df = pd.DataFrame({
        "ra": ra, "dec": dec,
        "z": rng.uniform(0.1, 1.0, n).astype("f4"),
        "z_type": np.array(["spec"] * n, dtype=object),
        "z_err": np.full(n, 0.01, dtype="f4"),
        "galaxy_id": np.arange(n, dtype="i8"),
        "survey_id": np.array(["phase17"] * n, dtype=object),
        "_original_row_index": np.arange(n, dtype="i8"),
        "_healpix32": hp.ang2pix(32, ra, dec, nest=True, lonlat=True).astype("i4"),
        "snr": snr,
        "delta": deltas,
    })
    out = tmp_path / "phase17_viz" / "oneuniverse"
    out.mkdir(parents=True)
    write_ouf_dataset(
        df=df, out_dir=out,
        survey_name="phase17", survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
        loader=LoaderSpec(name="phase17_viz", version="0"),
        column_dtypes={"delta": "list<f4>"},
        extra_stats_columns=["snr"],
    )

    view = DatasetView.from_path(out.parent)
    hi_snr = view.read(extra_filters={"snr": (100.0, None)})

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    lengths = [len(x) for x in df["delta"]]
    ax[0].hist(lengths, bins=20, color="tab:blue", alpha=0.8)
    ax[0].set_xlabel("delta length per row")
    ax[0].set_ylabel("count")
    ax[0].set_title("variable-length `delta` payload")

    ax[1].hist(df["snr"], bins=40, color="tab:gray", alpha=0.6, label="all")
    ax[1].hist(hi_snr["snr"], bins=40, color="tab:red", alpha=0.8,
               label="extra_filters snr ≥ 100")
    ax[1].set_xlabel("snr")
    ax[1].legend()
    ax[1].set_title("extra-range pushdown")

    ax[2].plot(df["delta"].iloc[0], lw=0.8)
    ax[2].plot(df["delta"].iloc[1], lw=0.8)
    ax[2].plot(df["delta"].iloc[2], lw=0.8)
    ax[2].set_xlabel("pixel")
    ax[2].set_ylabel("delta")
    ax[2].set_title("3 example `delta` series (different lengths)")

    fig.tight_layout()
    out_png = OUT / "phase17_variable_length_and_extra_stats.png"
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    assert out_png.exists() and out_png.stat().st_size > 30_000
    try:
        from PIL import Image
    except ImportError:
        return
    with Image.open(out_png) as im:
        assert im.width >= 800 and im.height >= 200
```

- [ ] **Step 2: Run the test**

```bash
pytest test/test_visual_phase17.py -v
```

Expected: 1 passed; PNG created at `test/test_output/phase17_variable_length_and_extra_stats.png`.

- [ ] **Step 3: Commit**

```bash
git add test/test_visual_phase17.py \
        test/test_output/phase17_variable_length_and_extra_stats.png
git commit -m "phase17/T8: visual diagnostic — variable-length delta + extra-stats pushdown"
```

---

## Task 9: Docs

**Files:**
- Modify: `oneuniverse/CLAUDE.md` (OUF 2.2 → 2.3 mention; variable-length column note)
- Modify: `plans/README.md` (Phase 17 status)
- Modify: `research/schema_generalisation_audit.md` (Phase 17 close-out cross-ref)

- [ ] **Step 1: Update `oneuniverse/CLAUDE.md`**

Change "OUF 2.2 (format on disk)" to "OUF 2.3 (format on disk)" and
under the sub-spec list add:

```
PartitionStats now carries `extra_ranges: Dict[str, (lo, hi)]`
populated when `write_ouf_dataset(extra_stats_columns=[...])` is set;
DatasetView prunes + pushdowns via `extra_filters={...}`.
Variable-length payloads (Lyα δ, lightcurves, GAIA XP,
DESI BITWEIGHTS) route through
`write_ouf_dataset(column_dtypes={"col": "list<f4>" | "f4[N]" | "i8[N]" | "large_list<f4>"})`
(Phase 17).
```

- [ ] **Step 2: Update `plans/README.md`**

Add the Phase 17 row (with placeholder count, replaced in T10):

```
| 17 | Variable-length columns + generic `PartitionStats.extra_ranges` (dtype mini-language, `column_dtypes` writer kwarg, `extra_stats_columns`, `DatasetView.extra_filters`, OUF → 2.3.0) | **complete (2026-05-28, NNN/NNN tests green)** |
```

- [ ] **Step 3: Update `research/schema_generalisation_audit.md`**

In the "Suggested staging into phases" section, replace the
Phase 17 line with:

```
- **Phase 17 — Variable-length columns + generic partition stats.**
  Landed 2026-05-28. Adds `_chunk_to_table(column_dtypes=...)` with a
  small dtype mini-language (`f4[N]`, `i8[N]`, `list<f4>`,
  `large_list<f4>`), `PartitionStats.extra_ranges`,
  `write_ouf_dataset(extra_stats_columns=...)`,
  `DatasetView.extra_filters`. OUF 2.3.0. See
  [`../plans/2026-05-28-phase17-variable-length-and-partition-stats.md`](../plans/2026-05-28-phase17-variable-length-and-partition-stats.md).
```

- [ ] **Step 4: Commit**

```bash
git add oneuniverse/CLAUDE.md plans/README.md \
        research/schema_generalisation_audit.md
git commit -m "docs(phase17): OUF 2.3, variable-length columns, extra_ranges"
```

---

## Task 10: Close-out

- [ ] **Step 1: Run the full suite**

```bash
pytest -q 2>&1 | tail -3
```

Expected: green; record the count (should be ~418 = 406 baseline + ~12
new Phase 17 tests).

- [ ] **Step 2: Replace `NNN/NNN` in plans/README.md with the real count.**

- [ ] **Step 3: Update memory**

Append to
`/home/ravoux/.claude/projects/-home-ravoux-Documents-Python/memory/project_oneuniverse_stabilisation.md`:

```markdown
## Phase 17 — Variable-length columns + generic partition stats (complete 2026-05-28)

- `oneuniverse.data.dtype_lang` mini-language: `f4 / i8 / U32 /
  f4[N] / list<f4> / large_list<f4>`.
- `_chunk_to_table(column_dtypes=...)` + `write_ouf_dataset(column_dtypes=...)`
  route variable-length payloads (Lya delta, lightcurves, GAIA XP,
  DESI BITWEIGHTS) through pyarrow `FixedSizeList`, `List`, or
  `LargeList`.
- `PartitionStats.extra_ranges: Dict[str, (lo, hi)]` populated by
  `write_ouf_dataset(extra_stats_columns=[...])`.
- `DatasetView._select_partitions / scan / read` accept
  `extra_filters={...}` for per-column pushdown + partition pruning.
- OUF bump 2.2.0 -> 2.3.0; all earlier minor versions still parse.
- Tests: NNN/NNN green.
- Per-phase plan:
  `plans/2026-05-28-phase17-variable-length-and-partition-stats.md`.
```

- [ ] **Step 4: Final commit**

```bash
git add plans/README.md \
        /home/ravoux/.claude/projects/-home-ravoux-Documents-Python/memory/project_oneuniverse_stabilisation.md
git commit -m "phase17: close-out — OUF 2.3.0, NNN tests green, variable-length + extra_ranges"
```

---

## Self-review checklist

- [ ] No cosmology metadata anywhere.
- [ ] `list<f4>` / `f4[N]` / `i8[N]` / `large_list<f4>` round-trip
      through writer + reader.
- [ ] `PartitionStats.extra_ranges` defaults to `{}` and old
      manifests load cleanly.
- [ ] `DatasetView.read(extra_filters=...)` returns a strict subset
      of `read()` for the same view.
- [ ] OUF 2.0 / 2.1 / 2.2 still parse.
- [ ] Visual PNG `phase17_variable_length_and_extra_stats.png` exists,
      ≥ 30 kB, ≥ 800 × 200 px.
- [ ] Full suite green.

## Spec-coverage map

| Requirement | Task |
|---|---|
| Mini-language parser | T1 |
| `_chunk_to_table` routes by dtype | T2 |
| Writer signature + reader round-trip | T3 |
| `PartitionStats.extra_ranges` | T4 |
| `_default_stats_builder` populates `extra_ranges` | T5 |
| `DatasetView` honours `extra_filters` | T6 |
| OUF 2.3.0 bump + back-compat | T7 |
| Visual diagnostic | T8 |
| Docs | T9 |
| Close-out + memory | T10 |
