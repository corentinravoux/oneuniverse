# Phase S17 — General simulation storage, IO & optimisation (multi-backend substrate)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the OUF-Sim storage/IO/optimisation substrate **format-agnostic** — proven by a second, structurally-different dummy backend produced from the dummy sim — and close the concrete IO/optimisation gaps the Pillar-3 validation verdict flagged: particle **wrap-in-place** reads through a native adapter, **ExecutionPlan-driven bounded-memory** streaming reads, **MPI rank-partitioned** + **GPU** read hooks, and a **scale benchmark** that proves the bounded-memory claim with numbers.

**Architecture:** Keep the proven on-disk store format (JSON manifest + parquet chunks + HEALPix-NEST sky partitions + memmap `.npy` tiles + sidecar `_index.parquet`). Add a *second native input format* (`packed_npy`: one chunk-sorted slab per snapshot + a `header.json` with per-chunk row ranges — the Abacus-like "spatially pre-sorted" layout real codes ship) and a *second converter* (`PackedSimConverter`) that plugs into a new **format-agnostic store builder** (`build_store`) reusing the existing per-product writers. The builder + `NativeReaderAdapter` registry are the seam a real backend (AbacusSummit ASDF, Gadget HDF5, BigFile) implements. Then make reads consume an `ExecutionPlan` (memory budget → batch size), wrap particles in place (index-only over the sorted slab), and expose MPI/GPU read hooks.

**Tech Stack:** numpy, pyarrow, healpy, pyyaml (all present); optional `mpi4py` / `cudf` (import-guarded). **No real-simulation deps.** **Rule 1:** no `oneuniverse.data` / `combine` imports (the `test_sim_no_pillar1_imports.py` guard scans `oneuniverse/simulation/` recursively — it must stay green every task).

**Scope boundary (YAGNI):** The generality proof targets the four **core bulk products** (`snapshots`, `fields`, `halos`, `lightcone`) — enough to prove the converter/adapter abstraction is not coupled to the linear layout. The extended products (`tree`, `phase_space`, `gr_fields`, `amr`, `ic_posterior`, `checkpoints`) stay linear-specific; a second backend covering the core boundary is the point. This plan does **not** touch the science (forward modelling, inference, incremental updates) — those are deferred per the Pillar-3 plan §3.4.

---

## What already exists (do not rebuild)

- `oneuniverse/simulation/oufsim/write.py` — `write_oufsim_store` (linear) + module-level per-product writers `_write_chunked_catalog`, `_write_field_tiles`, `_write_field_reference`, `_write_lightcone`; `ingest_field`.
- `oneuniverse/simulation/oufsim/read.py` — `SimStore` (`read_box`, `read_field_box`, `read_amr_box`, `read_cone`, `last_read_stats`; `read_box` already has `columns` / `n_threads` / `device="gpu"` / `pushdown` params and a `native_file` field-memmap branch).
- `oneuniverse/simulation/oufsim/native.py` — `NativeReaderAdapter` ABC (`read_field_region` only) + `NumpyFieldAdapter`.
- `oneuniverse/simulation/oufsim/index.py` — Layer-1 toolkit (`cartesian_chunk_ids`, `chunk_coords`, `bbox_of`, `cube_overlaps_bbox`, `tile_specs`, `healpix_partition_ids`, `cone_partition_pixels`, `skypatch_partition_pixels`).
- `oneuniverse/simulation/oufsim/_parallel.py` — `map_partitions(fn, items, *, n_threads, use_mpi)` (mpi4py import-guarded; `i % size` rank assignment + `allgather` of index rows).
- `oneuniverse/simulation/oufsim/view.py` — `SimDatasetView.iter_box(...)` (batched streaming reads).
- `oneuniverse/simulation/oufsim/bench.py` — `measure_read(fn) -> ReadBenchmark(wall_s, peak_bytes, n_rows)`.
- `oneuniverse/simulation/execution.py` — `ExecutionMode` + `ExecutionPlan(mode, memory_budget_bytes, batch_rows, device, n_chunks_estimate)`.
- `oneuniverse/simulation/capabilities.py` — `BackendCapabilities(... heavy_step_modes, modes_for, supports)`.
- `oneuniverse/simulation/converter.py` — `SimConverter` ABC (`detect`/`declare_products`/`read_cosmology`/`read_unit_frame`; `convert()` **raises NotImplementedError**) + `register`/`get_converter`/`detect_converter`/`registered_codes`.
- `oneuniverse/simulation/linear/converter.py` — `LinearSimConverter` (`convert()` delegates to `write_oufsim_store`).
- `oneuniverse/simulation/linear/generate.py` — `generate_linear_sim(out, cosmo, *, box_size, n_grid, redshifts, seed, ...)`.

---

## File Structure (new / modified in S17)

| File | Responsibility |
|---|---|
| Modify `oufsim/native.py` | `read_rows` on the adapter ABC (default `NotImplementedError`); adapter **registry** (`register_adapter`/`get_adapter`/`ADAPTERS`); register `NumpyFieldAdapter` as `"npy"`. |
| Create `linear/pack.py` | `write_packed_native(linear_dir, out_dir, *, particle_chunk_nside)` — derive the `packed_npy` native (chunk-sorted slab + `header.json`). |
| Modify `oufsim/native.py` | `PackedNpyAdapter` (`read_field_region` + `read_rows`), registered `"packed_npy"`. |
| Create `oufsim/build.py` | `NativeProduct` dataclass + `build_store(...)` — format-agnostic orchestrator reusing `write.py` writers; core kinds `catalog`/`field`/`lightcone`; `projection` per product. |
| Create `packed/__init__.py`, `packed/converter.py` | `PackedSimConverter` — the 2nd backend; `convert()` via adapter + `build_store`. |
| Modify `oufsim/write.py` | `_write_chunked_catalog_reference(...)` — index-only particle wrap over the sorted slab. |
| Modify `oufsim/read.py` | `read_box` reference branch (adapter `read_rows`); `ExecutionPlan` consumption; MPI rank-partition + GPU hooks. |
| Modify `oufsim/__init__.py` | export `build_store`, `NativeProduct`, `PackedNpyAdapter`, `get_adapter`. |
| Modify `execution.py` | `ExecutionPlan.batch_for(bytes_per_row) -> int` (budget → batch). |
| Create `oufsim/scale_bench.py` | scale-sweep harness (convert+read wall/peak vs N; wrap-vs-reencode size). |
| Tests under `test/` | one `test_*.py` per task (paths below). |

---

## Pre-flight

- [ ] **Step 0: Baseline green.**

```bash
cd /home/ravoux/Documents/Python/Packages/oneuniverse
pytest test/test_sim_*.py test/test_lin_*.py test/test_oufsim_*.py -q 2>&1 | tail -3
```

Expected: all pass. Record the count.

---

## Task 1: Native adapter — row reads + format registry

The adapter ABC reads fields only. Wrap-in-place of **particles** needs a row reader, and a registry so the store resolves the right adapter from the manifest's `native_format`.

**Files:** Modify `oufsim/native.py`, `oufsim/__init__.py`; Test `test/test_oufsim_adapter_registry.py`.

- [ ] **Step 1: Write the failing test**

```python
# test/test_oufsim_adapter_registry.py
"""S17 T1 — native adapter row reads + format registry."""
import numpy as np
import pytest

from oneuniverse.simulation.oufsim.native import (
    NumpyFieldAdapter, get_adapter, register_adapter, NativeReaderAdapter,
)


def test_registry_resolves_by_format():
    assert isinstance(get_adapter("npy"), NumpyFieldAdapter)
    with pytest.raises(KeyError):
        get_adapter("does_not_exist")


def test_field_adapter_has_no_row_product(tmp_path):
    a = np.arange(8 * 8 * 8, dtype=np.float64).reshape(8, 8, 8)
    p = tmp_path / "f.npy"; np.save(p, a)
    ad = get_adapter("npy")
    sub = ad.read_field_region(p, (slice(0, 4), slice(0, 4), slice(0, 4)))
    assert sub.shape == (4, 4, 4)
    with pytest.raises(NotImplementedError):
        ad.read_rows(p, slice(0, 4))


def test_register_adapter_is_idempotent_by_format():
    @register_adapter
    class _Dummy(NativeReaderAdapter):
        native_format = "dummy_fmt_t1"
        def read_field_region(self, path, cell_slice):
            return np.zeros((1, 1, 1))
    assert get_adapter("dummy_fmt_t1").native_format == "dummy_fmt_t1"
```

- [ ] **Step 2: Run — FAIL** (`get_adapter`/`register_adapter`/`read_rows` unknown).

Run: `pytest test/test_oufsim_adapter_registry.py -v`

- [ ] **Step 3: Implement** — replace `oufsim/native.py` with:

```python
"""Layer-2 native-format readers for the wrap-in-place (`reference`) projection.

A `reference` store holds only manifest + sidecar index; the bulk data stays
in the native files and is read through a `NativeReaderAdapter` (memmap /
partial read). The dummy ships two formats — `npy` (scattered linear layout)
and `packed_npy` (chunk-sorted slab) — registered in `ADAPTERS`. A real
backend (parallel-HDF5, ASDF/pack9, BigFile) implements the same ABC and
registers itself; that is how a petabyte sim is wrapped without copying.
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import ClassVar, Dict, Optional, Sequence, Tuple, Union

import numpy as np

CellSlice = Tuple[slice, slice, slice]


class NativeReaderAdapter(abc.ABC):
    """Partial reader over a native simulation file (no whole-array load)."""

    native_format: ClassVar[str] = "abstract"

    @abc.abstractmethod
    def read_field_region(self, path: Union[str, Path],
                          cell_slice: CellSlice) -> np.ndarray:
        """Return a sub-array of a native 3-D field (memmap-backed)."""

    def read_rows(self, path: Union[str, Path], row_slice: slice,
                  columns: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
        """Return {column: array} for a contiguous row range of a point product.

        Optional capability: formats without a row product (e.g. a bare field
        `.npy`) leave this unimplemented.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no row product (read_rows)")


ADAPTERS: Dict[str, NativeReaderAdapter] = {}


def register_adapter(cls):
    """Class decorator: register an adapter instance by its ``native_format``."""
    fmt = getattr(cls, "native_format", None)
    if not fmt or fmt == "abstract":
        raise ValueError(f"{cls.__name__} must set a concrete native_format")
    ADAPTERS[fmt] = cls()
    return cls


def get_adapter(native_format: str) -> NativeReaderAdapter:
    if native_format not in ADAPTERS:
        raise KeyError(
            f"no native adapter for format {native_format!r}; "
            f"known: {sorted(ADAPTERS)}")
    return ADAPTERS[native_format]


@register_adapter
class NumpyFieldAdapter(NativeReaderAdapter):
    """numpy `.npy` field adapter — the scattered linear native format."""

    native_format = "npy"

    def read_field_region(self, path, cell_slice):
        arr = np.load(path, mmap_mode="r")
        return np.array(arr[cell_slice])     # materialise only the sub-region
```

Then in `oufsim/__init__.py` add `get_adapter`, `register_adapter`, `ADAPTERS` to the import from `.native` and to `__all__`.

- [ ] **Step 4: Run — PASS.** `pytest test/test_oufsim_adapter_registry.py -v`

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/simulation/oufsim/native.py oneuniverse/simulation/oufsim/__init__.py test/test_oufsim_adapter_registry.py
git commit -m "phaseS17/T1: native adapter row reads + format registry (get_adapter/register_adapter)"
```

---

## Task 2: `packed_npy` native format + adapter (the 2nd, distinct dataset)

Derive a second native dataset from a linear native dir: particles **chunk-sorted** into one slab per snapshot with a `header.json` recording each chunk's `[row_start, row_stop)` + bbox — the spatially-pre-sorted layout real codes (AbacusSummit cells, Gadget Hilbert) ship, and the precondition for index-only particle wrapping (T5).

**Files:** Create `linear/pack.py`; Modify `oufsim/native.py` (add `PackedNpyAdapter`); Test `test/test_lin_pack.py`.

- [ ] **Step 1: Write the failing test**

```python
# test/test_lin_pack.py
"""S17 T2 — packed_npy native format + adapter."""
import json

import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear import generate_linear_sim
from oneuniverse.simulation.linear.pack import write_packed_native
from oneuniverse.simulation.oufsim.native import get_adapter


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_packed_native_is_chunk_sorted_with_ranges(tmp_path):
    lin = generate_linear_sim(tmp_path / "lin", _cosmo(), box_size=200.0,
                              n_grid=32, redshifts=(0.0,), seed=2)
    pk = write_packed_native(lin, tmp_path / "pk", particle_chunk_nside=4)
    hdr = json.loads((pk / "header.json").read_text())
    assert hdr["native_format"] == "packed_npy"
    ci = hdr["snapshots"]["z0.000"]["chunk_index"]
    # contiguous, non-overlapping, covering all rows
    assert ci[0]["row_start"] == 0
    assert all(ci[i]["row_stop"] == ci[i + 1]["row_start"]
               for i in range(len(ci) - 1))
    assert ci[-1]["row_stop"] == 32 ** 3


def test_packed_adapter_reads_field_and_rows(tmp_path):
    lin = generate_linear_sim(tmp_path / "lin", _cosmo(), box_size=200.0,
                              n_grid=32, redshifts=(0.0,), seed=2)
    pk = write_packed_native(lin, tmp_path / "pk", particle_chunk_nside=4)
    hdr = json.loads((pk / "header.json").read_text())
    ad = get_adapter("packed_npy")
    # field region matches the linear field
    fpath = pk / hdr["fields"]["z0.000"]["file"]
    sub = ad.read_field_region(fpath, (slice(0, 8), slice(0, 8), slice(0, 8)))
    ref = np.load(lin / "z0.000" / "field.npy")[:8, :8, :8]
    assert np.allclose(sub, ref)
    # row read of the first chunk returns named columns inside that chunk bbox
    c0 = hdr["snapshots"]["z0.000"]["chunk_index"][0]
    ppath = pk / hdr["snapshots"]["z0.000"]["file"]
    cols = ad.read_rows(ppath, slice(c0["row_start"], c0["row_stop"]),
                        columns=("x", "y", "z"))
    assert set(cols) == {"x", "y", "z"}
    assert cols["x"].min() >= c0["xlo"] - 1e-6
    assert cols["x"].max() <= c0["xhi"] + 1e-6
```

- [ ] **Step 2: Run — FAIL** (`linear.pack` / `packed_npy` adapter absent).

- [ ] **Step 3a: Implement `linear/pack.py`**

```python
# oneuniverse/simulation/linear/pack.py
"""Derive a `packed_npy` native dataset from a linear native dir.

Layout (one directory):

    {out}/
      header.json                  # box/grid/cosmology + per-product block map
      snapshots_z0.000.npy         # (N,6) particles, CHUNK-SORTED (x,y,z,vx,vy,vz)
      fields_z0.000.npy            # (n,n,n) density field
      halos_z0.000.parquet         # halos (small; left as parquet)
      lightcone.parquet            # sky (small; left as parquet)

The particle slab is sorted by cartesian chunk id so each chunk is a
contiguous [row_start, row_stop) range — the precondition for index-only
wrapping (Phase S17 T5). This mirrors how AbacusSummit / Gadget ship cells in
a spatial order. `PART_COLS` is the canonical particle column order.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
import yaml

from oneuniverse.simulation.oufsim.index import cartesian_chunk_ids, chunk_coords

PART_COLS = ("x", "y", "z", "vx", "vy", "vz")


def _ztag(z: float) -> str:
    return f"z{z:.3f}"


def write_packed_native(linear_dir: Union[str, Path], out_dir: Union[str, Path],
                        *, particle_chunk_nside: int = 4) -> Path:
    """Convert a linear native dir into a `packed_npy` native dir. Returns it."""
    linear_dir = Path(linear_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((linear_dir / "config.yaml").read_text())
    box = float(cfg["box_size"])
    n_side = int(particle_chunk_nside)

    header = {
        "native_format": "packed_npy",
        "box_size": box,
        "n_grid": int(cfg["n_grid"]),
        "redshifts": [float(z) for z in cfg["redshifts"]],
        "cosmology": cfg["cosmology"],
        "part_cols": list(PART_COLS),
        "snapshots": {}, "fields": {}, "halos": {}, "lightcone": None,
    }

    for z in cfg["redshifts"]:
        zt = _ztag(float(z))
        parts = np.load(linear_dir / zt / "particles.npy")          # (N,6)
        pos = parts[:, :3]
        cid = cartesian_chunk_ids(pos, box, n_side)
        order = np.argsort(cid, kind="stable")                       # chunk-sort
        parts_sorted = np.ascontiguousarray(parts[order])
        cid_sorted = cid[order]
        fname = f"snapshots_{zt}.npy"
        np.save(out / fname, parts_sorted)

        uniq, starts = np.unique(cid_sorted, return_index=True)
        bounds = list(starts) + [len(cid_sorted)]
        chunk_index = []
        for i, cc in enumerate(uniq):
            sl = slice(int(bounds[i]), int(bounds[i + 1]))
            p = parts_sorted[sl, :3]
            cx, cy, cz = chunk_coords(int(cc), n_side)
            chunk_index.append({
                "chunk_id": int(cc), "cx": cx, "cy": cy, "cz": cz,
                "row_start": int(sl.start), "row_stop": int(sl.stop),
                "n_rows": int(sl.stop - sl.start),
                "xlo": float(p[:, 0].min()), "xhi": float(p[:, 0].max()),
                "ylo": float(p[:, 1].min()), "yhi": float(p[:, 1].max()),
                "zlo": float(p[:, 2].min()), "zhi": float(p[:, 2].max()),
            })
        header["snapshots"][zt] = {"file": fname, "n_side": n_side,
                                   "chunk_index": chunk_index}

        ffname = f"fields_{zt}.npy"
        np.save(out / ffname, np.load(linear_dir / zt / "field.npy"))
        header["fields"][zt] = {"file": ffname}

        hsrc = linear_dir / zt / "halos.parquet"
        if hsrc.is_file():
            import shutil
            shutil.copy(hsrc, out / f"halos_{zt}.parquet")
            header["halos"][zt] = {"file": f"halos_{zt}.parquet"}

    lc = linear_dir / "lightcone.parquet"
    if lc.is_file():
        import shutil
        shutil.copy(lc, out / "lightcone.parquet")
        header["lightcone"] = {"file": "lightcone.parquet"}

    (out / "header.json").write_text(json.dumps(header, indent=2))
    return out
```

- [ ] **Step 3b: Add `PackedNpyAdapter` to `oufsim/native.py`** (after `NumpyFieldAdapter`):

```python
@register_adapter
class PackedNpyAdapter(NativeReaderAdapter):
    """packed_npy adapter — chunk-sorted particle slab + field `.npy`.

    ``path`` points at a concrete block file. Particle slabs are (N,6) in the
    canonical column order; ``read_rows`` memmaps and slices a contiguous row
    range (the chunk's range from the store index).
    """

    native_format = "packed_npy"
    _PART_COLS = ("x", "y", "z", "vx", "vy", "vz")

    def read_field_region(self, path, cell_slice):
        arr = np.load(path, mmap_mode="r")
        return np.array(arr[cell_slice])

    def read_rows(self, path, row_slice, columns=None):
        arr = np.load(path, mmap_mode="r")            # (N, 6), no full load
        block = np.array(arr[row_slice])              # only the row range
        cols = columns if columns is not None else self._PART_COLS
        idx = {name: j for j, name in enumerate(self._PART_COLS)}
        return {name: block[:, idx[name]] for name in cols}
```

- [ ] **Step 4: Run — PASS.** `pytest test/test_lin_pack.py -v`

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/simulation/linear/pack.py oneuniverse/simulation/oufsim/native.py test/test_lin_pack.py
git commit -m "phaseS17/T2: packed_npy native format (chunk-sorted slab + header) + PackedNpyAdapter"
```

---

## Task 3: Format-agnostic store builder (`build_store` + `NativeProduct`)

The store can only be built from the linear layout (`write_oufsim_store`). Add a builder driven by **product descriptors** that pull native data through *any* mechanism and call the **same** per-product writers — the seam every backend reuses.

**Files:** Create `oufsim/build.py`; Modify `oufsim/__init__.py`; Test `test/test_oufsim_build.py`.

- [ ] **Step 1: Write the failing test**

```python
# test/test_oufsim_build.py
"""S17 T3 — format-agnostic build_store reproduces the linear store reads."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear import generate_linear_sim
from oneuniverse.simulation.oufsim import SimStore, write_oufsim_store
from oneuniverse.simulation.oufsim.build import NativeProduct, build_store
from oneuniverse.simulation.selectors import Cube
from oneuniverse.simulation.unit_frame import UnitFrameSpec


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_build_store_matches_write_oufsim_store(tmp_path):
    lin = generate_linear_sim(tmp_path / "lin", _cosmo(), box_size=200.0,
                              n_grid=32, redshifts=(0.0,), seed=2,
                              with_lightcone=False)
    ref_store = write_oufsim_store(lin, tmp_path / "ref", sim_name="d",
                                   particle_chunk_nside=4)
    parts = np.load(lin / "z0.000" / "particles.npy")
    field = np.load(lin / "z0.000" / "field.npy")
    products = [
        NativeProduct(name="snapshots", kind="catalog", z=0.0,
                      load=lambda parts=parts: {
                          "x": parts[:, 0], "y": parts[:, 1], "z": parts[:, 2],
                          "vx": parts[:, 3], "vy": parts[:, 4], "vz": parts[:, 5]},
                      columns=("x", "y", "z", "vx", "vy", "vz"), n_side=4),
        NativeProduct(name="fields", kind="field", z=0.0,
                      load=lambda field=field: field),
    ]
    built = build_store(tmp_path / "built", sim_name="d", cosmo=_cosmo(),
                        unit_frame=UnitFrameSpec(length_unit="Mpc/h",
                            mass_unit="Msun/h", velocity_unit="km/s peculiar",
                            frame="box"),
                        box_size=200.0, n_grid=32, redshifts=(0.0,),
                        products=products, code="test.builder")
    cube = Cube(0, 60, 0, 60, 0, 60)
    a = SimStore(ref_store).read_box("snapshots", 0.0, cube)
    b = SimStore(built).read_box("snapshots", 0.0, cube)
    assert len(a["x"]) == len(b["x"])
    fa, _ = SimStore(ref_store).read_field_box(0.0, cube)
    fb, _ = SimStore(built).read_field_box(0.0, cube)
    assert np.allclose(fa, fb)
```

- [ ] **Step 2: Run — FAIL** (`oufsim.build` absent).

- [ ] **Step 3: Implement `oufsim/build.py`**

```python
# oneuniverse/simulation/oufsim/build.py
"""Format-agnostic OUF-Sim store builder.

A converter describes its products as ``NativeProduct`` descriptors (a name,
a kind, a redshift, and a ``load`` callable that returns the native arrays via
*any* mechanism — direct numpy, a NativeReaderAdapter, an HDF5 reader). The
builder loops them and calls the **same** per-product writers as the linear
``write_oufsim_store``, then emits the manifest. This is the seam a real
backend reuses: implement an adapter + emit ``NativeProduct``s, get a store.

Core kinds: ``catalog`` (particles/halos, Cube-chunked), ``field`` (memmap
tiles or reference), ``lightcone`` (HEALPix-NEST). Extended products stay in
the linear writer (out of S17 scope).
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field as _field
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from oneuniverse.simulation._version import OUFSIM_FORMAT_VERSION
from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.manifest import OUFSimManifest
from oneuniverse.simulation.oufsim._io import write_json
from oneuniverse.simulation.oufsim.write import (
    INDEX_FILE, OUFSIM_SUBDIR, _write_chunked_catalog, _write_field_reference,
    _write_field_tiles, _write_lightcone,
)
from oneuniverse.simulation.provenance import ProvenanceSpec
from oneuniverse.simulation.unit_frame import UnitFrameSpec


@dataclass
class NativeProduct:
    name: str
    kind: str                                  # "catalog" | "field" | "lightcone"
    z: Optional[float]
    load: Callable[[], object]                 # -> dict cols | ndarray
    columns: Tuple[str, ...] = ()
    pos_keys: Tuple[str, str, str] = ("x", "y", "z")
    n_side: int = 4                            # catalog chunking
    tile_cells: int = 32                       # field tiling
    nside_part: int = 2                        # lightcone partition
    projection: str = "reencode"               # "reencode" | "reference"
    native_path: Optional[str] = None          # field reference target


def _ztag(z: float) -> str:
    return f"z{z:.3f}"


def build_store(
    out_root: Union[str, Path], *, sim_name: str, cosmo: CosmologySpec,
    unit_frame: UnitFrameSpec, box_size: float, n_grid: int,
    redshifts: Sequence[float], products: Sequence[NativeProduct],
    code: str, sim_kind: str = "pm", native_format: Optional[str] = None,
    n_threads: int = 1, overwrite: bool = False,
) -> Path:
    """Build an OUF-Sim store from product descriptors. Returns the store dir."""
    store = Path(out_root) / sim_name / OUFSIM_SUBDIR
    if store.exists():
        if not overwrite:
            raise FileExistsError(f"{store} exists; pass overwrite=True")
        import shutil
        shutil.rmtree(store)
    store.mkdir(parents=True)

    layout: Dict[str, dict] = {}
    product_names = []
    n_particles_total = 0

    for p in products:
        zt = _ztag(float(p.z)) if p.z is not None else None
        if p.kind == "catalog":
            cols = p.load()
            pos = np.stack([cols[k] for k in p.pos_keys], axis=1)
            info = _write_chunked_catalog(
                store / p.name / zt, cols, pos, box_size, p.n_side,
                n_threads=n_threads)
            info["dir"] = f"{p.name}/{zt}"
            info["index"] = f"{p.name}/{zt}/{INDEX_FILE}"
            layout.setdefault(p.name, {})[zt] = info
            if p.name == "snapshots":
                n_particles_total = max(n_particles_total, info["n_rows"])
        elif p.kind == "field":
            if p.projection == "reference":
                info = _write_field_reference(
                    store / p.name / zt, Path(p.native_path), n_grid, box_size)
            else:
                info = _write_field_tiles(
                    store / p.name / zt, np.asarray(p.load()), box_size,
                    p.tile_cells)
            info["dir"] = f"{p.name}/{zt}"
            info["index"] = f"{p.name}/{zt}/{INDEX_FILE}"
            layout.setdefault(p.name, {})[zt] = info
        elif p.kind == "lightcone":
            info = _write_lightcone(store / p.name, p.load(), p.nside_part)
            info["dir"] = p.name
            info["index"] = f"{p.name}/{INDEX_FILE}"
            layout[p.name] = info
        else:
            raise ValueError(f"build_store: unknown kind {p.kind!r}")
        if p.name not in product_names:
            product_names.append(p.name)

    manifest = OUFSimManifest(
        oufsim_format_version=OUFSIM_FORMAT_VERSION, sim_name=sim_name,
        sim_kind=sim_kind, code=code, code_version=None,
        layout_schema="per_cosmology_phase_snapshot", backends=(code,),
        has_input=False, has_output=True, products=tuple(product_names),
        n_snapshots=len(tuple(redshifts)), redshifts=tuple(float(z) for z in redshifts),
        box_size=float(box_size), n_particles=int(n_particles_total) or None,
        cosmology=cosmo, unit_frame=unit_frame,
        provenance=ProvenanceSpec(
            code=code, code_version=None, git_hash=None, original_paths=(),
            ingested_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(),
            converter="build_store"),
    )
    payload = manifest.to_dict()
    payload["store_layout"] = layout
    payload["n_grid"] = int(n_grid)
    if native_format is not None:
        payload["native_format"] = native_format
    write_json(store / "manifest.json", payload)
    return store
```

Then export in `oufsim/__init__.py`: add `from oneuniverse.simulation.oufsim.build import NativeProduct, build_store` and extend `__all__`.

- [ ] **Step 4: Run — PASS.** `pytest test/test_oufsim_build.py -v`

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/simulation/oufsim/build.py oneuniverse/simulation/oufsim/__init__.py test/test_oufsim_build.py
git commit -m "phaseS17/T3: format-agnostic build_store + NativeProduct (reuses per-product writers)"
```

---

## Task 4: Second backend — `PackedSimConverter` (proves "other simulators plug in")

A converter over `packed_npy` that implements the real `convert()` (the ABC stub) via the adapter + `build_store`, and reads **identically** to the linear store for the same underlying sim.

**Files:** Create `packed/__init__.py`, `packed/converter.py`; Test `test/test_packed_converter.py`.

- [ ] **Step 1: Write the failing test**

```python
# test/test_packed_converter.py
"""S17 T4 — a 2nd backend produces an equivalent store from a different format."""
import numpy as np

from oneuniverse.simulation.converter import detect_converter, get_converter
from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear import generate_linear_sim
from oneuniverse.simulation.linear.pack import write_packed_native
from oneuniverse.simulation.oufsim import SimStore, write_oufsim_store
from oneuniverse.simulation.packed.converter import PackedSimConverter  # noqa: F401
from oneuniverse.simulation.selectors import Cube


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_packed_converter_detects_and_registers(tmp_path):
    lin = generate_linear_sim(tmp_path / "lin", _cosmo(), box_size=200.0,
                              n_grid=32, redshifts=(0.0,), seed=2,
                              with_lightcone=False)
    pk = write_packed_native(lin, tmp_path / "pk", particle_chunk_nside=4)
    cls = detect_converter(pk)
    assert cls is not None and cls.code == PackedSimConverter.code
    assert get_converter("packed_npy") is PackedSimConverter


def test_packed_store_reads_match_linear(tmp_path):
    lin = generate_linear_sim(tmp_path / "lin", _cosmo(), box_size=200.0,
                              n_grid=32, redshifts=(0.0,), seed=2,
                              with_lightcone=False)
    pk = write_packed_native(lin, tmp_path / "pk", particle_chunk_nside=4)
    lin_store = write_oufsim_store(lin, tmp_path / "ls", sim_name="d",
                                   particle_chunk_nside=4)
    pk_store = PackedSimConverter().convert(pk, tmp_path / "ps", sim_name="d")
    cube = Cube(20, 90, 20, 90, 20, 90)
    a = SimStore(lin_store).read_box("snapshots", 0.0, cube,
                                     columns=("x", "y", "z"))
    b = SimStore(pk_store).read_box("snapshots", 0.0, cube,
                                    columns=("x", "y", "z"))
    # same set of particles in the cube (order may differ -> compare sorted)
    assert len(a["x"]) == len(b["x"])
    np.testing.assert_allclose(np.sort(a["x"]), np.sort(b["x"]))
    fa, _ = SimStore(lin_store).read_field_box(0.0, cube)
    fb, _ = SimStore(pk_store).read_field_box(0.0, cube)
    assert np.allclose(fa, fb)
```

- [ ] **Step 2: Run — FAIL** (`packed` package absent).

- [ ] **Step 3: Implement** `packed/__init__.py` (one line: `from oneuniverse.simulation.packed.converter import PackedSimConverter  # noqa: F401`) and `packed/converter.py`:

```python
# oneuniverse/simulation/packed/converter.py
"""PackedSimConverter — 2nd backend over the packed_npy native format.

Demonstrates that the store machinery is not coupled to the linear layout: a
new code is added by (1) a NativeReaderAdapter (here PackedNpyAdapter) and
(2) a SimConverter that emits NativeProduct descriptors + calls build_store.
A real backend (AbacusSummit ASDF, Gadget HDF5) follows the same recipe.
"""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Tuple

import numpy as np
import pyarrow.parquet as pq

from oneuniverse.simulation.capabilities import BackendCapabilities
from oneuniverse.simulation.converter import SimConverter, register
from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.execution import ExecutionMode
from oneuniverse.simulation.oufsim.build import NativeProduct, build_store
from oneuniverse.simulation.oufsim.native import get_adapter
from oneuniverse.simulation.product import ProductDecl
from oneuniverse.simulation.unit_frame import UnitFrameSpec


def _load_packed_catalog(slab_path: Path) -> dict:
    arr = np.load(slab_path)          # (N,6) chunk-sorted
    cols = ("x", "y", "z", "vx", "vy", "vz")
    return {c: arr[:, j] for j, c in enumerate(cols)}


def _load_packed_field(field_path: Path) -> np.ndarray:
    return np.load(field_path)


def _load_packed_halos(halo_path: Path) -> dict:
    t = pq.read_table(halo_path)
    return {n: t.column(n).to_numpy(zero_copy_only=False)
            for n in t.column_names}


def _load_lightcone(lc_path: Path) -> dict:
    t = pq.read_table(lc_path)
    return {n: t.column(n).to_numpy(zero_copy_only=False)
            for n in t.column_names}


@register
class PackedSimConverter(SimConverter):
    code = "packed_npy"
    sim_kind = "pm"
    capabilities = BackendCapabilities(
        name="packed_npy",
        native_format="packed_npy (chunk-sorted slab + header.json)",
        supports_random_access=True, supports_streaming=True,
        heavy_step_modes={
            "particle_chunking": (ExecutionMode.SEQUENTIAL,),
            "field_tiling": (ExecutionMode.SEQUENTIAL,),
        },
    )

    def detect(self, path: Path) -> bool:
        hdr = Path(path) / "header.json"
        if not hdr.is_file():
            return False
        try:
            raw = json.loads(hdr.read_text())
        except json.JSONDecodeError:
            return False
        return raw.get("native_format") == "packed_npy"

    def declare_products(self, src: Path) -> Tuple[ProductDecl, ...]:
        return (
            ProductDecl("snapshots", "packed_npy slab", ("cartesian_chunk",),
                        ("x", "y", "z", "vx", "vy", "vz")),
            ProductDecl("fields", "packed_npy mesh", ("grid_tile",), ("delta",)),
            ProductDecl("halos", "packed parquet", ("cartesian_chunk",),
                        ("halo_id", "x", "y", "z", "delta_peak", "mass")),
        )

    def read_cosmology(self, src: Path) -> CosmologySpec:
        raw = json.loads((Path(src) / "header.json").read_text())
        return CosmologySpec.from_dict(raw["cosmology"])

    def read_unit_frame(self, src: Path) -> UnitFrameSpec:
        return UnitFrameSpec(length_unit="Mpc/h", mass_unit="Msun/h",
                             velocity_unit="km/s peculiar", comoving=True,
                             frame="box")

    def convert(self, src: Path, out: Path, *, projection: str = "reencode",
                build_indexes: bool = True, sim_name: str = "packsim",
                overwrite: bool = False, **kwargs) -> Path:
        src = Path(src)
        hdr = json.loads((src / "header.json").read_text())
        get_adapter("packed_npy")          # validate the format is registered
        box = float(hdr["box_size"]); n_grid = int(hdr["n_grid"])
        redshifts = [float(z) for z in hdr["redshifts"]]
        products = []
        for zt, blk in hdr["snapshots"].items():
            z = float(zt[1:])
            products.append(NativeProduct(
                name="snapshots", kind="catalog", z=z,
                load=partial(_load_packed_catalog, src / blk["file"]),
                columns=("x", "y", "z", "vx", "vy", "vz"),
                n_side=int(blk.get("n_side", 4))))
        for zt, blk in hdr["fields"].items():
            products.append(NativeProduct(
                name="fields", kind="field", z=float(zt[1:]),
                load=partial(_load_packed_field, src / blk["file"])))
        for zt, blk in hdr.get("halos", {}).items():
            products.append(NativeProduct(
                name="halos", kind="catalog", z=float(zt[1:]),
                load=partial(_load_packed_halos, src / blk["file"]),
                columns=("halo_id", "x", "y", "z", "delta_peak", "mass"),
                n_side=2))
        if hdr.get("lightcone"):
            products.append(NativeProduct(
                name="lightcone", kind="lightcone", z=None,
                load=partial(_load_lightcone, src / hdr["lightcone"]["file"]),
                nside_part=2))
        return build_store(
            out, sim_name=sim_name, cosmo=self.read_cosmology(src),
            unit_frame=self.read_unit_frame(src), box_size=box, n_grid=n_grid,
            redshifts=redshifts, products=products, code=self.code,
            sim_kind=self.sim_kind, native_format="packed_npy",
            overwrite=overwrite)
```

- [ ] **Step 4: Run — PASS.** `pytest test/test_packed_converter.py -v`

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/simulation/packed/ test/test_packed_converter.py
git commit -m "phaseS17/T4: PackedSimConverter — 2nd backend; packed store reads match linear (generality proof)"
```

---

## Task 5: Particle wrap-in-place — index-only `reference` reads through the adapter

Close the S15 gap: with a chunk-sorted slab (T2), wrap particles **without copying** — the index carries `{native_file, row_start, row_stop, bbox, columns}` and the reader memmaps the slab through the adapter.

**Files:** Modify `oufsim/write.py` (`_write_chunked_catalog_reference`), `oufsim/build.py` (catalog `reference` branch), `oufsim/read.py` (`read_box` reference branch), `packed/converter.py` (offer `projection="reference"`); Test `test/test_oufsim_particle_reference.py`.

- [ ] **Step 1: Write the failing test**

```python
# test/test_oufsim_particle_reference.py
"""S17 T5 — particle reference projection is index-only and reads match."""
import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear import generate_linear_sim
from oneuniverse.simulation.linear.pack import write_packed_native
from oneuniverse.simulation.oufsim import SimStore
from oneuniverse.simulation.packed.converter import PackedSimConverter
from oneuniverse.simulation.selectors import Cube


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def _dir_size(p):
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def test_particle_reference_is_index_only_and_reads_match(tmp_path):
    lin = generate_linear_sim(tmp_path / "lin", _cosmo(), box_size=200.0,
                              n_grid=32, redshifts=(0.0,), seed=2,
                              with_lightcone=False)
    pk = write_packed_native(lin, tmp_path / "pk", particle_chunk_nside=4)
    enc = PackedSimConverter().convert(pk, tmp_path / "enc", sim_name="d",
                                       projection="reencode")
    ref = PackedSimConverter().convert(pk, tmp_path / "ref", sim_name="e",
                                       projection="reference")
    # snapshots dir of the reference store holds no copied parquet floats
    snap_ref = ref / "snapshots" / "z0.000"
    assert not any(f.suffix == ".parquet" and f.name.startswith("part_")
                   for f in snap_ref.iterdir())
    assert _dir_size(snap_ref) < 0.1 * _dir_size(enc / "snapshots" / "z0.000")
    # reads identical
    cube = Cube(10, 80, 10, 80, 10, 80)
    a = SimStore(enc).read_box("snapshots", 0.0, cube, columns=("x", "y", "z"))
    b = SimStore(ref).read_box("snapshots", 0.0, cube, columns=("x", "y", "z"))
    assert len(a["x"]) == len(b["x"])
    np.testing.assert_allclose(np.sort(a["x"]), np.sort(b["x"]))
```

- [ ] **Step 2: Run — FAIL** (no reference projection for catalogs).

- [ ] **Step 3a: `_write_chunked_catalog_reference` in `oufsim/write.py`**

```python
def _write_chunked_catalog_reference(prod_dir, chunk_index, native_file,
                                     columns) -> dict:
    """Index-only particle wrap: rows point at a chunk-sorted native slab.

    ``chunk_index`` = the packed header's per-chunk records (bbox + contiguous
    [row_start, row_stop)). No float data is copied — only the sidecar index.
    """
    prod_dir.mkdir(parents=True, exist_ok=True)
    nf = str(Path(native_file).resolve())
    rows = []
    for c in chunk_index:
        rows.append({
            "chunk_id": int(c["chunk_id"]), "cx": c["cx"], "cy": c["cy"],
            "cz": c["cz"], "xlo": c["xlo"], "xhi": c["xhi"], "ylo": c["ylo"],
            "yhi": c["yhi"], "zlo": c["zlo"], "zhi": c["zhi"],
            "n_rows": int(c["n_rows"]), "file": "", "native_file": nf,
            "native_format": "packed_npy",
            "row_start": int(c["row_start"]), "row_stop": int(c["row_stop"]),
            "columns": list(columns),
        })
    _write_index(prod_dir / INDEX_FILE, rows)
    return {"partition": "cartesian_chunk_reference", "n_chunks": len(rows),
            "n_rows": int(sum(c["n_rows"] for c in chunk_index)),
            "projection": "reference"}
```

- [ ] **Step 3b: catalog `reference` in `build_store`** — extend `NativeProduct` with `chunk_index: Optional[list] = None`, and in the `kind == "catalog"` branch:

```python
        if p.kind == "catalog":
            if p.projection == "reference":
                from oneuniverse.simulation.oufsim.write import (
                    _write_chunked_catalog_reference)
                info = _write_chunked_catalog_reference(
                    store / p.name / zt, p.chunk_index, p.native_path,
                    p.columns)
            else:
                cols = p.load()
                pos = np.stack([cols[k] for k in p.pos_keys], axis=1)
                info = _write_chunked_catalog(
                    store / p.name / zt, cols, pos, box_size, p.n_side,
                    n_threads=n_threads)
            info["dir"] = f"{p.name}/{zt}"
            info["index"] = f"{p.name}/{zt}/{INDEX_FILE}"
            layout.setdefault(p.name, {})[zt] = info
            if p.name == "snapshots":
                n_particles_total = max(n_particles_total, info["n_rows"])
```

- [ ] **Step 3c: `read_box` reference branch in `oufsim/read.py`** — inside `_read_one`, before the parquet read:

```python
        def _read_one(r):
            nf = r.get("native_file")
            if nf and r.get("row_stop") is not None:        # particle reference
                from oneuniverse.simulation.oufsim.native import get_adapter
                ad = get_adapter(r.get("native_format", "packed_npy"))
                want = read_cols if read_cols is not None else r.get("columns")
                got = ad.read_rows(nf, slice(int(r["row_start"]),
                                             int(r["row_stop"])), want)
                # x/y/z always needed for the cube cut
                for k in ("x", "y", "z"):
                    if k not in got:
                        got.update(ad.read_rows(nf, slice(int(r["row_start"]),
                                   int(r["row_stop"])), ("x", "y", "z")))
                        break
                return got
            if use_gpu:
                ...        # existing cudf path unchanged
            ...            # existing pyarrow path unchanged
```

- [ ] **Step 3d: packed converter offers reference** — in `PackedSimConverter.convert`, when `projection == "reference"` build the snapshots `NativeProduct` with `projection="reference"`, `native_path=src/blk["file"]`, `chunk_index=blk["chunk_index"]`, `columns=("x","y","z","vx","vy","vz")` (halos/fields stay reencode — bulk→reference convention).

- [ ] **Step 4: Run — PASS.** `pytest test/test_oufsim_particle_reference.py test/test_packed_converter.py -v`

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/simulation/oufsim/write.py oneuniverse/simulation/oufsim/build.py oneuniverse/simulation/oufsim/read.py oneuniverse/simulation/packed/converter.py test/test_oufsim_particle_reference.py
git commit -m "phaseS17/T5: particle wrap-in-place (reference) — index-only over the sorted slab, reads via adapter"
```

---

## Task 6: ExecutionPlan budget → batch size + bounded-memory streaming read

`ExecutionPlan.memory_budget_bytes` is declared but no code derives a batch size from it. Add the derivation and prove a large selector read stays under the budget when streamed.

**Files:** Modify `oufsim/execution.py`; Test `test/test_execution_budget.py`, `test/test_oufsim_stream_budget.py`.

- [ ] **Step 1: Write the failing tests**

```python
# test/test_execution_budget.py
"""S17 T6 — budget -> batch derivation."""
import pytest

from oneuniverse.simulation.execution import ExecutionMode, ExecutionPlan


def test_batch_for_derives_from_budget():
    plan = ExecutionPlan(mode=ExecutionMode.SEQUENTIAL,
                         memory_budget_bytes=1_000_000)
    # 6 float64 cols = 48 bytes/row; 1e6 budget / 48 ~ 20833, with safety < that
    n = plan.batch_for(bytes_per_row=48)
    assert 0 < n <= 1_000_000 // 48


def test_explicit_batch_rows_wins():
    plan = ExecutionPlan(mode=ExecutionMode.SEQUENTIAL,
                         memory_budget_bytes=1_000_000, batch_rows=512)
    assert plan.batch_for(bytes_per_row=48) == 512


def test_bytes_per_row_must_be_positive():
    plan = ExecutionPlan(mode=ExecutionMode.SEQUENTIAL,
                         memory_budget_bytes=1_000_000)
    with pytest.raises(ValueError):
        plan.batch_for(bytes_per_row=0)
```

```python
# test/test_oufsim_stream_budget.py
"""S17 T6 — streaming a big read honours the memory budget."""
from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.execution import ExecutionMode, ExecutionPlan
from oneuniverse.simulation.linear import generate_linear_sim
from oneuniverse.simulation.oufsim import SimStore, SimDatasetView, write_oufsim_store
from oneuniverse.simulation.selectors import Cube


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_streamed_batches_respect_budget(tmp_path):
    lin = generate_linear_sim(tmp_path / "lin", _cosmo(), box_size=300.0,
                              n_grid=48, redshifts=(0.0,), seed=2,
                              with_lightcone=False)
    store = write_oufsim_store(lin, tmp_path / "s", sim_name="d",
                               particle_chunk_nside=2)
    plan = ExecutionPlan(mode=ExecutionMode.SEQUENTIAL,
                         memory_budget_bytes=48 * 4096)   # ~4096 rows/batch
    view = SimDatasetView(store)
    cube = Cube(0, 300, 0, 300, 0, 300)                   # whole box
    sizes = [len(b["x"]) for b in view.iter_box("snapshots", 0.0, cube,
                                                 plan=plan)]
    full = SimStore(store).read_box("snapshots", 0.0, cube)
    assert sum(sizes) == len(full["x"])
    assert max(sizes) <= 4096                             # each batch bounded
```

- [ ] **Step 2: Run — FAIL** (`batch_for` unknown).

- [ ] **Step 3: Implement** — add to `ExecutionPlan` in `execution.py`:

```python
    def batch_for(self, bytes_per_row: int, *, safety: float = 0.5) -> int:
        """Rows per streamed batch under the memory budget.

        ``safety`` reserves headroom for transient copies (concat, masks).
        An explicit ``batch_rows`` overrides the derivation.
        """
        if bytes_per_row <= 0:
            raise ValueError(f"bytes_per_row must be > 0, got {bytes_per_row!r}")
        if self.batch_rows is not None:
            return self.batch_rows
        n = int(self.memory_budget_bytes * safety // bytes_per_row)
        return max(1, n)
```

Then in `SimDatasetView.iter_box`, when `plan is not None and plan.batch_rows is None`, derive `batch_rows = plan.batch_for(bytes_per_row=_BYTES_PER_ROW)` where `_BYTES_PER_ROW = 6 * 8` (module constant; 6 float64 particle columns). Keep the explicit-`batch_rows` path unchanged.

- [ ] **Step 4: Run — PASS.** `pytest test/test_execution_budget.py test/test_oufsim_stream_budget.py -v`

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/simulation/execution.py oneuniverse/simulation/oufsim/view.py test/test_execution_budget.py test/test_oufsim_stream_budget.py
git commit -m "phaseS17/T6: ExecutionPlan.batch_for (budget->batch) + budget-honouring streamed reads"
```

---

## Task 7: MPI rank-partitioned reads + GPU read hook

Rule 3 (MPI/GPU first-class) is declared on the write side and stubbed on read. Make `read_box` rank-partition its chunk reads (each rank reads its own chunks; testable via a pure helper without `mpi4py`) and keep the GPU device hook honest.

**Files:** Modify `oufsim/read.py`; Create `oufsim/_partition.py`; Test `test/test_oufsim_rank_partition.py`.

- [ ] **Step 1: Write the failing test**

```python
# test/test_oufsim_rank_partition.py
"""S17 T7 — deterministic rank partitioning of chunk reads."""
from oneuniverse.simulation.oufsim._partition import partition_by_rank


def test_partition_is_disjoint_and_complete():
    rows = list(range(10))
    parts = [partition_by_rank(rows, rank=r, size=3) for r in range(3)]
    assert sorted(x for p in parts for x in p) == rows      # complete
    seen = set()
    for p in parts:
        assert not (set(p) & seen)                          # disjoint
        seen |= set(p)


def test_single_rank_gets_everything():
    rows = list(range(5))
    assert partition_by_rank(rows, rank=0, size=1) == rows
```

- [ ] **Step 2: Run — FAIL** (`_partition` absent).

- [ ] **Step 3: Implement** `oufsim/_partition.py`:

```python
# oneuniverse/simulation/oufsim/_partition.py
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
```

Then in `read_box`, add an `mpi: bool = False` parameter; when `mpi` and `mpi4py` importable, resolve `rank, size` from `MPI.COMM_WORLD` and replace `hit` with `partition_by_rank(hit, rank=rank, size=size)` before the read, recording `last_read_stats["rank"]`/`["size"]`. Absent `mpi4py` → no-op (single rank). The GPU `device="gpu"` branch already falls back to CPU and records the resolved device — leave it; add a `last_read_stats["device"]` assertion-friendly note in the docstring that real GPUDirect needs hardware (untestable in CI; honest).

- [ ] **Step 4: Run — PASS.** `pytest test/test_oufsim_rank_partition.py -v`

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/simulation/oufsim/_partition.py oneuniverse/simulation/oufsim/read.py test/test_oufsim_rank_partition.py
git commit -m "phaseS17/T7: MPI rank-partitioned reads (pure partition_by_rank helper) + honest GPU hook"
```

---

## Task 8: Scale benchmark + wrap-vs-reencode + plots + close-out

Prove the bounded-memory and wrap-in-place claims with **numbers across both backends**, and ship a diagnostic figure (visual-testing convention).

**Files:** Create `oufsim/scale_bench.py`, `scripts/build_s17_demo.py`; Test `test/test_oufsim_scale_bench.py`, `test/test_visual_s17.py`; Modify `CLAUDE.md`, `plans/README.md`, memory.

- [ ] **Step 1: `oufsim/scale_bench.py`**

```python
# oneuniverse/simulation/oufsim/scale_bench.py
"""Scale-sweep: convert + read wall/peak vs grid size, and store size by
projection. Returns plain dicts so a script can plot + a test can assert
bounded growth (Rule 5)."""
from __future__ import annotations

import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.linear import generate_linear_sim
from oneuniverse.simulation.linear.pack import write_packed_native
from oneuniverse.simulation.oufsim import SimStore
from oneuniverse.simulation.packed.converter import PackedSimConverter
from oneuniverse.simulation.selectors import Cube


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def sweep(tmp: Path, cosmo: CosmologySpec, grids: Sequence[int],
          *, box: float = 300.0) -> List[Dict]:
    out = []
    for ng in grids:
        lin = generate_linear_sim(tmp / f"lin{ng}", cosmo, box_size=box,
                                  n_grid=ng, redshifts=(0.0,), seed=2,
                                  with_lightcone=False)
        pk = write_packed_native(lin, tmp / f"pk{ng}", particle_chunk_nside=4)
        tracemalloc.start(); t0 = time.perf_counter()
        enc = PackedSimConverter().convert(pk, tmp / f"enc{ng}", sim_name="d",
                                           projection="reencode")
        wall = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        ref = PackedSimConverter().convert(pk, tmp / f"ref{ng}", sim_name="e",
                                           projection="reference")
        cube = Cube(0, box / 4, 0, box / 4, 0, box / 4)
        SimStore(enc).read_box("snapshots", 0.0, cube)
        out.append({
            "n_grid": ng, "n_particles": ng ** 3,
            "convert_wall_s": round(wall, 4), "convert_peak_mb": peak / 1e6,
            "store_reencode_mb": _dir_size(enc) / 1e6,
            "store_reference_mb": _dir_size(ref) / 1e6,
            "native_mb": _dir_size(pk) / 1e6,
        })
    return out
```

- [ ] **Step 2: Bounded-growth test**

```python
# test/test_oufsim_scale_bench.py
"""S17 T8 — convert peak memory grows sub-linearly; reference << reencode."""
from oneuniverse.simulation.cosmology import CosmologySpec
from oneuniverse.simulation.oufsim.scale_bench import sweep


def _cosmo():
    return CosmologySpec(omega_m=0.31, omega_b=0.048, h=0.67, n_s=0.96,
                         sigma8=0.81, t_cmb=2.7255)


def test_reference_store_is_tiny_vs_reencode(tmp_path):
    rows = sweep(tmp_path, _cosmo(), grids=(16, 32))
    for r in rows:
        assert r["store_reference_mb"] < 0.5 * r["store_reencode_mb"]
    # peak memory grows slower than particle count's 8x (16->32)
    ratio_mem = rows[1]["convert_peak_mb"] / max(rows[0]["convert_peak_mb"], 1e-6)
    assert ratio_mem < 8.0
```

- [ ] **Step 3: Demo script + plot** — `scripts/build_s17_demo.py`: run `sweep(grids=(32,48,64))` under `/home/ravoux/Documents/Science/Cosmography/oneuniverse_simulation/s17_demo`, write `RUN_SUMMARY.json`, and a 2-panel figure (`s17_scaling.png`): (left) `convert_peak_mb` + `convert_wall_s` vs `n_particles`; (right) bar chart `native / reference / reencode` store MB. Save the figure into `test/test_output/`.

- [ ] **Step 4: Visual test**

```python
# test/test_visual_s17.py
"""S17 T8 — the scaling diagnostic figure exists and is non-trivial."""
from pathlib import Path


def test_s17_figure_exists():
    p = Path(__file__).parent / "test_output" / "s17_scaling.png"
    assert p.is_file() and p.stat().st_size > 5_000
```

(The demo script writes the PNG; run it before this test, or have the test invoke `sweep` + render if absent.)

- [ ] **Step 5: Full suite green**

```bash
pytest -q 2>&1 | tail -3
```

- [ ] **Step 6: Docs + memory close-out**
  - `Packages/oneuniverse/CLAUDE.md` — `oufsim/` bullet: note the **adapter registry + 2nd backend (`packed_npy`) + `build_store` + particle reference**; add a `packed/` bullet.
  - `plans/README.md` — add S17 row, mark complete.
  - Append to memory `project_oneuniverse_stabilisation.md` (or a new `project_oufsim_multibackend.md`): "S17 — storage substrate is multi-backend (linear + packed_npy via `build_store`/adapter registry); particle wrap-in-place index-only; ExecutionPlan budget→batch; MPI rank-partition + GPU hooks. Real backends (ASDF/HDF5/BigFile) implement `NativeReaderAdapter` + a `SimConverter` + emit `NativeProduct`s."

- [ ] **Step 7: Commit**

```bash
git add oneuniverse/simulation/oufsim/scale_bench.py scripts/build_s17_demo.py test/test_oufsim_scale_bench.py test/test_visual_s17.py CLAUDE.md plans/README.md
git commit -m "phaseS17/T8: scale benchmark + wrap-vs-reencode plots + multi-backend close-out docs"
```

---

## Success criteria

- A second, structurally-different native format (`packed_npy`) is ingested by a second converter (`PackedSimConverter`) through `build_store` + the adapter registry, and its store **reads identically** to the linear store for the same sim.
- `SimConverter.convert()` is implemented for ≥1 backend on the generic builder (no longer a stub for packed).
- Particles wrap **in place**: the `reference` snapshots store is index-only (<10% of re-encode) and reads match the re-encode path exactly (S15 gap closed for the sorted-native case).
- A read honours `ExecutionPlan.memory_budget_bytes` (batch derived; max batch bounded).
- Reads rank-partition deterministically (pure `partition_by_rank`); GPU hook falls back honestly.
- Scale sweep shows convert peak memory growing sub-linearly in particle count and `reference` ≪ `reencode` store size; a diagnostic figure is committed.
- **Rule 1 guard green every task**; only `mpi4py`/`cudf` added, both import-guarded.

## Maps to pinned Pillar-3 rules

| Rule | Where satisfied |
|---|---|
| 1 — minimal coupling | guard green; `oufsim/` + `packed/` use only numpy/pyarrow/healpy/pyyaml |
| 2 — partial access load-bearing | reference reads memmap only the chunk's row range / field tile |
| 3 — MPI/GPU first-class | T7 rank-partitioned reads + GPU device hook |
| 4 — no mini-sim runs | none added; storage/IO only |
| 5 — optimisation load-bearing | T5 wrap-in-place, T6 budget→batch bounded memory, T8 measured scaling |

## Generality contract (the seam a real simulator implements)

To add a real code (AbacusSummit ASDF, Gadget HDF5, BigFile):
1. Implement a `NativeReaderAdapter` (`read_field_region` + `read_rows`) over the native format; `@register_adapter`.
2. Implement a `SimConverter` (`detect`/`declare_products`/`read_cosmology`/`read_unit_frame` + `convert()` emitting `NativeProduct`s into `build_store`); `@register`.
3. If the native is spatially pre-sorted (Abacus cells / Gadget Hilbert), use `projection="reference"` for the bulk products → index-only, no copy. Otherwise `reencode`.
No store, reader, or index code changes. `packed_npy` is the worked example.

## Self-review checklist

- [ ] `get_adapter`/`register_adapter` + `read_rows` land in T1; used by T2/T5/T7.
- [ ] `NativeProduct`/`build_store` signature in T3 matches the calls in T4 (`PackedSimConverter.convert`) and T5 (catalog reference branch).
- [ ] `_write_chunked_catalog_reference` (T5) consumes the packed header `chunk_index` shape written in T2 (`row_start`/`row_stop`/bbox/`chunk_id`).
- [ ] `read_box` reference branch (T5) reads `native_format`/`row_start`/`row_stop`/`columns` index fields written in T5.
- [ ] `ExecutionPlan.batch_for` (T6) consumed by `SimDatasetView` with `_BYTES_PER_ROW`.
- [ ] No placeholders; every code step is complete.
- [ ] Scope: only core products; extended products explicitly out of scope.
