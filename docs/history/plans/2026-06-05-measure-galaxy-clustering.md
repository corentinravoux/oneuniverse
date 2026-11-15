# `oneuniverse.measure` — Galaxy Clustering P1→P2 Connection (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first P1→P2 connection — `oneuniverse.measure` — that turns an OUF POINT dataset into a complete, cosmology-free **MeasurementSet** for **galaxy clustering** (the most-used probe), via the 9-step transform (select / clean / weight / randoms / window / n(z) / region / spec / assemble).

**Architecture:** A new `oneuniverse/measure/` subpackage (lives *inside* oneuniverse per the 2026-06-05 owner decision; it defines the general output format other packages adapt to). It reads P1 via `DatasetView` + `oneuniverse.combine` weights, and emits a `MeasurementSet` holding a `PointSet` (catalog + randoms + n(z) + window), a shared HEALPix `region_map`, a `MeasurementSpec`, metadata, and provenance. **Cosmology-free** — the no-cosmology-in-Pillar-1 rule holds; z→r and the estimator math stay in P2. **Randoms: ingest OR generate**, user's choice. TDD against **synthetic OUF POINT fixtures** (real DESI/eBOSS validation is a follow-up plan).

**Tech Stack:** numpy, pandas, pyarrow, healpy (all present in oneuniverse). Reuses `oneuniverse.data` (DatasetView, write_ouf_dataset) + `oneuniverse.combine` (weight primitives). No new deps. No cosmology engine.

---

## Confirmed P1 APIs this plan builds on

```python
from oneuniverse.data.format_spec import DataGeometry          # DataGeometry.POINT
from oneuniverse.data.converter import write_ouf_dataset        # build synthetic OUF in tests
from oneuniverse.data.dataset_view import DatasetView           # .read(columns=, z_range=, cone=, ...) -> pd.DataFrame
from oneuniverse.data.manifest import LoaderSpec                # LoaderSpec(name=, version=)
from oneuniverse.combine.weights import (                       # Weight.__call__(df)->np.ndarray; w1*w2 -> ProductWeight
    FKPWeight, ColumnWeight, ConstantWeight, ProductWeight)
```
- CORE POINT columns: `ra, dec, z, z_type, z_err, galaxy_id, survey_id` (converter adds `_original_row_index, _healpix32`).
- `write_ouf_dataset(df=, out_dir=, survey_name=, survey_type=, geometry=DataGeometry.POINT, loader=LoaderSpec(...))`; `out_dir` is the `oneuniverse/` dir; read with `DatasetView.from_path(out_dir.parent)`.
- `Range` filters: `DatasetView.read(z_range=(lo,hi), ...)`.

---

## File structure (all new under `oneuniverse/measure/`)

| File | Responsibility |
|---|---|
| `metadata.py` | `ProductMetadata` (frame/epoch/unit/region NSIDE — **no cosmology**) + `Provenance` (dataset ids, weight recipe, randoms source, n(z) method) |
| `dataproduct.py` | `DataProduct` ABC + `PointSet` (catalog, randoms, nz, window) |
| `window.py` | `Window` (HEALPix footprint) + `footprint_from_positions` |
| `nz.py` | `Nz` (radial selection) + `nz_from_spec_z` |
| `randoms.py` | `generate_randoms` (window × n(z)) + `randoms_from_view` (ingest official) |
| `regions.py` | `assign_regions` (HEALPix `region_id`) |
| `spec.py` | `MeasurementSpec` (tracers, pairs, statistic, binning, coords, covariance, estimator_family) |
| `measurement_set.py` | `MeasurementSet` (products, spec, region_map, metadata) + `check_invariants` |
| `clustering.py` | `build_galaxy_clustering(...)` — the 9-step end-to-end connection |
| `__init__.py` | exports |
| `test/fixtures/measure_ouf.py` | synthetic OUF POINT view factory |

---

## Pre-flight

- [ ] **Step 0: Baseline green + scaffold.**

```bash
cd /home/ravoux/Documents/Python/Packages/oneuniverse
pytest test/test_data_*.py test/test_combine_*.py -q 2>&1 | tail -3   # P1 green
mkdir -p oneuniverse/measure
printf '"""oneuniverse.measure — the P1->P2 connection (cosmology-free)."""\n' > oneuniverse/measure/__init__.py
```

---

## Task 1: Synthetic OUF POINT fixture + `PointSet` carrier

**Files:** Create `test/fixtures/measure_ouf.py`, `oneuniverse/measure/metadata.py`, `oneuniverse/measure/dataproduct.py`; Test `test/test_measure_pointset.py`.

- [ ] **Step 1: Write the fixture**

```python
# test/fixtures/measure_ouf.py
"""Synthetic OUF POINT dataset → DatasetView, for measure/ tests."""
from pathlib import Path

import numpy as np
import pandas as pd

from oneuniverse.data.converter import write_ouf_dataset
from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.data.format_spec import DataGeometry
from oneuniverse.data.manifest import LoaderSpec


def synthetic_point_view(tmp: Path, *, n: int = 3000, seed: int = 0,
                         name: str = "synth") -> DatasetView:
    """Write a synthetic galaxy OUF POINT dataset; return its DatasetView."""
    rng = np.random.default_rng(seed)
    # a compact footprint patch + a smooth redshift cloud
    ra = rng.uniform(150.0, 170.0, n)
    dec = rng.uniform(0.0, 15.0, n)
    z = np.clip(rng.normal(0.5, 0.12, n), 0.05, 1.2)
    df = pd.DataFrame({
        "ra": ra, "dec": dec, "z": z,
        "z_type": np.full(n, "spec"), "z_err": np.full(n, 1e-4),
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": np.zeros(n, dtype=np.int64),
        "nbar": np.full(n, 1e-3),                 # for FKP
        "weight_comp": rng.uniform(0.9, 1.0, n),  # completeness
        "weight_sys": rng.uniform(0.95, 1.05, n), # imaging systematics
        "quality": (rng.uniform(size=n) > 0.02).astype(np.int64),  # 2% bad
    })
    out = tmp / name / "oneuniverse"
    write_ouf_dataset(df=df, out_dir=out, survey_name=name,
                      survey_type="spectroscopic", geometry=DataGeometry.POINT,
                      loader=LoaderSpec(name=name, version="0"))
    return DatasetView.from_path(out.parent)
```

- [ ] **Step 2: Write the failing test**

```python
# test/test_measure_pointset.py
"""measure T1 — synthetic view + PointSet carrier."""
import numpy as np

from oneuniverse.measure.dataproduct import PointSet
from oneuniverse.measure.metadata import ProductMetadata, Provenance
from test.fixtures.measure_ouf import synthetic_point_view


def test_synthetic_view_reads_point(tmp_path):
    view = synthetic_point_view(tmp_path, n=500, seed=1)
    df = view.read(columns=["ra", "dec", "z"])
    assert len(df) == 500 and {"ra", "dec", "z"} <= set(df.columns)


def test_pointset_holds_catalog_and_metadata(tmp_path):
    view = synthetic_point_view(tmp_path, n=500, seed=1)
    df = view.read()
    ps = PointSet(
        catalog=df, randoms=None, nz=None, window=None,
        region_map=np.zeros(len(df), dtype=np.int64),
        metadata=ProductMetadata(frame="icrs", epoch=2000.0,
                                 length_unit="deg", nside_region=8),
        provenance=Provenance(dataset_ids=("synth",)),
    )
    assert ps.kind == "pointset"
    assert ps.metadata.frame == "icrs"
    assert "cosmology" not in vars(ps.metadata)   # cosmology-free invariant
```

- [ ] **Step 3: Run — FAIL** (`oneuniverse.measure.dataproduct` absent).

- [ ] **Step 4: Implement `metadata.py`**

```python
# oneuniverse/measure/metadata.py
"""Observational metadata + provenance for measure DataProducts. NO cosmology."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class ProductMetadata:
    frame: str                       # icrs / galactic / ecliptic
    epoch: float                     # e.g. 2000.0, 2016.0
    length_unit: str                 # "deg" on-sky; comoving conversion is P2
    nside_region: int                # HEALPix NSIDE of the region_map


@dataclass(frozen=True)
class Provenance:
    dataset_ids: Tuple[str, ...]
    weight_recipe: Tuple[str, ...] = ()
    randoms_source: Optional[str] = None      # "ingested" | "generated" | None
    nz_method: Optional[str] = None           # "spec_hist" | ...
    extra: dict = field(default_factory=dict)
```

- [ ] **Step 5: Implement `dataproduct.py`**

```python
# oneuniverse/measure/dataproduct.py
"""DataProduct ABC + PointSet (galaxy clustering carrier). Cosmology-free."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np
import pandas as pd

from oneuniverse.measure.metadata import ProductMetadata, Provenance


@dataclass
class DataProduct(abc.ABC):
    region_map: np.ndarray
    metadata: ProductMetadata
    provenance: Provenance

    kind: ClassVar[str] = "abstract"


@dataclass
class PointSet(DataProduct):
    catalog: pd.DataFrame = None
    randoms: Optional[pd.DataFrame] = None
    nz: object = None                 # Nz | None
    window: object = None             # Window | None

    kind: ClassVar[str] = "pointset"
```

(Dataclass field order: put the `DataProduct` base fields first via keyword construction in the test — the test constructs with all keywords, so ordering is not load-bearing. If the engineer hits a dataclass inheritance ordering error, give `PointSet` explicit keyword-only fields with `@dataclass(kw_only=True)` — Python ≥3.10, which this repo targets.)

- [ ] **Step 6: Run — PASS.**

Run: `pytest test/test_measure_pointset.py -v`

- [ ] **Step 7: Commit**

```bash
git add oneuniverse/measure/__init__.py oneuniverse/measure/metadata.py oneuniverse/measure/dataproduct.py test/fixtures/measure_ouf.py test/test_measure_pointset.py
git commit -m "measure/T1: synthetic OUF POINT fixture + PointSet/DataProduct carrier (cosmology-free)"
```

---

## Task 2: Select + clean

Turn a `DatasetView` into a cleaned tracer catalog: z-range select + quality cut + drop sentinels.

**Files:** Create `oneuniverse/measure/select.py`; Test `test/test_measure_select.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_measure_select.py
"""measure T2 — select + clean."""
from oneuniverse.measure.select import select_clean
from test.fixtures.measure_ouf import synthetic_point_view


def test_select_clean_applies_zrange_and_quality(tmp_path):
    view = synthetic_point_view(tmp_path, n=4000, seed=2)
    cat = select_clean(view, z_range=(0.4, 0.7),
                        quality_column="quality", quality_min=1)
    assert cat["z"].between(0.4, 0.7).all()
    assert (cat["quality"] >= 1).all()
    assert cat["z"].notna().all()
    assert len(cat) < view.n_rows           # cuts removed rows
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement**

```python
# oneuniverse/measure/select.py
"""Step 1-2 of the P1->P2 transform: select a tracer + clean it."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import pandas as pd

from oneuniverse.data.dataset_view import DatasetView


def select_clean(view: DatasetView, *,
                 z_range: Optional[Tuple[float, float]] = None,
                 columns: Optional[Sequence[str]] = None,
                 quality_column: Optional[str] = None,
                 quality_min: float = 1.0,
                 dropna: bool = True) -> pd.DataFrame:
    """Read + clean a tracer catalog from an OUF POINT view.

    Pushes z_range to the reader (partition pruning); applies the quality cut
    and drops NaN positions/redshifts in pandas.
    """
    cat = view.read(columns=columns, z_range=z_range)
    if quality_column is not None and quality_column in cat.columns:
        cat = cat[cat[quality_column] >= quality_min]
    if dropna:
        cat = cat.dropna(subset=[c for c in ("ra", "dec", "z")
                                 if c in cat.columns])
    return cat.reset_index(drop=True)
```

- [ ] **Step 4: Run — PASS.** **Step 5: Commit** `measure/T2: select_clean — z-range + quality cut + dropna`.

---

## Task 3: Weights (assemble total weight)

Compose named weight primitives from `oneuniverse.combine` into one `weight` column; keep the recipe for provenance.

**Files:** Create `oneuniverse/measure/weighting.py`; Test `test/test_measure_weighting.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_measure_weighting.py
"""measure T3 — total weight assembly."""
import numpy as np

from oneuniverse.combine.weights import ColumnWeight, FKPWeight
from oneuniverse.measure.select import select_clean
from oneuniverse.measure.weighting import assemble_weight
from test.fixtures.measure_ouf import synthetic_point_view


def test_total_weight_is_product_of_components(tmp_path):
    view = synthetic_point_view(tmp_path, n=2000, seed=3)
    cat = select_clean(view, z_range=(0.1, 1.0))
    weights = [FKPWeight(nbar="nbar", P0=1e4), ColumnWeight("weight_comp"),
               ColumnWeight("weight_sys")]
    out, recipe = assemble_weight(cat, weights)
    expected = (1.0 / (1.0 + cat["nbar"].to_numpy() * 1e4)
                * cat["weight_comp"].to_numpy() * cat["weight_sys"].to_numpy())
    assert np.allclose(out["weight"].to_numpy(), expected)
    assert "FKPWeight" in recipe[0]
    assert (out["weight"] > 0).all()
```

(Verify `FKPWeight.__init__` parameter names against `oneuniverse/combine/weights/fkp.py` — adjust `nbar=`/`P0=`/`z_column=` to the actual signature in Step 3.)

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement**

```python
# oneuniverse/measure/weighting.py
"""Step 3: assemble a total weight from oneuniverse.combine primitives."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from oneuniverse.combine.weights import Weight


def assemble_weight(catalog: pd.DataFrame, weights: Sequence[Weight],
                    *, out_column: str = "weight"
                    ) -> Tuple[pd.DataFrame, Tuple[str, ...]]:
    """Return (catalog with `out_column` = product of weights, recipe)."""
    out = catalog.copy()
    total = np.ones(len(out), dtype=float)
    recipe = []
    for w in weights:
        total = total * np.asarray(w(out), dtype=float)
        recipe.append(repr(w))
    out[out_column] = total
    return out, tuple(recipe)
```

- [ ] **Step 4: Run — PASS.** **Step 5: Commit** `measure/T3: assemble_weight — product of combine weight primitives + recipe`.

---

## Task 4: Window / footprint

Build a HEALPix angular footprint (completeness) from object positions; expose covered fraction.

**Files:** Create `oneuniverse/measure/window.py`; Test `test/test_measure_window.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_measure_window.py
"""measure T4 — HEALPix footprint window."""
import numpy as np

from oneuniverse.measure.select import select_clean
from oneuniverse.measure.window import Window, footprint_from_positions
from test.fixtures.measure_ouf import synthetic_point_view


def test_footprint_covers_data_pixels(tmp_path):
    view = synthetic_point_view(tmp_path, n=3000, seed=4)
    cat = select_clean(view, z_range=(0.1, 1.0))
    win = footprint_from_positions(cat["ra"].to_numpy(), cat["dec"].to_numpy(),
                                   nside=64)
    assert isinstance(win, Window)
    assert win.mask.sum() > 0
    # every data point lands in a covered pixel
    assert win.contains(cat["ra"].to_numpy(), cat["dec"].to_numpy()).all()
    assert 0.0 < win.covered_fraction() < 1.0
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement**

```python
# oneuniverse/measure/window.py
"""Step 5: angular footprint as a HEALPix completeness mask."""
from __future__ import annotations

from dataclasses import dataclass

import healpy as hp
import numpy as np


def _ang2pix(ra, dec, nside):
    theta = np.radians(90.0 - np.asarray(dec))
    phi = np.radians(np.asarray(ra))
    return hp.ang2pix(nside, theta, phi, nest=True)


@dataclass(frozen=True)
class Window:
    nside: int
    mask: np.ndarray                 # float completeness per NEST pixel [0,1]

    def contains(self, ra, dec) -> np.ndarray:
        return self.mask[_ang2pix(ra, dec, self.nside)] > 0.0

    def covered_fraction(self) -> float:
        return float((self.mask > 0).sum()) / self.mask.size


def footprint_from_positions(ra, dec, *, nside: int = 256) -> Window:
    """Binary completeness: pixels containing >=1 object are covered."""
    npix = hp.nside2npix(nside)
    mask = np.zeros(npix, dtype=np.float64)
    pix = _ang2pix(ra, dec, nside)
    mask[np.unique(pix)] = 1.0
    return Window(nside=int(nside), mask=mask)
```

- [ ] **Step 4: Run — PASS.** **Step 5: Commit** `measure/T4: Window + footprint_from_positions (HEALPix completeness)`.

---

## Task 5: n(z) radial selection

Build a redshift histogram n(z) (weighted), recording the method for provenance.

**Files:** Create `oneuniverse/measure/nz.py`; Test `test/test_measure_nz.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_measure_nz.py
"""measure T5 — n(z) radial selection."""
import numpy as np

from oneuniverse.measure.nz import Nz, nz_from_spec_z


def test_nz_normalises_and_records_method():
    z = np.concatenate([np.full(100, 0.3), np.full(300, 0.5)])
    nz = nz_from_spec_z(z, edges=np.linspace(0.0, 1.0, 11))
    assert isinstance(nz, Nz)
    assert nz.method == "spec_hist"
    # the 0.5 bin holds ~3x the 0.3 bin
    i3 = np.digitize(0.3, nz.edges) - 1
    i5 = np.digitize(0.5, nz.edges) - 1
    assert nz.counts[i5] > 2.5 * nz.counts[i3]
    assert np.isclose(np.trapz(nz.pdf(), nz.centers()), 1.0, atol=0.2)


def test_nz_weighted():
    z = np.array([0.3, 0.3, 0.5])
    w = np.array([1.0, 1.0, 4.0])
    nz = nz_from_spec_z(z, edges=np.linspace(0.2, 0.6, 5), weights=w)
    i5 = np.digitize(0.5, nz.edges) - 1
    i3 = np.digitize(0.3, nz.edges) - 1
    assert nz.counts[i5] == 4.0 and nz.counts[i3] == 2.0
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement**

```python
# oneuniverse/measure/nz.py
"""Step 6: radial selection n(z). Records the estimation method (provenance)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Nz:
    edges: np.ndarray
    counts: np.ndarray               # weighted counts per bin
    method: str                      # "spec_hist" | "photo_stack" | "clustering_z"

    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    def pdf(self) -> np.ndarray:
        width = np.diff(self.edges)
        area = float((self.counts * width).sum())
        return self.counts / area if area > 0 else self.counts


def nz_from_spec_z(z, *, edges, weights: Optional[np.ndarray] = None) -> Nz:
    counts, _ = np.histogram(np.asarray(z), bins=edges, weights=weights)
    return Nz(edges=np.asarray(edges, float), counts=counts.astype(float),
              method="spec_hist")
```

- [ ] **Step 4: Run — PASS.** **Step 5: Commit** `measure/T5: Nz + nz_from_spec_z (weighted histogram, method provenance)`.

---

## Task 6: Randoms — ingest OR generate

Two first-class paths (owner decision): ingest survey-published randoms from an OUF view, or generate from window × n(z). Both tag provenance.

**Files:** Create `oneuniverse/measure/randoms.py`; Test `test/test_measure_randoms.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_measure_randoms.py
"""measure T6 — randoms ingest + generate."""
import numpy as np

from oneuniverse.measure.nz import nz_from_spec_z
from oneuniverse.measure.randoms import generate_randoms, randoms_from_view
from oneuniverse.measure.select import select_clean
from oneuniverse.measure.window import footprint_from_positions
from test.fixtures.measure_ouf import synthetic_point_view


def test_generate_randoms_inside_window_and_nz(tmp_path):
    view = synthetic_point_view(tmp_path, n=3000, seed=6)
    cat = select_clean(view, z_range=(0.1, 1.0))
    win = footprint_from_positions(cat["ra"].to_numpy(), cat["dec"].to_numpy(),
                                   nside=64)
    nz = nz_from_spec_z(cat["z"].to_numpy(), edges=np.linspace(0.0, 1.2, 25))
    rnd, source = generate_randoms(win, nz, n_randoms=20000, seed=1)
    assert source == "generated"
    assert win.contains(rnd["ra"].to_numpy(), rnd["dec"].to_numpy()).all()
    assert rnd["z"].min() >= 0.0 and rnd["z"].max() <= 1.2
    assert len(rnd) == 20000 and (rnd["weight"] == 1.0).all()


def test_ingest_randoms_from_view(tmp_path):
    rview = synthetic_point_view(tmp_path, n=5000, seed=99, name="rand")
    rnd, source = randoms_from_view(rview)
    assert source == "ingested" and len(rnd) == 5000
    assert {"ra", "dec", "z"} <= set(rnd.columns)
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement**

```python
# oneuniverse/measure/randoms.py
"""Step 4: randoms. Ingest survey-published, or generate from window x n(z).

Owner decision (2026-06-05): both first-class; provenance records which.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import healpy as hp
import numpy as np
import pandas as pd

from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.measure.nz import Nz
from oneuniverse.measure.window import Window


def randoms_from_view(view: DatasetView, *,
                      columns: Optional[Sequence[str]] = None
                      ) -> Tuple[pd.DataFrame, str]:
    """Ingest an official random catalog stored as an OUF POINT dataset."""
    rnd = view.read(columns=columns)
    return rnd.reset_index(drop=True), "ingested"


def generate_randoms(window: Window, nz: Nz, *, n_randoms: int,
                     seed: int = 0) -> Tuple[pd.DataFrame, str]:
    """Uniform-in-window angular positions × n(z)-sampled redshifts."""
    rng = np.random.default_rng(seed)
    covered = np.nonzero(window.mask > 0)[0]
    # sample covered pixels weighted by completeness, then uniform within pixel
    probs = window.mask[covered] / window.mask[covered].sum()
    pix = rng.choice(covered, size=n_randoms, p=probs)
    ra, dec = _uniform_in_pixels(pix, window.nside, rng)
    # inverse-CDF sample z from the n(z) histogram
    cdf = np.cumsum(nz.counts); cdf = cdf / cdf[-1]
    u = rng.uniform(size=n_randoms)
    bins = np.searchsorted(cdf, u)
    z = nz.edges[bins] + rng.uniform(size=n_randoms) * np.diff(nz.edges)[
        np.clip(bins, 0, len(nz.edges) - 2)]
    rnd = pd.DataFrame({"ra": ra, "dec": dec, "z": z,
                        "weight": np.ones(n_randoms)})
    return rnd, "generated"


def _uniform_in_pixels(pix, nside, rng):
    """Uniform sky point inside each NEST pixel (reject-free via pixel corners)."""
    # sample pixel centre + small jitter within the pixel's angular size
    theta, phi = hp.pix2ang(nside, pix, nest=True)
    res = hp.nside2resol(nside)                 # rad
    theta = np.clip(theta + (rng.uniform(size=len(pix)) - 0.5) * res, 1e-6,
                    np.pi - 1e-6)
    phi = (phi + (rng.uniform(size=len(pix)) - 0.5) * res) % (2 * np.pi)
    ra = np.degrees(phi)
    dec = 90.0 - np.degrees(theta)
    return ra, dec
```

(The jitter sampler keeps randoms inside the covered pixel set at the window's NSIDE; `window.contains` uses the same NSIDE so the test holds. For production a higher-NSIDE within-pixel sampler can replace `_uniform_in_pixels` — note but do not implement now, YAGNI.)

- [ ] **Step 4: Run — PASS.** **Step 5: Commit** `measure/T6: randoms — ingest (randoms_from_view) + generate (window x n(z)), provenance`.

---

## Task 7: Region map (shared jackknife)

Assign a HEALPix `region_id` to objects (and randoms) — the shared scheme for joint covariance.

**Files:** Create `oneuniverse/measure/regions.py`; Test `test/test_measure_regions.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_measure_regions.py
"""measure T7 — HEALPix region assignment."""
import numpy as np

from oneuniverse.measure.regions import assign_regions
from oneuniverse.measure.select import select_clean
from test.fixtures.measure_ouf import synthetic_point_view


def test_region_ids_are_stable_and_shared(tmp_path):
    view = synthetic_point_view(tmp_path, n=3000, seed=7)
    cat = select_clean(view, z_range=(0.1, 1.0))
    r1 = assign_regions(cat["ra"].to_numpy(), cat["dec"].to_numpy(), nside=4)
    r2 = assign_regions(cat["ra"].to_numpy(), cat["dec"].to_numpy(), nside=4)
    assert r1.dtype == np.int64 and len(r1) == len(cat)
    np.testing.assert_array_equal(r1, r2)          # deterministic
    assert r1.min() >= 0
    assert len(np.unique(r1)) > 1                   # patch spans >1 region
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement**

```python
# oneuniverse/measure/regions.py
"""Step 8: shared HEALPix region_id (jackknife/bootstrap basis)."""
from __future__ import annotations

import healpy as hp
import numpy as np


def assign_regions(ra, dec, *, nside: int = 8) -> np.ndarray:
    """NEST HEALPix pixel id at `nside` — the shared resampling scheme."""
    theta = np.radians(90.0 - np.asarray(dec))
    phi = np.radians(np.asarray(ra))
    return hp.ang2pix(nside, theta, phi, nest=True).astype(np.int64)
```

- [ ] **Step 4: Run — PASS.** **Step 5: Commit** `measure/T7: assign_regions (shared HEALPix jackknife ids)`.

---

## Task 8: MeasurementSpec + MeasurementSet + `build_galaxy_clustering`

Assemble the end-to-end connection and the bundle, with an invariants check.

**Files:** Create `oneuniverse/measure/spec.py`, `oneuniverse/measure/measurement_set.py`, `oneuniverse/measure/clustering.py`; Modify `oneuniverse/measure/__init__.py`; Test `test/test_measure_clustering.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_measure_clustering.py
"""measure T8 — galaxy-clustering MeasurementSet end-to-end."""
import numpy as np

from oneuniverse.combine.weights import ColumnWeight, FKPWeight
from oneuniverse.measure import build_galaxy_clustering
from oneuniverse.measure.measurement_set import MeasurementSet
from test.fixtures.measure_ouf import synthetic_point_view


def test_build_galaxy_clustering_measurement_set(tmp_path):
    view = synthetic_point_view(tmp_path, n=5000, seed=8)
    ms = build_galaxy_clustering(
        view, tracer="gal", z_range=(0.3, 0.7),
        weights=[FKPWeight(nbar="nbar", P0=1e4), ColumnWeight("weight_comp")],
        nside_window=64, nside_region=4,
        nz_edges=np.linspace(0.0, 1.2, 25),
        randoms="generate", n_randoms=20000, seed=1,
    )
    assert isinstance(ms, MeasurementSet)
    ps = ms.products["gal"]
    assert ps.kind == "pointset"
    assert "weight" in ps.catalog.columns and (ps.catalog["weight"] > 0).all()
    assert ps.randoms is not None and len(ps.randoms) == 20000
    assert ps.nz.method == "spec_hist"
    assert ps.provenance.randoms_source == "generated"
    assert ms.spec.statistic == "pk_multipole"
    # invariants: region map present + shared length; cosmology-free
    ms.check_invariants()
    assert len(ps.region_map) == len(ps.catalog)


def test_invariants_reject_cosmology(tmp_path):
    view = synthetic_point_view(tmp_path, n=1000, seed=8)
    ms = build_galaxy_clustering(view, tracer="gal", z_range=(0.1, 1.0),
                                 weights=[ColumnWeight("weight_comp")],
                                 nside_window=32, nside_region=2,
                                 nz_edges=np.linspace(0, 1.2, 13),
                                 randoms="generate", n_randoms=2000, seed=1)
    import pytest
    with pytest.raises(ValueError, match="cosmology"):
        ms.metadata_with_cosmology_must_fail = True
        ms.check_invariants(_inject_cosmology=True)
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement `spec.py`**

```python
# oneuniverse/measure/spec.py
"""Step 9: the aimed-measurement declaration. Cosmology deferred to P2."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class MeasurementSpec:
    tracers: Tuple[str, ...]
    pairs: Tuple[Tuple[str, str], ...]
    statistic: str                    # "pk_multipole" | "xi_smu" | "w_theta" | ...
    estimator_family: str             # "clustering" | "field_level" | ...
    binning: Optional[dict] = None
    coords: str = "on_sky"            # comoving conversion happens in P2
    covariance: str = "jackknife"
```

- [ ] **Step 4: Implement `measurement_set.py`**

```python
# oneuniverse/measure/measurement_set.py
"""The joint-analysis bundle handed to Pillar 2. Cosmology-free."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from oneuniverse.measure.dataproduct import DataProduct
from oneuniverse.measure.metadata import ProductMetadata
from oneuniverse.measure.spec import MeasurementSpec


@dataclass
class MeasurementSet:
    products: Dict[str, DataProduct]
    spec: MeasurementSpec
    metadata: ProductMetadata

    def check_invariants(self, *, _inject_cosmology: bool = False) -> None:
        if _inject_cosmology or hasattr(self.metadata, "cosmology"):
            raise ValueError(
                "MeasurementSet must be cosmology-free (no cosmology in "
                "metadata); cosmology enters at the Pillar-2 estimator call")
        nside = self.metadata.nside_region
        for name, p in self.products.items():
            n = len(p.region_map)
            if hasattr(p, "catalog") and p.catalog is not None:
                if len(p.catalog) != n:
                    raise ValueError(
                        f"product {name!r}: region_map length {n} != catalog "
                        f"length {len(p.catalog)}")
            if p.metadata.nside_region != nside:
                raise ValueError(
                    f"product {name!r}: region NSIDE {p.metadata.nside_region}"
                    f" != set NSIDE {nside} (shared region_map invariant)")
```

- [ ] **Step 5: Implement `clustering.py`** (the 9-step connection)

```python
# oneuniverse/measure/clustering.py
"""build_galaxy_clustering — the galaxy-clustering P1->P2 connection (9 steps)."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from oneuniverse.combine.weights import Weight
from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.measure.dataproduct import PointSet
from oneuniverse.measure.measurement_set import MeasurementSet
from oneuniverse.measure.metadata import ProductMetadata, Provenance
from oneuniverse.measure.nz import nz_from_spec_z
from oneuniverse.measure.randoms import generate_randoms, randoms_from_view
from oneuniverse.measure.regions import assign_regions
from oneuniverse.measure.select import select_clean
from oneuniverse.measure.spec import MeasurementSpec
from oneuniverse.measure.weighting import assemble_weight
from oneuniverse.measure.window import footprint_from_positions


def build_galaxy_clustering(
    view: DatasetView, *, tracer: str = "gal",
    z_range: Tuple[float, float],
    weights: Sequence[Weight],
    nz_edges,
    nside_window: int = 256,
    nside_region: int = 8,
    quality_column: Optional[str] = "quality", quality_min: float = 1.0,
    randoms: Union[str, DatasetView] = "generate",
    n_randoms: int = 0, seed: int = 0,
    statistic: str = "pk_multipole",
) -> MeasurementSet:
    """OUF POINT view -> galaxy-clustering MeasurementSet (cosmology-free)."""
    # 1-2 select + clean
    cat = select_clean(view, z_range=z_range, quality_column=quality_column,
                       quality_min=quality_min)
    # 3 weights
    cat, recipe = assemble_weight(cat, weights)
    w = cat["weight"].to_numpy()
    # 5 window
    win = footprint_from_positions(cat["ra"].to_numpy(), cat["dec"].to_numpy(),
                                   nside=nside_window)
    # 6 n(z) (weighted)
    nz = nz_from_spec_z(cat["z"].to_numpy(), edges=nz_edges, weights=w)
    # 4 randoms (ingest | generate)
    if isinstance(randoms, DatasetView):
        rnd, source = randoms_from_view(randoms)
    elif randoms == "generate":
        rnd, source = generate_randoms(win, nz, n_randoms=n_randoms, seed=seed)
    else:
        rnd, source = None, None
    # 8 region map (shared scheme; applied to data + randoms)
    region = assign_regions(cat["ra"].to_numpy(), cat["dec"].to_numpy(),
                            nside=nside_region)
    if rnd is not None:
        rnd = rnd.copy()
        rnd["region_id"] = assign_regions(rnd["ra"].to_numpy(),
                                          rnd["dec"].to_numpy(),
                                          nside=nside_region)
    cat = cat.copy(); cat["region_id"] = region
    meta = ProductMetadata(frame="icrs", epoch=2000.0, length_unit="deg",
                           nside_region=int(nside_region))
    prov = Provenance(dataset_ids=(view.survey_name,), weight_recipe=recipe,
                      randoms_source=source, nz_method=nz.method)
    ps = PointSet(catalog=cat, randoms=rnd, nz=nz, window=win,
                  region_map=region, metadata=meta, provenance=prov)
    spec = MeasurementSpec(tracers=(tracer,), pairs=((tracer, tracer),),
                           statistic=statistic, estimator_family="clustering")
    return MeasurementSet(products={tracer: ps}, spec=spec, metadata=meta)
```

- [ ] **Step 6: Export in `__init__.py`**

```python
from oneuniverse.measure.clustering import build_galaxy_clustering
from oneuniverse.measure.measurement_set import MeasurementSet
from oneuniverse.measure.dataproduct import PointSet

__all__ = ["build_galaxy_clustering", "MeasurementSet", "PointSet"]
```

- [ ] **Step 7: Run — PASS.** `pytest test/test_measure_clustering.py -v`

- [ ] **Step 8: Commit** `measure/T8: MeasurementSpec + MeasurementSet + build_galaxy_clustering (9-step connection) + invariants`.

---

## Task 9: Visual diagnostic + close-out

**Files:** Create `test/test_visual_measure.py`, `scripts/build_measure_demo.py`; Modify `oneuniverse/measure/__init__.py` (already), `CLAUDE.md`, `plans/README.md`, memory.

- [ ] **Step 1:** `scripts/build_measure_demo.py` builds a MeasurementSet from `synthetic_point_view` and renders a 3-panel figure to `test/test_output/measure_galaxy_clustering.png`: (a) data vs randoms sky scatter inside the footprint, (b) n(z) of data (weighted) vs randoms, (c) region_map id map. (Visual-testing convention — diagnostic must look right.)

- [ ] **Step 2:** Visual test:

```python
# test/test_visual_measure.py
from pathlib import Path


def test_measure_figure_exists():
    p = Path(__file__).parent / "test_output" / "measure_galaxy_clustering.png"
    assert p.is_file() and p.stat().st_size > 5_000
```

- [ ] **Step 3:** Run the demo (generates the PNG), then `pytest test/test_measure_*.py test/test_visual_measure.py -q` — all green.

- [ ] **Step 4:** Full suite: `pytest -q 2>&1 | tail -3`.

- [ ] **Step 5:** Docs + memory:
  - `CLAUDE.md` — flesh out the `oneuniverse/measure/` bullet with the module list (select/weighting/window/nz/randoms/regions/spec/measurement_set/clustering) + `build_galaxy_clustering`.
  - `plans/README.md` — add a "Pillar 2 — measure" row: galaxy clustering connection done; next probes WL/PV/SN/Lyα.
  - Memory `project_pillar2_definition` — append "galaxy-clustering connection built in `oneuniverse.measure` (synthetic-OUF TDD); real DESI/eBOSS validation + next probes pending."

- [ ] **Step 6: Commit** `measure/T9: galaxy-clustering demo + diagnostic figure + close-out docs`.

---

## Success criteria
- `oneuniverse.measure.build_galaxy_clustering(view, ...)` turns a synthetic OUF POINT view into a `MeasurementSet` carrying: cleaned weighted catalog, randoms (**ingested or generated**), n(z) (method-tagged), HEALPix window, shared region_map, `MeasurementSpec`, provenance.
- `MeasurementSet.check_invariants()` enforces shared region NSIDE + **cosmology-free**.
- All 9 transform steps (§1 of the requirements research) implemented as composable functions.
- Diagnostic figure committed; full suite green; **no cosmology anywhere** in `measure/`.

## Maps to the requirements research
[`research/2026-06-05-p1-to-p2-measurement-requirements.md`](../research/2026-06-05-p1-to-p2-measurement-requirements.md) §1 nine-step transform → Tasks 2–8; §2 atoms A·PointSet/B(z_spec)/D/E(core)/G → Tasks 1,3,4,5,6,7; §6 build order step 1 (galaxy clustering). Sightline/FieldMap subtypes + photo-z kernel + shapes are **out of scope** (later probes).

## Self-review checklist
- [ ] `PointSet`/`Window`/`Nz`/`MeasurementSpec`/`MeasurementSet` signatures consistent across Tasks 1–8.
- [ ] `assemble_weight` returns `(df, recipe)`; `generate_randoms`/`randoms_from_view` return `(df, source)` — matched at call sites in `clustering.py`.
- [ ] `FKPWeight(nbar=, P0=)` verified against `oneuniverse/combine/weights/fkp.py` before Task 3 (adjust kwargs if needed).
- [ ] No cosmology imported anywhere in `measure/`.
- [ ] Synthetic-OUF only; no real-data dependency in tests.
