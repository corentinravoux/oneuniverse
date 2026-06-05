# `oneuniverse.measure` — Weak Lensing (Cosmic Shear + GGL + 3×2pt) Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`).

**Goal:** Add the **weak-lensing** probe to `oneuniverse.measure`: source shape catalogs (`e1,e2` + metacal/lensfit calibration), the **photo-z kernel** (`qp`), tomographic n(z), and the `build_cosmic_shear` / `build_3x2pt` connections — emitting cosmology-free `MeasurementSet`s.

**Architecture:** Reuse the galaxy-clustering spine (`select_clean`, `assemble_weight`, `footprint_from_positions`, `assign_regions`, `MeasurementSet`). New surface: shape atoms on `PointSet`, a `photoz` field carrying P1's `ProbabilisticRedshift`, per-bin tomographic n(z), and a multi-product `MeasurementSet` (lens × source) for 3×2pt. **Cosmology-free** — shear calibration and z→r stay observational / P2.

**Tech Stack:** numpy, pandas, healpy; reuses `oneuniverse.combine.weights.ShearWeight`, `oneuniverse.data.dataset_view.DatasetView.load_pdf`, `oneuniverse.data.pdf.ProbabilisticRedshift`, `oneuniverse.data.tomographic_nz.TomographicNzSpec`. No cosmology engine.

---

## Reused (already built — do not rebuild)
`oneuniverse.measure`: `select_clean`, `assemble_weight`, `footprint_from_positions`, `Window`, `nz_from_spec_z`, `Nz`, `assign_regions`, `MeasurementSet`, `MeasurementSpec`, `PointSet`, `ProductMetadata`, `Provenance`, the synthetic-OUF fixture pattern (`test/fixtures/measure_ouf.py`).

## Confirmed P1 APIs
- Shear columns (OUF schema, all optional): `e1, e2, e1_err, e2_err, R11, R22, R_S, m_bias, c1_bias, c2_bias, shear_weight`.
- `ShearWeight(kind="metacal"|"lensfit", shape_weight_col="shear_weight", R11_col="R11", R22_col="R22", R_S_col="R_S", m_col="m_bias")` — `Weight.__call__(df)->ndarray`.
- `DatasetView.load_pdf() -> ProbabilisticRedshift` (manifest needs `pdf_spec`); `ProbabilisticRedshift.mean()/std()/cdf()/sample(n_per, seed)`.
- `TomographicNzSpec(bin_assignment_column="tomo_bin", ...)` — per-row tomo bin.

## File structure (new / modified)
| File | Responsibility |
|---|---|
| Modify `measure/dataproduct.py` | add `photoz` (+ `tomo_bin`) optional fields to `PointSet` |
| Modify `measure/spec.py` | add `pair_statistics: Optional[dict]` for mixed 3×2pt terms |
| Create `measure/shapes.py` | `attach_shear(cat, *, kind)` — validate shape cols + shear weight + recipe |
| Create `measure/photoz.py` | `attach_photoz(view)` -> `ProbabilisticRedshift`; `PhotozKernel` thin wrapper |
| Create `measure/tomography.py` | `tomographic_nz(cat, kernel, *, bin_column, z_grid)` -> dict[bin]→`Nz` |
| Create `measure/lensing.py` | `build_cosmic_shear(...)`, `build_3x2pt(...)` |
| Modify `test/fixtures/measure_ouf.py` | `synthetic_shear_view(...)` (shapes + pdf_spec) |
| Tests | one per task under `test/` |

---

## Task 1: Shear-source atoms on PointSet
**Files:** Modify `measure/dataproduct.py`; Create `measure/shapes.py`; Test `test/test_measure_shapes.py`.

- [ ] **Step 1: Extend the synthetic fixture** — add `synthetic_shear_view(tmp, n, seed, kind="metacal")` to `test/fixtures/measure_ouf.py` writing CORE + `e1,e2,e1_err,e2_err,R11,R22,R_S,shear_weight` columns (random small ellipticities, `R11=R22≈0.7`, `R_S≈0`, `shear_weight∈[0.5,1]`), via `write_ouf_dataset(..., survey_type="photometric")`.

- [ ] **Step 2: Failing test**

```python
# test/test_measure_shapes.py
import sys
from pathlib import Path

import numpy as np

from oneuniverse.measure.shapes import attach_shear

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_shear_view  # noqa: E402


def test_attach_shear_validates_and_weights(tmp_path):
    view = synthetic_shear_view(tmp_path, n=2000, seed=1, kind="metacal")
    cat = view.read()
    out, recipe = attach_shear(cat, kind="metacal")
    assert {"e1", "e2", "shear_weight"} <= set(out.columns)
    assert "weight" in out.columns and (out["weight"] >= 0).all()
    assert "metacal" in recipe.lower()
    # missing-column path raises clearly
    import pytest
    with pytest.raises(ValueError, match="shape column"):
        attach_shear(cat.drop(columns=["e1"]), kind="metacal")
```

- [ ] **Step 3: Implement `measure/shapes.py`**

```python
"""Shear-source atoms: validate shape columns + assemble the shear weight."""
from __future__ import annotations

from typing import Tuple

import pandas as pd

from oneuniverse.combine.weights import ShearWeight

_REQUIRED = ("e1", "e2", "shear_weight")


def attach_shear(catalog: pd.DataFrame, *, kind: str = "metacal",
                 out_column: str = "weight") -> Tuple[pd.DataFrame, str]:
    """Validate shape columns and set `out_column` = ShearWeight(kind)."""
    missing = [c for c in _REQUIRED if c not in catalog.columns]
    if missing:
        raise ValueError(f"attach_shear: missing shape column(s) {missing}")
    out = catalog.copy()
    w = ShearWeight(kind=kind)
    out[out_column] = w(out)
    return out, repr(w)
```

- [ ] **Step 4: Add `PointSet.photoz` + `PointSet.tomo_bin`** in `measure/dataproduct.py` (both `= None`, kw_only). No behaviour change for clustering.

- [ ] **Step 5: Run — PASS.** **Step 6: Commit** `measure/wl-T1: shear-source shape atoms (attach_shear) + PointSet.photoz/tomo_bin`.

---

## Task 2: Photo-z kernel attach
**Files:** Create `measure/photoz.py`; extend fixture with `pdf_spec`; Test `test/test_measure_photoz.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_measure_photoz.py
import sys
from pathlib import Path

import numpy as np

from oneuniverse.measure.photoz import attach_photoz

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_photoz_view  # noqa: E402


def test_attach_photoz_returns_kernel(tmp_path):
    view = synthetic_photoz_view(tmp_path, n=1500, seed=2)   # writes pdf_spec
    kernel = attach_photoz(view)
    assert kernel.mean().shape[0] == view.n_rows
    assert np.all(kernel.std() > 0)
```

- [ ] **Step 2:** Add `synthetic_photoz_view` to the fixture: build a gridded p(z) catalog (reuse `test/fixtures/pdf_catalog.py:make_gaussian_pdf_catalog` if present, else per-row gaussian on a z-grid) + `write_ouf_dataset(..., pdf_spec=PdfSpec(...))` so `load_pdf()` works.

- [ ] **Step 3: Run — FAIL. Step 4: Implement `measure/photoz.py`**

```python
"""Photo-z kernel: P1's per-object p(z) attached as the measure atom."""
from __future__ import annotations

from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.data.pdf import ProbabilisticRedshift


def attach_photoz(view: DatasetView) -> ProbabilisticRedshift:
    """Return the per-object photo-z kernel (qp) from an OUF PDF dataset."""
    return view.load_pdf()
```

- [ ] **Step 5: Run — PASS. Step 6: Commit** `measure/wl-T2: attach_photoz — per-object p(z) kernel from OUF PdfSpec`.

---

## Task 3: Tomographic n(z)
**Files:** Create `measure/tomography.py`; Test `test/test_measure_tomography.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_measure_tomography.py
import numpy as np

from oneuniverse.measure.nz import Nz
from oneuniverse.measure.tomography import tomographic_nz


def test_tomographic_nz_stacks_per_bin():
    n = 600
    z_grid = np.linspace(0.0, 2.0, 41)
    # two bins: bin 0 peaks at 0.4, bin 1 at 1.0
    means = np.where(np.arange(n) < n // 2, 0.4, 1.0)
    import pandas as pd
    cat = pd.DataFrame({"tomo_bin": (np.arange(n) >= n // 2).astype(int),
                        "z": means})

    class _K:                       # minimal kernel stand-in: point masses
        def sample(self, n_per, seed=None):
            return np.repeat(means[:, None], n_per, axis=1)

    nzs = tomographic_nz(cat, _K(), bin_column="tomo_bin", z_grid=z_grid)
    assert set(nzs) == {0, 1}
    assert all(isinstance(v, Nz) for v in nzs.values())
    c0, c1 = nzs[0].centers(), nzs[1].centers()
    assert c0[np.argmax(nzs[0].counts)] < 0.7      # bin 0 low-z
    assert c1[np.argmax(nzs[1].counts)] > 0.7      # bin 1 high-z
```

- [ ] **Step 2: Run — FAIL. Step 3: Implement `measure/tomography.py`**

```python
"""Per-tomographic-bin n(z): stack the photo-z kernel within each bin."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from oneuniverse.measure.nz import Nz


def tomographic_nz(catalog: pd.DataFrame, kernel, *, bin_column: str,
                   z_grid: np.ndarray, n_per: int = 10, seed: int = 0
                   ) -> Dict[int, Nz]:
    """Stack kernel samples per bin into an Nz (method='photo_stack')."""
    draws = kernel.sample(n_per, seed=seed)         # (N, n_per)
    out: Dict[int, Nz] = {}
    bins = catalog[bin_column].to_numpy()
    for b in np.unique(bins):
        z = draws[bins == b].ravel()
        counts, _ = np.histogram(z, bins=z_grid)
        out[int(b)] = Nz(edges=np.asarray(z_grid, float),
                         counts=counts.astype(float), method="photo_stack")
    return out
```

- [ ] **Step 4: Run — PASS. Step 5: Commit** `measure/wl-T3: tomographic_nz — per-bin photo-z stack`.

---

## Task 4: `build_cosmic_shear`
**Files:** Modify `measure/spec.py` (add `pair_statistics`); Create `measure/lensing.py`; Test `test/test_measure_cosmic_shear.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_measure_cosmic_shear.py
import sys
from pathlib import Path

import numpy as np

from oneuniverse.measure import MeasurementSet
from oneuniverse.measure.lensing import build_cosmic_shear

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_shear_view  # noqa: E402


def test_build_cosmic_shear(tmp_path):
    view = synthetic_shear_view(tmp_path, n=4000, seed=3, kind="metacal",
                                with_pdf=True, n_tomo=2)
    ms = build_cosmic_shear(view, tracer="src", kind="metacal",
                            tomo_column="tomo_bin",
                            z_grid=np.linspace(0, 2, 41), nside_region=4)
    assert isinstance(ms, MeasurementSet)
    ps = ms.products["src"]
    assert {"e1", "e2", "weight"} <= set(ps.catalog.columns)
    assert ps.photoz is not None
    assert isinstance(ps.nz, dict) and set(ps.nz) == {0, 1}   # per-bin n(z)
    assert ms.spec.statistic == "xi_pm"
    assert ms.spec.estimator_family == "lensing"
    ms.check_invariants()
```

- [ ] **Step 2: Run — FAIL. Step 3: Add `pair_statistics`** to `MeasurementSpec` (`Optional[dict] = None`).

- [ ] **Step 4: Implement `build_cosmic_shear` in `measure/lensing.py`**

```python
"""Weak-lensing connections: cosmic shear + 3x2pt. Cosmology-free."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.measure.dataproduct import PointSet
from oneuniverse.measure.measurement_set import MeasurementSet
from oneuniverse.measure.metadata import ProductMetadata, Provenance
from oneuniverse.measure.photoz import attach_photoz
from oneuniverse.measure.regions import assign_regions
from oneuniverse.measure.select import select_clean
from oneuniverse.measure.shapes import attach_shear
from oneuniverse.measure.spec import MeasurementSpec
from oneuniverse.measure.tomography import tomographic_nz
from oneuniverse.measure.window import footprint_from_positions


def build_cosmic_shear(view: DatasetView, *, tracer: str = "src",
                       kind: str = "metacal", tomo_column: str = "tomo_bin",
                       z_grid, nside_window: int = 256, nside_region: int = 8,
                       statistic: str = "xi_pm") -> MeasurementSet:
    cat = select_clean(view)                                  # 1-2
    cat, srecipe = attach_shear(cat, kind=kind)               # 3 (shear weight)
    kernel = attach_photoz(view)                              # 7 photo-z kernel
    nzs = tomographic_nz(cat, kernel, bin_column=tomo_column, # 6 per-bin n(z)
                         z_grid=z_grid)
    win = footprint_from_positions(cat["ra"].to_numpy(),      # 5
                                   cat["dec"].to_numpy(), nside=nside_window)
    region = assign_regions(cat["ra"].to_numpy(),             # 8
                            cat["dec"].to_numpy(), nside=nside_region)
    cat = cat.copy(); cat["region_id"] = region
    meta = ProductMetadata(frame="icrs", epoch=2000.0, length_unit="deg",
                           nside_region=int(nside_region))
    prov = Provenance(dataset_ids=(view.survey_name,),
                      weight_recipe=(srecipe,), nz_method="photo_stack")
    ps = PointSet(catalog=cat, randoms=None, nz=nzs, window=win,
                  region_map=region, metadata=meta, provenance=prov,
                  photoz=kernel, tomo_bin=cat[tomo_column].to_numpy())
    spec = MeasurementSpec(tracers=(tracer,), pairs=((tracer, tracer),),
                           statistic=statistic, estimator_family="lensing")
    return MeasurementSet(products={tracer: ps}, spec=spec, metadata=meta)
```

(The `check_invariants` catalog-length check already tolerates `nz` being a dict — it only inspects `region_map`/`catalog`. No change needed.)

- [ ] **Step 5: Run — PASS. Step 6: Commit** `measure/wl-T4: build_cosmic_shear (shapes + photo-z kernel + tomographic n(z))`.

---

## Task 5: `build_3x2pt` (multi-product, shared region)
**Files:** Modify `measure/lensing.py`; Test `test/test_measure_3x2pt.py`.

- [ ] **Step 1: Failing test**

```python
# test/test_measure_3x2pt.py
import sys
from pathlib import Path

import numpy as np

from oneuniverse.measure.lensing import build_3x2pt

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.measure_ouf import synthetic_point_view, synthetic_shear_view  # noqa: E402


def test_build_3x2pt_shares_region_and_pairs(tmp_path):
    lens = synthetic_point_view(tmp_path, n=4000, seed=4, name="lens")
    src = synthetic_shear_view(tmp_path, n=4000, seed=5, kind="metacal",
                               with_pdf=True, n_tomo=2, name="src")
    ms = build_3x2pt(lens, src, z_grid=np.linspace(0, 2, 41), nside_region=4,
                     lens_z_range=(0.2, 0.6),
                     lens_weights_columns=("weight_comp",))
    assert set(ms.products) == {"lens", "src"}
    assert ms.metadata.nside_region == 4
    assert ("lens", "src") in ms.spec.pairs            # the GGL cross term
    assert ms.spec.pair_statistics[("lens", "src")] == "gamma_t"
    assert ms.spec.pair_statistics[("src", "src")] == "xi_pm"
    ms.check_invariants()                              # both share NSIDE
```

- [ ] **Step 2: Run — FAIL. Step 3: Implement `build_3x2pt`**

```python
def build_3x2pt(lens_view, source_view, *, z_grid, nside_region: int = 8,
                nside_window: int = 256, lens_z_range=(0.0, 2.0),
                lens_weights_columns: Tuple[str, ...] = ("weight_comp",),
                kind: str = "metacal", tomo_column: str = "tomo_bin"
                ) -> MeasurementSet:
    from oneuniverse.combine.weights import ColumnWeight
    from oneuniverse.measure.weighting import assemble_weight
    # lens PointSet (clustering)
    lcat = select_clean(lens_view, z_range=lens_z_range)
    lcat, lrec = assemble_weight(
        lcat, [ColumnWeight(c) for c in lens_weights_columns])
    lreg = assign_regions(lcat["ra"].to_numpy(), lcat["dec"].to_numpy(),
                          nside=nside_region)
    lcat = lcat.copy(); lcat["region_id"] = lreg
    meta = ProductMetadata(frame="icrs", epoch=2000.0, length_unit="deg",
                           nside_region=int(nside_region))
    lens_ps = PointSet(
        catalog=lcat, randoms=None,
        nz=None, window=footprint_from_positions(
            lcat["ra"].to_numpy(), lcat["dec"].to_numpy(), nside=nside_window),
        region_map=lreg, metadata=meta,
        provenance=Provenance(dataset_ids=(lens_view.survey_name,),
                              weight_recipe=lrec))
    # source PointSet (shear) via build_cosmic_shear, then re-region to shared
    src_ms = build_cosmic_shear(source_view, tracer="src", kind=kind,
                                tomo_column=tomo_column, z_grid=z_grid,
                                nside_window=nside_window,
                                nside_region=nside_region)
    src_ps = src_ms.products["src"]
    spec = MeasurementSpec(
        tracers=("lens", "src"),
        pairs=(("lens", "lens"), ("lens", "src"), ("src", "src")),
        statistic="mixed", estimator_family="lensing",
        pair_statistics={("lens", "lens"): "w_theta",
                         ("lens", "src"): "gamma_t",
                         ("src", "src"): "xi_pm"})
    return MeasurementSet(products={"lens": lens_ps, "src": src_ps},
                          spec=spec, metadata=meta)
```

- [ ] **Step 4: Run — PASS. Step 5: Commit** `measure/wl-T5: build_3x2pt — lens×source 3x2pt bundle, shared region, per-pair statistics`.

---

## Task 6: Visual + close-out
- [ ] `scripts/build_measure_wl_demo.py`: cosmic-shear MeasurementSet → 3-panel figure (ellipticity whisker map; per-bin tomographic n(z); region map) → `test/test_output/measure_weak_lensing.png`. Visual test asserts it exists.
- [ ] Full suite green. Update `CLAUDE.md` measure bullet (WL added), `plans/README.md` (WL row), memory `project_pillar2_definition` (WL connection built).
- [ ] Commit `measure/wl-T6: cosmic-shear demo + docs`.

## Success criteria
- `build_cosmic_shear` emits a source `PointSet` with shapes + shear weight + photo-z kernel + per-bin tomographic n(z); cosmology-free.
- `build_3x2pt` emits a 2-product MeasurementSet sharing one region_map with per-pair statistics (`w_theta`, `gamma_t`, `xi_pm`).
- Reuses the spine (no duplicated select/window/region code). Full suite green.

## Maps to requirements research
[`research/2026-06-05-p1-to-p2-measurement-requirements.md`](../research/2026-06-05-p1-to-p2-measurement-requirements.md) §2 atoms C(shapes+calib), B(photo-z kernel, tomographic n(z)); §3 cosmic shear / GGL / 3×2pt rows.

## Self-review
- [ ] `PointSet.photoz`/`tomo_bin` added in T1, used in T4. `MeasurementSpec.pair_statistics` added in T4, used in T5.
- [ ] `attach_shear` returns `(df, recipe)`; `tomographic_nz` returns `dict[int, Nz]`; matched at call sites.
- [ ] `ShearWeight(kind=)` + `load_pdf()` signatures verified against P1 before Steps.
- [ ] No cosmology in `measure/lensing.py`.
