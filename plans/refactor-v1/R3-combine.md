# R3 — combine: weights, defaults, WeightedCatalog

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `oneuniverse.combine` as three modules: composable per-object weights on pyarrow tables, a small default-weight registry keyed by `(survey_kind, z_type)`, and a `WeightedCatalog` that hydrates once through `OneuidQuery` and combines per-object values across surveys with vectorised group operations.

**Architecture:** `Weight.__call__(table) -> float64[]` with declared `columns`; `*` composes. `WeightedCatalog` holds one long table `(oneuid, dataset, <columns>, weight)`; `combine` groups by `oneuid` with `np.unique(return_inverse)` + `np.add.at` / `np.minimum.at`.

**Tech Stack:** numpy, pyarrow, healpy.

**Spec:** design §5 (`combine/`), audit `audit_data_combine.md` §6 items 10–11, §7 "combine strategies". Master plan Global Constraints apply (no cosmology; `P0` and `nbar` are user-supplied numbers, documented as such).

## Interfaces consumed (R1, R2)

`DatasetView`, `OneuidIndex`, `OneuidQuery.hydrate_table`, `Database`, `_core.healpix.radec_to_pix`, `data.schema.z_types`.

## Interfaces produced

```
oneuniverse.combine.weights
  class Weight(ABC): columns: frozenset[str] (abstract property) ; __call__(self, table: pa.Table | pd.DataFrame) -> np.ndarray[float64]   # validates columns present (SchemaError), shape (n,), finite or raises ValidationError
                     __mul__(other: Weight) -> ProductWeight ; describe() -> str
  class ProductWeight(Weight): factors: tuple[Weight, ...]
  class ColumnWeight(Weight): column: str
  class ConstantWeight(Weight): value: float = 1.0
  class InverseVarianceWeight(Weight): err_column: str ; missing: Literal["unit", "zero", "raise"] = "unit"   # NaN or 0 error -> 1.0 / 0.0 / ValidationError
  class FKPWeight(Weight): nbar: Callable[[np.ndarray], np.ndarray] | str ; P0: float ; z_column: str = "z"    # w = 1 / (1 + nbar(z) P0); str = column holding nbar
  class HealpixMapWeight(Weight): map: np.ndarray ; nside: int ; nest: bool = True ; nan: Literal["raise"] | float = "raise" ; ra_column="ra" ; dec_column="dec"
                                  @classmethod from_fits(path, *, nest=True)
  class ShearWeight(Weight): kind: Literal["metacal", "lensfit"] ; shape_weight_column="shear_weight" ; sigma_e_columns: tuple[str, str] | None = None
                              # metacal: R_eff = (R11 + R22)/2 + R_S ; lensfit: R_eff = 1 + m_bias ; w = shape_weight / (R_eff² + σ_e²) with σ_e² = mean of the two sigma columns squared when given
  class QualityMaskWeight(Weight): column: str ; op: Literal["==", "!=", "<", "<=", ">", ">="] ; threshold: float
oneuniverse.combine.defaults
  default_weight_for(survey_kind: str, z_type: str) -> Weight ; register_default(survey_kind, z_type, factory: Callable[[], Weight], *, replace=False) ; defaults() -> dict[tuple[str, str], Callable]
  built-ins: ("spectroscopic","spec") -> InverseVarianceWeight("z_err") ; ("photometric","phot") -> InverseVarianceWeight("z_err") ; ("photometric","phot_pdf") -> InverseVarianceWeight("z_pdf_std") ; ("peculiar_velocity","pv") -> InverseVarianceWeight("v_pec_err")
oneuniverse.combine.catalog
  class WeightedCatalog
    @classmethod from_oneuid(index: OneuidIndex, db: Database, *, columns: Sequence[str] | None = None) -> WeightedCatalog   # hydrates once; columns default = all shared columns of the datasets
    .datasets -> list[str] ; .table -> pa.Table (long) ; .add_weight(dataset: str, weight: Weight) -> self ; .fill_defaults(db, *, skip_unknown=False) -> self   # from each dataset's manifest survey.kind + per-row z_type
    .weights() -> pa.Table (oneuid, dataset, weight)   # ValidationError if a dataset has no weight
    .combine(value: str, *, variance: str | None = None, strategy: Literal["ivar", "best", "mean"] = "ivar") -> pa.Table (oneuid, value, variance, n_sources, best_dataset)
oneuniverse.combine (__init__) exports: Weight, ProductWeight, ColumnWeight, ConstantWeight, InverseVarianceWeight, FKPWeight, HealpixMapWeight, ShearWeight, QualityMaskWeight, default_weight_for, register_default, WeightedCatalog
```

---

### Task 1: `combine.weights`

**Files:**
- Create: `oneuniverse/combine/__init__.py`, `oneuniverse/combine/weights.py`
- Test: `test/combine/__init__.py`, `test/combine/test_weights.py`

- [ ] **Step 1: Write the failing tests**

```python
T = pa.table({"ra": [0.0, 10.0, 20.0], "dec": [0.0, 0.0, 0.0], "z": [0.5, 1.0, 1.5], "z_err": [0.01, np.nan, 0.0], "q": [1, 0, 1], "nbar_col": [1e-4, 2e-4, 3e-4]})

def test_column_and_constant_and_product():
    w = ColumnWeight("q") * ConstantWeight(2.0); out = w(T); assert out.tolist() == [2.0, 0.0, 2.0] and w.columns == {"q"} and isinstance(w, ProductWeight)

def test_ivar_missing_policies():
    assert InverseVarianceWeight("z_err")(T).tolist() == [1e4, 1.0, 1.0]
    assert InverseVarianceWeight("z_err", missing="zero")(T).tolist() == [1e4, 0.0, 0.0]
    with pytest.raises(ValidationError): InverseVarianceWeight("z_err", missing="raise")(T)

def test_fkp_callable_and_column():
    assert np.allclose(FKPWeight(nbar=lambda z: 1e-4 * np.ones_like(z), P0=1e4)(T), 1 / (1 + 1.0))
    assert np.allclose(FKPWeight(nbar="nbar_col", P0=1e4)(T), 1 / (1 + np.array([1, 2, 3]) * 1.0))

def test_healpix_map_weight_nest_and_ring_and_nan():
    m = np.arange(12 * 4 * 4, dtype=float); w = HealpixMapWeight(m, nside=4)(T)
    assert np.array_equal(w, m[radec_to_pix(4, T["ra"].to_numpy(), T["dec"].to_numpy())])
    m_ring = m.copy(); assert np.array_equal(HealpixMapWeight(m_ring, nside=4, nest=False)(T), m[hp.ang2pix(4, T["ra"].to_numpy(), T["dec"].to_numpy(), lonlat=True)])
    m[radec_to_pix(4, [0.0], [0.0])[0]] = np.nan
    with pytest.raises(ValidationError): HealpixMapWeight(m, nside=4)(T)
    assert HealpixMapWeight(m, nside=4, nan=0.0)(T)[0] == 0.0

def test_shear_weight_metacal_and_lensfit():
    S = pa.table({"shear_weight": [1.0, 1.0], "R11": [0.8, 0.8], "R22": [0.8, 0.8], "R_S": [0.0, 0.1], "m_bias": [-0.1, 0.0], "e1_err": [0.3, 0.3], "e2_err": [0.3, 0.3]})
    assert np.allclose(ShearWeight("metacal")(S), [1 / 0.8**2, 1 / 0.9**2])
    assert np.allclose(ShearWeight("lensfit", sigma_e_columns=("e1_err", "e2_err"))(S), [1 / (0.9**2 + 0.09), 1 / (1.0 + 0.09)])

def test_quality_mask_and_missing_column():
    assert QualityMaskWeight("q", ">=", 1)(T).tolist() == [1.0, 0.0, 1.0]
    with pytest.raises(SchemaError, match="nope"): ColumnWeight("nope")(T)

def test_accepts_dataframe(): assert ColumnWeight("q")(T.to_pandas()).tolist() == [1.0, 0.0, 1.0]
```

- [ ] **Step 2: Run, expect failure**
- [ ] **Step 3: Implement** (frozen dataclasses; `Weight.__call__` converts a DataFrame with `pa.Table.from_pandas`, checks `self.columns <= set(table.column_names)`, calls `self.compute(table) -> ndarray`, validates shape and `np.isfinite`).
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** — `feat(combine): weight primitives on arrow tables`

---

### Task 2: `combine.defaults`

**Files:**
- Create: `oneuniverse/combine/defaults.py`
- Test: `test/combine/test_defaults.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_builtin_keys_use_registered_z_types():
    for (_, zt) in defaults(): assert zt in z_types()
    assert isinstance(default_weight_for("peculiar_velocity", "pv"), InverseVarianceWeight) and default_weight_for("peculiar_velocity", "pv").err_column == "v_pec_err"
    with pytest.raises(NotFoundError): default_weight_for("spectroscopic", "pv")

def test_register_default_replace_policy():
    with pytest.raises(ValueError): register_default("spectroscopic", "spec", lambda: ConstantWeight())
    register_default("spectroscopic", "spec", lambda: ConstantWeight(3.0), replace=True); assert default_weight_for("spectroscopic", "spec")(pa.table({"x": [1]}))[0] == 3.0
    register_default("spectroscopic", "spec", lambda: InverseVarianceWeight("z_err"), replace=True)  # restore
```

- [ ] **Step 2–5:** Run (fail) → implement (module-level dict, factories called on each `default_weight_for`) → run (pass) → commit `feat(combine): default weights by survey kind and z_type`.

---

### Task 3: `combine.catalog`

**Files:**
- Create: `oneuniverse/combine/catalog.py`
- Test: `test/combine/test_catalog.py` (reuses the two-survey fixture builder from `test/data/test_oneuid.py` — move it to `test/fixtures/two_surveys.py` as `make_two_survey_db(tmp_path, n_each=30_000, n_common=10_000, seed=0) -> Database`)

- [ ] **Step 1: Write the failing tests**

```python
@pytest.fixture
def wc(tmp_path):
    db = make_two_survey_db(tmp_path); idx = db.build_oneuid(["a", "b"], rules=CrossMatchRules(), name="ab")
    return WeightedCatalog.from_oneuid(idx, db, columns=["z", "z_err", "mag"]), db

def test_long_table_shape(wc):
    cat, db = wc; assert cat.datasets == ["a", "b"] and cat.table.num_rows == 60_000 and set(cat.table.column_names) >= {"oneuid", "dataset", "z", "z_err", "mag"}

def test_weights_require_every_dataset(wc):
    cat, db = wc
    with pytest.raises(ValidationError, match="b"): cat.add_weight("a", ConstantWeight()).weights()
    cat.fill_defaults(db); w = cat.weights(); assert w.num_rows == 60_000 and (w["weight"].to_numpy() > 0).all()

def test_combine_ivar_matches_formula(wc):
    cat, db = wc; cat.add_weight("a", InverseVarianceWeight("z_err")).add_weight("b", InverseVarianceWeight("z_err"))
    out = cat.combine("z", variance="z_err"); multi = out.filter(pc.field("n_sources") == 2)
    assert out.num_rows == 50_000 and multi.num_rows == 10_000
    row = multi.slice(0, 1).to_pylist()[0]; members = cat.table.filter(pc.field("oneuid") == row["oneuid"]).to_pandas()
    ivar = 1 / members["z_err"].to_numpy() ** 2; assert row["value"] == pytest.approx((members["z"] * ivar).sum() / ivar.sum()) and row["variance"] == pytest.approx(1 / ivar.sum())

def test_combine_best_and_mean(wc):
    cat, db = wc; cat.fill_defaults(db)
    best = cat.combine("z", variance="z_err", strategy="best"); assert set(best["best_dataset"].to_pylist()) <= {"a", "b"}
    mean = cat.combine("mag", strategy="mean"); assert mean.num_rows == 50_000 and "variance" in mean.column_names

def test_combine_is_vectorised_fast(wc):
    cat, db = wc; cat.fill_defaults(db); t = time.perf_counter(); cat.combine("z", variance="z_err"); assert time.perf_counter() - t < 1.0
```

- [ ] **Step 2: Run, expect failure**
- [ ] **Step 3: Implement.** `from_oneuid`: `OneuidQuery(index, db).hydrate_table(all oneuids, columns=columns)`; `fill_defaults`: for each dataset, `kind = db.manifest(ds).survey.kind`; z_type per row from the index; group rows by `(kind, z_type)` → `default_weight_for`; `skip_unknown` → `ConstantWeight(1)` for unknown pairs; `combine`: `uid, inv = np.unique(oneuid, return_inverse=True)`; `ivar`: `wsum = add.at(w*ivar)`, `value = add.at(w*ivar*v)/wsum`, `variance = 1/wsum`; `best`: `np.minimum.at` on variance then match; `mean`: weighted mean + weighted sample variance.
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** — `feat(combine): WeightedCatalog with vectorised combination`

---

### Task 4: Close-out

- [ ] **Step 1:** `pytest -n 6` and single-process (< 60 s); guards green (no cosmology identifiers in `combine/`).
- [ ] **Step 2:** STATUS.md + memory; commit + push claude_config. Commit — `docs: R3 close-out`.

**Exit criterion:** all weights and `WeightedCatalog` tested on R2 output; `combine()` on 60 k rows under 1 s.
