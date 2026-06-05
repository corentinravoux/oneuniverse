# Pillar 2 — Canonical Definition (Estimators, Likelihoods, the DataProduct boundary)

**Date:** 2026-06-05.
**Status:** Definition / architecture doc. **No code yet** — this defines the
scope, the P1→P2 contract, the estimator taxonomy, and what is built vs not,
so implementation phases can be planned against a fixed target.
**Supersedes** the broad-strokes [`2026-05-28-pillar2-external-interfaces.md`](2026-05-28-pillar2-external-interfaces.md)
(kept for history; its galaxy-catalog-only `MeasurementSet` is generalised
here to the **Universal DataProduct**).

> **Two decisions locked (2026-06-05, owner):**
> 1. The P1→P2 input contract is a **Universal DataProduct** — one abstraction
>    covering point catalogs, Lyα sightlines, and HEALPix/voxel maps, consumed
>    by *every* estimator family. Not a galaxy-clustering-only object.
> 2. flip is **one estimator family of several**, not "Pillar 2".

---

## 1. Mission

Turn Pillar-1 data products into **cosmological constraints**: P(k), ξ(r),
C_ℓ, multi-tracer cross-spectra, field-level f×σ₈, Lyα P_1D / P_3D, SN Hubble
diagrams, forecasts, and the likelihoods/posteriors built on them.

**Pillar 2 is where cosmology enters.** H₀, Ωₘ, distance models, theory
templates, fiducial baselines — all chosen *at the estimator/likelihood call
site*, never in the data. The same DataProduct is re-analysable under a
different fiducial without re-ingesting.

## 2. Boundary (what is and is not Pillar 2)

| In scope (Pillar 2) | Out of scope |
|---|---|
| Read OUF parquet + weights + ONEUID/sub-object sidecars from disk | Ingest raw FITS / write OUF (Pillar 1) |
| Build randoms / windows / jackknife regions / n(z) | Cross-match surveys, apply selection weights (Pillar 1) |
| Construct the **DataProduct** + **MeasurementSet** contracts | Standardise survey columns (Pillar 1) |
| Compute estimators (P(k), ξ, C_ℓ, P_1D, P_3D, SN μ(z), cross) | Forward-model / simulate the field (Pillar 3) |
| Pick fiducial cosmology for z→r comoving conversion | Run N-body / hydro / IC sampling (Pillar 3) |
| Theory evaluation (cosmoprimo / CLASS / CAMB / pyCCL) | |
| Likelihoods + samplers (iminuit / emcee / cobaya / dynesty) | |
| Multi-tracer optimal weighting + joint covariance | |

The **input boundary** is the DataProduct (§4). The **output boundary** is a
result + covariance (and optionally a posterior), handed back to the user or
to Pillar 3 as a constraint.

## 3. Architectural shape

```
        ┌──────────────┐
        │ oneuniverse  │  Pillar 1
        │  data/ +     │  → OUF parquet (POINT / SIGHTLINE / CUBE / GW_SKYMAP)
        │  combine/    │     + weights + ONEUID + sub-object sidecars
        └──────┬───────┘
               │ disk artefacts
        ┌──────▼────────────────────────────────┐
        │  onemeasure  (NEW, Pillar-2 boundary)  │
        │  reads OUF → builds randoms / windows /│
        │  jackknife regions / n(z) → emits      │
        │  DataProduct + MeasurementSet + adapters│
        └──────┬─────────────────────────────────┘
               │ DataProduct / MeasurementSet  (cosmology-free)
   ┌───────────┼─────────────┬─────────────┬───────────────┐
   ▼           ▼             ▼             ▼               ▼
┌──────┐ ┌──────────┐ ┌────────────┐ ┌──────────┐ ┌──────────────┐
│ flip │ │ pycorr / │ │ p1desi /   │ │ lelantos │ │ onecorr (NEW)│
│ fxσ8 │ │ nbodykit │ │ lyapower   │ │ + lyavoid│ │ cross /      │
│      │ │ 2pt      │ │ Lyα P1D/3D │ │ Lyα×void │ │ multi-tracer │
└──────┘ └──────────┘ └────────────┘ └──────────┘ └──────────────┘
   └───────────┴─────────────┴─────────────┴───────────────┘
                              ▼
                  Result + covariance (+ posterior)
   cosmology enters HERE, per call: estimator(dp, cosmology=cosmo)
```

---

## 4. The Universal DataProduct — the P1→P2 contract

A **DataProduct** is the analysis-ready, **cosmology-free** object Pillar 2
consumes. It is one of three subtypes, each the augmented (randoms / window /
region / n(z)) form of a Pillar-1 OUF *geometry*:

| DataProduct subtype | from OUF geometry | carries | example tracers |
|---|---|---|---|
| **PointSet** | `POINT` | `ra, dec, z, weight, region_id` + `randoms` + `nz` + `window`; optional `velocity` / `dist_mod` / `mag` columns | galaxies, QSOs, peculiar-velocity, SNe, clusters |
| **Sightline** | `SIGHTLINE` | per-LOS `delta(λ|z)` + `mask` + `continuum` + LOS `ra, dec` + `region_id`; `wavelength_convention`, rest-frame state | Lyα forest, metal absorption |
| **FieldMap** | `CUBE` / `GW_SKYMAP` | pixel/voxel field values + `mask` + axis/NSIDE/ordering + `region_id` | CMB κ, tSZ y, depth/systematics maps, IFU cubes, HI |

### 4.1 Common invariants (all subtypes)

1. **No cosmology.** `metadata` carries `frame` (icrs/galactic/ecliptic),
   `epoch`, `unit`, `wavelength_convention` — observational only. z→r is the
   estimator's job.
2. **Shared `region_map`** — a HEALPix array (NSIDE in metadata) assigning every
   element to a jackknife/bootstrap region. The one field that makes *any* pair
   of DataProducts jointly resamplable (the basis of cross-covariance).
3. **Provenance** back to Pillar 1: ONEUID / dataset id + the weight recipe used.
4. **Subtype tag** + a uniform `.kind` so an estimator can declare what it
   accepts and reject the rest with a clear error (never silently mis-handle).

### 4.2 Sketch (defines shape, not an implementation)

```python
class DataProduct(ABC):
    kind: Literal["pointset", "sightline", "fieldmap"]
    region_map: np.ndarray          # HEALPix region_id per element
    metadata: ProductMetadata       # frame, epoch, unit, NSIDE — NO cosmology
    provenance: Provenance          # ONEUID / dataset id + weight recipe

class PointSet(DataProduct):        # kind="pointset"
    catalog: pa.Table               # ra, dec, z, weight, region_id [, velocity, dist_mod, mag]
    randoms: pa.Table               # same window + nz, compatible region_id
    nz: Nz                          # pdf(z) per-bin or per-row
    window: Window                  # angular/radial selection

class Sightline(DataProduct):       # kind="sightline"
    los: pa.Table                   # los_id, ra, dec, region_id, z_qso
    delta: VarLenArray              # per-LOS δ(λ|z)  (OUF list<f4>)
    mask: VarLenArray               # per-pixel mask / weight
    continuum: VarLenArray          # per-pixel fitted continuum

class FieldMap(DataProduct):        # kind="fieldmap"
    values: np.ndarray              # HEALPix vector or voxel grid
    mask: np.ndarray
    axes: AxisSpec                  # NSIDE+ordering, or WCS/axis units for cubes
```

### 4.3 MeasurementSet = a joint-analysis bundle of DataProducts

`MeasurementSet` is **not** a fourth data type — it is *what an estimator
consumes for a joint analysis*: one or more DataProducts that **share a
`region_map` and `metadata.frame`**, plus the bookkeeping for which pairs get
correlated and how covariance blocks assemble.

```python
class MeasurementSet:
    products: dict[str, DataProduct]       # named tracers (all share region_map)
    pairs: list[tuple[str, str]]           # which auto/cross terms to compute
    # cross-covariance assembled from the shared jackknife regions
```

Single-tracer is the one-product case. This subsumes the old
`MultiTracerMeasurementSet` (it is just `len(products) > 1`).

---

## 5. Estimator taxonomy (flip is one row)

Each family declares which DataProduct subtype(s) it consumes, what it
computes, who owns it, and where cosmology enters.

| Family | consumes | computes | owner package | status | cosmology |
|---|---|---|---|---|---|
| **Galaxy clustering 2-pt** | PointSet×PointSet | ξ(r), P(k), C_ℓ | `pycorr`/`nbodykit` (ext) + `onecorr` window/cov | adapter **not built** | z→r at call |
| **Field-level f×σ₈** | PointSet (PV + density) | velocity+density covariance → f×σ₈ | **`flip`** | **built**; DataProduct ingest **not** | fiducial in CovMatrix call |
| **Lyα P_1D** | Sightline | 1D flux power | **`p1desi`** | **built** (picca-native; adopt DP) | z→k at call |
| **Lyα P_3D** | Sightline / model | 3D flux power, Arinyo fit | **`lyapower`** | **built** (Nyx/gimlet-native) | model side |
| **Lyα tomography + voids** | Sightline | 3D δ map → void catalog → Lyα×void | **`lelantos`** + **`lyavoid`** | **built** (native) | z→r in map build |
| **SN Hubble diagram** | PointSet (μ, z) | distance–redshift, H₀/Ωₘ | `desidescsn` / Pantheon+ loader | **scripts only** | the fit itself |
| **Cross / multi-tracer** | any × any | galaxy×PV, galaxy×κ (PointSet×FieldMap), galaxy×Lyα (PointSet×Sightline), multi-tracer FKP, joint covariance | **`onecorr`** (NEW) | **not built** | per estimator |
| **Theory + likelihood** | results + cov | templates, posteriors, joint multi-probe | cosmoprimo/CCL/CLASS; flip Fit{Minuit,MCMC,Fisher}; future `cobaya` | partial (flip only) | owns the baseline |

**The point of the table:** "Pillar 2" is the *union* of these rows. flip is
the field-level row. The unifying work is (a) the DataProduct boundary so they
share one input contract, and (b) `onecorr` + a joint-likelihood layer so
results combine.

---

## 6. The adapter layer (`onemeasure.adapters`)

Thin shims: **DataProduct → native tool input**. Auditable in one place; each
opt-in behind its own optional dependency. **Adapters never compute science —
only re-shape.**

```python
adapters/flip.py      to_flip_data_vector(ps: PointSet, *, kind="velocity") -> flip.DataVector
adapters/pycorr.py    to_pycorr_inputs(ms: MeasurementSet) -> pycorr.TwoPointCorrelationFunction
adapters/picca.py     to_picca_delta_dir(sl: Sightline, out: Path) -> Path      # Lyα
adapters/nbodykit.py  to_nbodykit_catalog(ps: PointSet) -> nbodykit.CatalogSource
adapters/qp.py        to_qp_ensemble(view) -> qp.Ensemble                       # photo-z
```

Retrofit order for the *existing* native packages (flip/p1desi/lyapower/
lyavoid/lelantos): add a `from_dataproduct(...)` factory on the input side,
keep the native path during a deprecation window, never break current scripts.

---

## 7. Cosmology rule (load-bearing, same spirit as Pillar 1's)

- DataProduct + MeasurementSet carry **no cosmology field**.
- Every estimator entry point takes `cosmology=` explicitly:
  `result = estimator(dp, cosmology=cosmo, ...)`.
- `cosmology=None` → **warn + Planck-2018 default via cosmoprimo**, or raise.
  **Never** a silent hidden baseline.
- Engines: cosmoprimo (recommended façade), CLASS (classy), pyCCL — already
  wired in `flip.power_spectra`.

This is the mirror of [[feedback_no_cosmology_in_pillar1]] on the P2 side: the
data stays portable; the fiducial is always an explicit, swappable choice.

---

## 8. Package map (built vs new)

| Package | role | status |
|---|---|---|
| `onemeasure` | **NEW** — DataProduct + MeasurementSet builders + adapters; the P1→P2 boundary | not built |
| `flip` | field-level f×σ₈ (the anchor family) | built; needs DataProduct ingest |
| `p1desi` | Lyα P_1D | built; native input |
| `lyapower` | Lyα P_3D (Arinyo / Nyx) | built; native input |
| `lelantos` | Lyα tomographic maps | built; native input |
| `lyavoid` | Lyα×void cross-correlation | built; native input |
| `desidescsn` | SN analysis | scripts only |
| `onecorr` | **NEW** — cross / multi-tracer + joint covariance + window deconvolution | not built |
| external | `pycorr`, `nbodykit`, `picca` (consumed via adapters) | external |

---

## 9. Honest built-vs-not summary

- **Built (real science, but each on its own native input, not the contract):**
  flip, p1desi, lyapower, lyavoid, lelantos. They prove Pillar-2 science exists.
- **Not built:** the DataProduct/MeasurementSet contract, `onemeasure` (the
  boundary), `onecorr` (cross/multi-tracer), DataProduct ingestion in any
  existing package, the pycorr clustering adapter, the joint-likelihood layer.
- **Caveat from Pillar 1:** the contract can only be exercised end-to-end on
  the surveys P1 actually ingests — currently **DESI + eBOSS only**
  ([[project_p1_real_ingestion_status]]). Other tracers need their P1 loader
  validated first.

So Pillar 2 = **real estimator science exists in 5 packages; the unifying
contract, boundary package, and cross-correlation layer are undefined-until-now
and unbuilt.** This doc fixes the target; building is sequenced below.

---

## 10. Recommended build order (direction, not commitment)

1. **P0 — `onemeasure` core + PointSet.** DataProduct ABC + PointSet +
   MeasurementSet; build randoms/window/region/n(z) from a synthetic OUF POINT
   dataset (with `oneuniverse.combine` weights); assert invariants §4.1.
2. **PA — flip adopts DataProduct.** `flip.data_vector.from_dataproduct(ps)`;
   synthetic OUF → MeasurementSet → `flip.FitMinuit` recovers f×σ₈.
3. **PB — pycorr adapter.** PointSet×PointSet 2pt; eBOSS×DESI cross via shared
   regions; covariance block check.
4. **PC — Sightline subtype + Lyα adoption.** `from_dataproduct` on p1desi /
   lelantos; DESI DR1 Lyα → P_1D reproduces published.
5. **PD — FieldMap subtype + map×catalog.** galaxy×κ via PointSet×FieldMap.
6. **PE — `onecorr`.** multi-tracer optimal weighting, window deconvolution,
   joint covariance assembly.
7. **PF — joint likelihood (`cobaya`).** multi-probe posteriors over a
   MeasurementSet + theory.

Each becomes a TDD plan (writing-plans) when started. P0–PA is the
minimum-viable spine (boundary + the one built estimator end-to-end).

## 11. Open questions / risks

- **DataProduct generality vs Lyα fit.** Sightlines are genuinely unlike point
  catalogs (no randoms; "window" = LOS selection + masked pixels). The subtype
  split (§4) keeps them distinct under one umbrella — but the *shared* surface
  is thin (region_map + metadata + provenance). Risk: the umbrella adds
  ceremony without payoff unless cross-correlation (galaxy×Lyα) is actually
  pursued. Mitigation: PointSet first; add Sightline only at PC when a consumer
  exists.
- **Retrofit cost.** p1desi/lyapower/lyavoid/lelantos have working native
  pipelines; adopting DataProduct is opt-in `from_dataproduct` factories, not a
  rewrite. Keep native paths.
- **Adapter location.** Prototype in `onemeasure.adapters`; graduate to each
  tool's repo once stable (decouples release cycles).
- **Optional-deps proliferation.** `onemeasure[flip|pycorr|picca|qp]`; consider
  a single `[science]` meta-extra.
- **Tool API drift.** Version-pin downstream tools in adapters.
- **onecorr vs pycorr.** Do not reinvent pycorr; `onecorr` only for genuinely
  new science (multi-tracer optimal weighting, map×catalog, joint covariance).

## 12. References

- [`2026-05-28-pillar2-external-interfaces.md`](2026-05-28-pillar2-external-interfaces.md) — superseded broad roadmap.
- [`2026-05-28-pillar1-data-combine-measure.md`](2026-05-28-pillar1-data-combine-measure.md) — Pillar 1 (DataProduct sources: OUF geometries).
- [`2026-05-28-pillar3-simulation-digital-twin.md`](2026-05-28-pillar3-simulation-digital-twin.md) — Pillar 3 (consumes constraints; produces mocks/covariance).
- `Packages/flip/`, `Packages/p1desi/`, `Packages/lyapower/`, `Packages/lyavoid/`, `Packages/lelantos/` — the five built estimator families.
- [`../research/survey_landscape_review.md`](../research/survey_landscape_review.md) — which cross-correlations are scientifically worth `onecorr`.
