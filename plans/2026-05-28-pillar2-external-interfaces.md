# Pillar 2 — `MeasurementSet` + External Scientific Tool Interfaces

**Date:** 2026-05-28 (revised 2026-05-29 — `MeasurementSet`
construction reassigned from Pillar 1 to Pillar 2).
**Scope:** `MeasurementSet` construction, window / random / jackknife /
n(z) builders, and the adapter layer that hands a `MeasurementSet`
to every external tool. Lives **outside** the `oneuniverse` package
(`oneuniverse` is Pillar 1 only).

This document is a **large-scope roadmap**, not a task plan.
Implementation lives in a new package (proposed name `onemeasure` or
similar) plus downstream tools (`flip`, future `onecorr`, third-party
`pycorr`, `nbodykit`, `picca`).

---

## 1. Mission

Bridge Pillar 1's on-disk OUF parquet (catalogs + weights + ONEUID +
sub-object sidecars) to the actual science — P(k), ξ(r), C_ℓ, joint
multi-tracer estimators, likelihoods, fits — by **owning** the
analysis-ready `MeasurementSet` contract and the adapters every
downstream tool consumes. Pillar 2 is where **cosmology enters**: H₀,
Ωₘ, distance models, theory templates.

## 2. Boundary clarity

| In scope | Out of scope |
|---|---|
| Read OUF parquet + ONEUID + weights from disk | Ingest raw FITS / write OUF |
| Build randoms / windows / regions / n(z) | Cross-match surveys (Pillar 1) |
| Construct `MeasurementSet` + `MultiTracerMeasurementSet` | Apply per-survey selection weights (Pillar 1) |
| Adapters to flip / pycorr / picca / qp / nbodykit | Standardise survey columns (Pillar 1) |
| Compute estimators (P(k), ξ(r), C_ℓ, …) | Forward-model the field (Pillar 3) |
| Pick fiducial cosmology for comoving conversion | Run sims / hydro / IC sampling (Pillar 3) |
| Theory model evaluation (limber, CLASS, CAMB, pyCCL) | |
| Likelihoods + samplers (cobaya, emcee, dynesty) | |
| Multi-tracer optimal weighting (Abramo-Leonard FKP) | |

## 3. Architectural shape

Pillar 2 is a **new package** (e.g. `onemeasure`) that owns the
`MeasurementSet` builders + adapters. Estimator tools (`flip`,
`pycorr`, `picca`, `nbodykit`) consume `MeasurementSet`s; they live
in their own repos.

```
        ┌──────────────┐
        │ oneuniverse  │  Pillar 1 (this repo)
        │   data/      │  → OUF parquet + ONEUID + sub-object
        │   combine/   │     sidecars on disk
        └──────┬───────┘
               │ disk artefacts
        ┌──────▼───────┐
        │  onemeasure  │  Pillar 2 (new package)
        │              │  reads OUF → builds randoms / windows /
        │              │  jackknife regions / n(z) → emits
        │              │  MeasurementSet + adapters
        └──────┬───────┘
               │ MeasurementSet
   ┌───────────┼───────────┐
   ▼           ▼           ▼
┌──────┐ ┌─────────┐ ┌──────────┐  Pillar 2 (external)
│ flip │ │ pycorr  │ │  picca   │  estimator tools
└──────┘ └─────────┘ └──────────┘
   │           │           │
   └───────────┼───────────┘
               ▼
       Result + covariance
```

Pillar 2 deliverables:
1. **`onemeasure` package** — `MeasurementSet` + builders (window,
   random, jackknife, n(z), multi-tracer).
2. **`MeasurementSet` adapters** — thin shims per estimator tool
   that consume the contract and convert to native tool input.
   Live inside `onemeasure.adapters` (auditable in one place) or in
   each tool's own repo.
3. **Cross-tool covariance bookkeeping** — block structure for
   joint estimators. Likely a sibling package (`onecorr` /
   `oneinference`).
4. **Documentation + tutorials** showing the end-to-end workflow
   `oneuniverse.data → .combine → onemeasure → tool`.

## 4. Subsystems

### 4.1 Adapters (thin layer in `onemeasure.adapters`)

One adapter per supported tool. Lives inside `onemeasure` so the
contract stays auditable in one place, but is opt-in (each adapter
has its own optional dependency).

```python
# onemeasure/adapters/flip.py
def to_flip_data_vector(ms: MeasurementSet, *, kind="velocity") -> "flip.DataVector": ...

# onemeasure/adapters/pycorr.py
def to_pycorr_inputs(ms: MeasurementSet) -> "pycorr.TwoPointCorrelationFunction": ...

# onemeasure/adapters/nbodykit.py
def to_nbodykit_catalog(ms: MeasurementSet) -> "nbodykit.source.CatalogSource": ...

# onemeasure/adapters/picca.py
def to_picca_delta_dir(ms: MeasurementSet, out_dir: Path) -> Path: ...

# onemeasure/adapters/qp.py
def to_qp_ensemble(view: DatasetView) -> "qp.Ensemble": ...
```

**Rule.** Adapters never compute science. They only re-shape data.

### 4.2 `flip` — velocity + density field-level inference

Already exists at `Packages/flip/`. Pillar 2 deliverables:
- Accept `MeasurementSet` directly in `FitMinuit.init_from_*` and
  `FitMCMC.init_from_*` constructors.
- Deprecate raw-dict input in favour of `MeasurementSet`.
- Add `MultiTracerMeasurementSet` support to joint density+velocity
  fits (currently `DensVel` data vectors).
- Document end-to-end notebook:
  `oneuniverse → flip MeasurementSet → f×σ₈ fit`.

### 4.3 Cross-correlation toolkit (`onecorr` — new package, deferred)

Standalone package for cross-survey multi-tracer estimators that
don't fit cleanly inside `flip` or `pycorr`. Owns:
- 2pt cross-correlation between arbitrary pairs of `MeasurementSet`s
  (galaxy × galaxy, galaxy × PV, galaxy × κ map, galaxy × tSZ).
- Map-vs-catalog estimators (galaxy × HEALPix CMB κ, galaxy ×
  Compton-y).
- Multi-tracer optimal weighting (Abramo-Leonard FKP, per-tracer
  FKP).
- Window-function deconvolution.
- Joint covariance assembly from shared jackknife regions.

**Status.** Not started. Likely Phase 24+. Defer until at least
two `MeasurementSet` adapters are working (flip + pycorr).

### 4.4 Theory layer (cosmology baseline enters here)

Pillar 2 tools each pick their own fiducial cosmology. Recommended
shared idiom:

```python
from cosmoprimo import Cosmology
from onemeasure import build_measurement_set

cosmo = Cosmology(h=0.7, Omega_m=0.315, ...)

ms = build_measurement_set(...)        # onemeasure, cosmology-free
result = pycorr_estimator(ms, cosmology=cosmo)   # estimator converts at call
```

`MeasurementSet` itself carries no `cosmology` field — frame / epoch /
unit metadata only. Every estimator entry point takes cosmology as a
kwarg.

**Engines to support.** pyCCL, CLASS (classy), cosmoprimo (recommended
default; lightweight; cross-engine façade). Already wired in
`flip.power_spectra`.

### 4.5 Likelihood + sampler layer

Wraps the above for inference. Each tool owns its own choice:
- `flip` ships `FitMinuit` (iminuit) + `FitMCMC` (emcee) +
  `FisherForecast`. Adequate.
- For posterior comparison and joint multi-probe fits, wrap into
  `cobaya` likelihoods. New `onecobaya` package optional.

## 5. The `MeasurementSet` contract — invariants

`onemeasure` builds a `MeasurementSet` from Pillar 1 disk artefacts;
estimator tools (`flip`, `pycorr`, `picca`, …) consume it. Every
`MeasurementSet` carries the following invariants:

1. `catalog` is a `pa.Table` with at minimum `ra`, `dec`, `z`,
   `weight`, and a `region_id` column.
2. `randoms` are drawn from the same `window` and `nz` as `catalog`,
   and carry compatible `region_id`s.
3. `region_map` is a HEALPix array (NSIDE declared in
   `MeasurementMetadata`) — guarantees jackknife / bootstrap
   consistency across tracers.
4. `metadata` carries `frame` (icrs/galactic/ecliptic), `epoch`,
   `unit` declarations. **No cosmology.**
5. `nz` exposes `pdf(z)` (per-bin) or `pdf(z, row_id)` (per-row).

For `MultiTracerMeasurementSet`:

6. All tracers share `region_map`.
7. All tracers share `metadata.frame` / `metadata.epoch`.
8. n(z) grids resampled to a common grid (caller chose intersect /
   union).

Estimator tools that need cosmological conversion (`z → r_comoving`)
apply it locally, using their own fiducial.

## 6. Roadmap (large strokes)

Pillar 2 work begins by spinning up the new `onemeasure` package and
shipping its first `MeasurementSet` builder against the existing OUF
2.4 disk format from Pillar 1. Phases A–F below are sequenced for
that order.

### Phase 0 — Stand up `onemeasure`

- New package: `onemeasure` (separate repo).
- Modules: `measurement_set.py`, `window.py`, `random.py`,
  `jackknife.py`, `nz.py`, `multitracer.py`, `adapters/`.
- Imports `oneuniverse` as a read-only dependency.
- End-to-end test: synthetic OUF POINT dataset (with `weight` column
  from `oneuniverse.combine`) → `MeasurementSet` → asserts
  invariants 1–5 above.

### Phase A — `flip` accepts `MeasurementSet`

- Add `flip.data_vector.from_measurement_set(ms, kind=...)` factory.
- Deprecate raw-dict constructors over one minor version.
- End-to-end test: synthetic OUF dataset → `MeasurementSet` →
  `flip.FitMinuit` → f×σ₈ recovers truth.
- End-to-end test: joint density + velocity via
  `MultiTracerMeasurementSet`.
- Tutorial notebook.

### Phase B — `pycorr` adapter + 2pt cross-correlation example

- `to_pycorr_inputs(ms)` adapter.
- End-to-end: eBOSS × DESI 2pt cross-correlation via shared jackknife
  regions.
- Validate covariance block structure (off-diagonal blocks
  consistent with single-tracer diagonals).

### Phase C — `picca` adapter for Lyα

- `to_picca_delta_dir(ms)` writes picca-format delta files from an
  OUF-native Lyα dataset (post-Phase-17).
- End-to-end: DESI DR1 Lyα → picca → P_1D / 3D recovers published
  results.

### Phase D — Map-vs-catalog estimators

- Adapter that pairs a catalog `MeasurementSet` with a HEALPix map
  (CMB κ, tSZ y, depth maps, etc.).
- Cross-correlation estimator (likely in a small new module since no
  external tool natively accepts this shape).

### Phase E — `onecorr` (new package)

- Multi-tracer optimal weighting.
- Window deconvolution.
- Joint covariance assembly.
- Theory-prediction layer for cross-spectra (limber + non-limber).

Defer until Phases A–D are stable.

### Phase F — `cobaya` likelihood wrappers

- Joint multi-probe likelihoods that take a list of
  `MeasurementSet`s + theory predictions.
- Posterior comparison across estimator choices.

## 7. What stays in `flip` (vs new packages)

`flip` keeps its focus on field-level inference with velocity +
density. Anything that fits the flip methodology lives in flip.
Cross-correlation between arbitrary tracers that does **not** fit
the flip covariance-model abstraction belongs in `onecorr`.

Rough split:
- **flip:** velocity covariance, density covariance, joint
  density-velocity covariance, f×σ₈ inference, FitMinuit / FitMCMC /
  Fisher, likelihood inversions, JAX acceleration.
- **onecorr:** model-agnostic 2pt estimators, map-vs-catalog,
  multi-tracer optimal weighting, joint covariance bookkeeping.
- **External (pycorr, nbodykit, picca):** existing tools, consumed
  via adapters.

## 8. Cosmology choices — explicit ownership

Every Pillar 2 entry point that needs cosmology declares it
explicitly:

```python
result = estimator(ms, cosmology=cosmo, ...)
```

Defaults (when none provided): Planck 2018 fiducial via cosmoprimo.
**Never silent**: `cosmology=None` should warn and use the default,
or raise — never proceed with a hidden baseline.

This rule makes Pillar 1's `MeasurementSet` portable: the same
measurement can be re-analysed under a different fiducial without
re-ingesting data.

## 9. Deliverables checklist (Pillar 2 minimum-viable)

- [ ] `onemeasure.adapters.flip` round-trips synthetic
      OUF → flip f×σ₈ fit.
- [ ] `onemeasure.adapters.pycorr` round-trips synthetic
      OUF → pycorr 2pt.
- [ ] Multi-tracer cross-correlation tutorial (eBOSS × DESI BGS)
      using `MultiTracerMeasurementSet`.
- [ ] Documentation that explains: which tool to use for which
      science, how cosmology choice is made explicit.
- [ ] At least one end-to-end re-analysis of a published result
      using the new contract.

## 10. Risks + open questions

- **Adapter location.** Adapters could live in `oneuniverse` (audit
  in one place) or in each tool's repo (decouples release cycles).
  Recommend: prototype in `onemeasure.adapters`, graduate
  to tool repos once stable.
- **Optional deps proliferation.** `oneuniverse[flip]`,
  `oneuniverse[pycorr]`, `oneuniverse[picca]`, … extras list grows.
  Consider a single `oneuniverse[science]` meta-extra.
- **Tool API drift.** Adapters need versioned pins on downstream
  tools. Plan for `if pycorr.__version__ < "X.Y": raise`.
- **`onecorr` vs `pycorr`.** Do not reinvent `pycorr`. `onecorr`
  should only exist for genuinely new science (multi-tracer optimal
  weighting + map-vs-catalog).
- **Cosmology engine choice.** cosmoprimo recommended but
  `flip.power_spectra` supports CCL + CLASS too. Document the
  default; allow override.

## 11. References

- [`2026-05-28-pillar1-data-combine-measure.md`](2026-05-28-pillar1-data-combine-measure.md)
  — Pillar 1 roadmap (must complete through Phase 21 first).
- [`2026-05-28-pillar3-simulation-digital-twin.md`](2026-05-28-pillar3-simulation-digital-twin.md)
  — Pillar 3 roadmap (simulation side of `MeasurementSet`).
- `Packages/flip/` — current flip implementation (Pillar 2 incumbent).
- [`../research/survey_landscape_review.md`](../research/survey_landscape_review.md)
  — drives what cross-correlations are scientifically interesting.
