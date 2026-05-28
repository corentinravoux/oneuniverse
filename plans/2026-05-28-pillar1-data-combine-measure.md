# Pillar 1 — Data, Combine, Measure

**Date:** 2026-05-28
**Scope:** Everything inside the `oneuniverse` package. Database
construction from raw catalogs → cross-survey combination → analysis-
ready `MeasurementSet` handoff. **No cosmology, no estimators, no
forward models.**

This document is a **large-scope roadmap**, not a task plan. Detailed
task-level plans for each Phase live under
`plans/YYYY-MM-DD-phaseN-*.md`.

---

## 1. Mission

Stand-up the canonical, model-free representation of every cosmology
survey dataset, plus the cross-survey machinery a downstream analysis
tool needs to consume it consistently. Pillar 1 is the **only** layer
that touches raw FITS/HDF5/CSV and the **only** layer that decides
what columns end up in the OUF parquet. Pillar 1 ends when a
`MeasurementSet` is handed to an external estimator or to
`oneuniverse.simulation`.

## 2. Boundary clarity

| In scope | Out of scope |
|---|---|
| Ingest survey catalogs | Compute P(k), ξ(r), C_ℓ, likelihoods |
| Standardise into OUF parquet | Pick H₀ / Ωₘ / fiducial baseline |
| Cross-match across surveys (ONEUID) | Convert z to comoving / luminosity distance |
| Apply survey-published selection weights | Compute mock covariance |
| Build random catalogs from windows | Fit theory models |
| Assign jackknife regions | Run MCMC / Fisher |
| Build per-bin or per-row n(z) | Forward-model the field |
| Emit `MeasurementSet` for downstream tools | Interpret biases / nuisance parameters |

See [[feedback_no_cosmology_in_pillar1]] for why H₀/Ωₘ live in
Pillars 2/3.

## 3. Subpackages

```
oneuniverse/
├── data/             ── ingest, schema, manifest, ONEUID, sub-object,
│                        DatasetView, partitioning
├── combine/          ── WeightedCatalog, Weight ABC + primitives,
│                        ProductWeight, combiners, registry
└── measure/          ── (NEW) MeasurementSet, random catalogs, window
                         functions, jackknife regions, n(z) builders
```

Phases 1–15 stabilised `data/` and `combine/`. Phases 16+ add
`measure/` and extend `data/` + `combine/` to cover the full survey
landscape from
[`../research/survey_landscape_review.md`](../research/survey_landscape_review.md).

## 4. Status snapshot (2026-05-28)

- 15 phases complete; 365/365 tests green; OUF 2.1.0 stable.
- Real-survey loaders for BOSS/eBOSS/DESI bright-galaxy/Rubin
  remain skeletons (Phase 13 deferred until Phases 16–20 expand the
  schema).
- `oneuniverse.weight` deprecated shim removed 2026-05-28; all entry
  points come from `oneuniverse.combine`.
- `oneuniverse.measure` does not yet exist.

## 5. Roadmap

Each phase ends with a working OUF version bump (where applicable),
green suite, and a Sphinx doc update.

### Phase 16 — Observational metadata expansion (`data/`)

**Goal.** Capture per-column and per-dataset observational metadata
that consumers need to reason about coordinates, frames, and spectra
— without committing to any cosmological model.

**Adds.**
- `CoordinateSpec` (frame, epoch, PM/parallax availability) to
  Manifest.
- `SpectrumSpec` (vacuum/air, log-binning, rest-frame state,
  λ-unit) to Manifest (SIGHTLINE datasets only).
- `ColumnDef` gains `frame`, `epoch`, `wavelength_convention`,
  `nullable`.
- Extensible `Z_TYPE_REGISTRY` with `register_z_type(name)`;
  Manifest records `observed_z_types`.
- Per-column redshift-frame annotation lets a single dataset carry
  both `z_helio` and `z_cmb` columns unambiguously.

**Surveys unlocked.** GAIA DR3 (epoch 2016.0), SDSS Legacy (air λ),
Pantheon+ (`zHD` vs `zHEL`), all PV surveys, Planck PSZ2 (galactic
frame), Lyα `z_lya`.

**OUF bump.** 2.1.0 → 2.2.0.

**Out of scope.** Cosmology metadata, distance-model selection.

### Phase 17 — Variable-length columns + generic partition stats

**Goal.** Lift the fixed-width column assumption that prevents
ingesting Lyα δ pixels, lightcurves, GAIA XP spectra, multi-filter
photometry, and DESI PIP bitweights.

**Adds.**
- `_chunk_to_table(..., column_dtypes=...)` accepts a mini-language:
  `list<f4>`, `f4[N]`, `i8[N]`, `struct{...}`, `list<struct{...}>`.
- `LargeList` support for >2 GB chunks (Lyα δ).
- `PartitionStats.extra_ranges: Dict[str, (lo, hi)]` for arbitrary
  pushdown axes (S/N, EBV, magnitude, BAL probability).
- `DatasetView.select(extra_filters={"snr": (10, None), ...})`.

**Surveys unlocked.** Lyα δ as native OUF rows (no picca sidecar
needed), ZTF/Rubin alert history, GAIA XP, DESI BITWEIGHTS,
multi-filter photometry without hardcoded band columns.

**OUF bump.** 2.2.0 → 2.3.0.

### Phase 18 — PDF polymorphism + tomographic n(z)

**Goal.** Align with `qp` (RAIL ecosystem) and support per-bin
tomographic n(z) as a manifest-level object.

**Adds.**
- `PdfSpec` parameterisations extended: `sample`, `hist`, plus
  sparse `grid_mask` and multi-axis `axis_labels` (multi-D
  posteriors).
- Configurable column-name aliases (`value_column`, `sigma_column`,
  `weights_column`).
- `TomographicNzSpec` on Manifest: bin edges + shared z grid +
  per-bin values + row-level `bin_assignment_column`.
- `ClassificationPdfSpec` on Manifest: ordered class labels +
  `class_pdf_values: f4[n_classes]` column.

**Surveys unlocked.** LSST RAIL native, KiDS-1000 / DES-Y3 /
HSC-Y3 tomographic n(z), classification probabilities (DESI
`SPECTYPE/SUBTYPE`, ZTF/Fink classifier outputs).

**OUF bump.** 2.3.0 → 2.4.0.

### Phase 19 — Shear group + weight expansion (`combine/`)

**Goal.** Make weak lensing a first-class probe and support DESI
PIP bitweights.

**Adds.**
- `SHEAR_COLUMNS` group: `e1, e2, e1_err, e2_err, R11, R22, R12,
  R21, R_S, m_bias, c1_bias, c2_bias, shear_weight`.
- `ShearWeight(kind="metacal" | "lensfit")` propagates
  shape-noise + selection bias.
- `PipBitweightWeight(mode="fraction" | "realisations")` expands
  `BITWEIGHTS: i8[64]`.
- Weight registry key extended to
  `(survey_type, sub_kind, z_type)` for sub-species (e.g. DESI
  `BGS_BRIGHT` vs `BGS_FAINT`).
- Optional `ClassificationWeight` and `TemporalWeight` primitives.

**Surveys unlocked.** DES Y3/Y6, KiDS-1000, HSC-Y3 metadetect,
Rubin shapes, DESI clustering with PIP weights.

**OUF bump.** 2.4.0 → 2.5.0 (because shear columns are a schema
extension).

### Phase 20 — Map-based ONEUID + multi-level sub-object

**Goal.** Cross-match catalogs against probability maps and chain
sub-object hierarchies beyond two levels.

**Adds.**
- `build_subobject_links_to_map(parent_dataset, event_map_dataset,
  *, overlap_kind, threshold, name)`.
- `SubobjectLinks` gains `score_column`, `relation_type`,
  `next_level`, `relation_metadata`.
- `Database.chain_subobjects(start_dataset, relations=[...])` walks
  multi-level chains transitively.
- `CrossMatchRules` accepts `attribute_filters: (AttributeFilter,
  ...)` for "match only if colour difference < 0.1" etc.
- `galaxy_id` widened to accept tuple / bytes payloads.
- Multi-order MOC HEALPix support via `mocpy` (new optional dep).

**Surveys unlocked.** GW × galaxy host association (GWTC), cluster
member chains (cluster → galaxy → spectrum → emission line),
deblender hierarchies (Rubin `parentObjectId → childObjectId →
diaSource`), composite-ID surveys (`PLATE-MJD-FIBERID`).

**OUF bump.** 2.5.0 → 2.6.0.

### Phase 21 — `oneuniverse.measure` — MeasurementSet contract

**Goal.** Introduce the standardised analysis-ready handoff object.
This is the **Pillar 1 / Pillar 2 boundary**: every downstream tool
consumes `MeasurementSet`, nothing parses OUF directly.

**Modules.**
- `oneuniverse/measure/measurement_set.py` — the bundle dataclass.
- `oneuniverse/measure/window.py` — window functions (mask,
  completeness, intersect / union).
- `oneuniverse/measure/random.py` — random-catalog generation from
  a window + n(z).
- `oneuniverse/measure/jackknife.py` — HEALPix-region assignment.
- `oneuniverse/measure/nz.py` — per-bin and per-row n(z) builders.
- `oneuniverse/measure/multitracer.py` — joint sample bundling
  (multiple `MeasurementSet`s sharing window / regions / frame).

**Core type.**

```python
@dataclass(frozen=True)
class MeasurementSet:
    catalog: pa.Table              # rows + applied weights
    randoms: pa.Table              # drawn from window + n(z)
    window: Window                 # mask / completeness
    nz: Nz                         # per-bin or per-row
    region_map: HealpixRegionMap   # jackknife / bootstrap assignment
    metadata: MeasurementMetadata  # frame, epoch, units — no cosmology
    covariance: Optional[CovarianceHandle] = None  # callable → C block

@dataclass(frozen=True)
class MultiTracerMeasurementSet:
    tracers: Mapping[str, MeasurementSet]
    shared_region_map: HealpixRegionMap
    shared_metadata: MeasurementMetadata
```

**Contract guarantees.** Every tracer in a multi-tracer bundle shares:
- one HEALPix jackknife region assignment (NSIDE configurable)
- one frame / epoch convention
- compatible n(z) grids (resampled to a common grid)
- intersected or unioned window (caller chooses)
- joint or per-tracer random catalogs (caller chooses)

**What MeasurementSet does not own.** Estimator math, theory
predictions, cosmology conversion, likelihoods. All deferred to
Pillar 2 / Pillar 3.

**Surveys unlocked.** All of them — `MeasurementSet` is what makes
`oneuniverse` usable to any external analysis tool.

**OUF bump.** Not applicable — `MeasurementSet` is an in-memory
handoff, not a persisted format. (We may add `MeasurementSet.write()`
later for caching but that's optional.)

### Phase 22 (optional) — Geometry expansion

**Goal.** Add geometries needed for sims and exotic data.

**Adds.**
- `CUBE` geometry for IFU (MaNGA, SAMI, MUSE), HI cubes, 21 cm
  intensity-mapping cubes.
- `PARTICLE` geometry for mock snapshots (AbacusSummit, MillenniumTNG,
  UNIT, Outer Rim).
- `GW_SKYMAP` geometry for row-per-event HEALPix probability maps.

Defer until concrete consumers appear. Pillar 3 (simulation) is the
likely first consumer of `PARTICLE`.

## 6. Loader onboarding (rolled-up Phase 13)

Phase 13 (real-survey loader writes) was postponed because the
schema couldn't carry every survey's columns. Once Phases 16–20
land, loaders are absorbed into the appropriate phase:

| Loader | Lands with phase | Why |
|---|---|---|
| BOSS / eBOSS clustering (galaxy + QSO) | 17, 19 | bitweights + PIP, multi-z columns |
| eBOSS Lyα δ as OUF native | 17 | variable-length δ arrays |
| DESI BGS / LRG / ELG / QSO | 16, 17, 19 | `Z_TYPE` + BITWEIGHTS + variable filters |
| DESI Lyα DR1 / DR2 | 17 | δ arrays + MOC HEALPix masks |
| DES Y3 / Y6 shear | 19 | shear group + ShearWeight |
| KiDS-1000 / HSC-Y3 shear | 19 | lensfit-mode ShearWeight |
| Rubin DP0.2 / DR1 photo-z | 18, 19 | qp `sample` PDFs + per-band shear |
| Euclid VIS + NISP photo-z | 18 | NNPZ 601-bin gridded PDFs |
| Roman HLSS grism | 17 | grism spectra as SIGHTLINE |
| Pantheon+ / Union3 SN Ia | 16, 17 | frame disambiguation + correlated covariance sidecar |
| GAIA DR3 | 16, 17 | epoch 2016.0 + XP spectra |
| GWTC-3 / GWTC-4 | 20 | row-per-event HEALPix payload |
| Planck PSZ2 / ACT / SPT clusters | 16, 20 | galactic frame + cluster→galaxy chains |
| eROSITA-DE DR1 | 16, 17 | per-source spectra + lightcurves |

Phase 23 = consolidated real-survey loader writes; ships once
prerequisite schema phases are done.

## 7. Deliverables checklist (Pillar 1 done)

A reasonable definition of "Pillar 1 complete" is:

- [ ] All 13 representative loaders above produce green-suite OUF
      datasets from real data.
- [ ] `MeasurementSet` API stable; round-trips through `flip` and at
      least one external tool (`pycorr` or `nbodykit`) without
      glue code.
- [ ] `MultiTracerMeasurementSet` round-trips through a cross-
      correlation estimator (likely `pycorr` or a thin in-flip path).
- [ ] Documentation: Sphinx autosummary covers `measure/`; tutorials
      for both standard and multi-tracer workflows.
- [ ] All cross-cutting modality items 1–17 from
      `research/survey_landscape_review.md` covered or explicitly
      deferred.

## 8. Things explicitly deferred

- Alert ingestion (ZTF/Rubin alert streams as Avro) — Pillar 1 in
  principle but high engineering cost; revisit after Phase 23.
- IFU cubes, HI cubes, 21 cm cubes — `CUBE` geometry, Phase 22+.
- N-body particle snapshots — `PARTICLE` geometry, owned jointly
  with Pillar 3.
- Cosmology baseline metadata — out of scope forever (see Pillar 2/3).

## 9. References

- [`../research/survey_landscape_review.md`](../research/survey_landscape_review.md)
  — cosmology survey landscape that drives schema needs.
- [`../research/schema_generalisation_audit.md`](../research/schema_generalisation_audit.md)
  — file-by-file audit + concrete API for Phases 16–22.
- [`2026-04-15-stabilisation-roadmap.md`](2026-04-15-stabilisation-roadmap.md)
  — original Phase 1–6 roadmap (historical).
- [`README.md`](README.md) — phase status table.
