# oneuniverse Plans

Phase-level and pillar-level roadmaps for the oneuniverse project.
The plans here describe **why** and **how** each chunk of work lands;
detailed task lists live in dated phase files.

## Three-pillar structure (2026-05-28)

The whole stack is organised around three pillars:

- **Pillar 1 — Data + Combine.** Everything inside the `oneuniverse`
  package. Database construction from raw catalogs → cross-survey
  combination → typed parquet artefacts on disk. No cosmology, no
  estimators, no forward models, **no `MeasurementSet`** (that
  moved to Pillar 2 on 2026-05-29).
  See [`2026-05-28-pillar1-data-combine-measure.md`](2026-05-28-pillar1-data-combine-measure.md).

- **Pillar 2 — `MeasurementSet` + External Scientific Tool
  Interfaces.** A new `onemeasure` package owns the `MeasurementSet`
  contract + builders (window, random, jackknife, n(z),
  multi-tracer) + adapters to `flip`, `pycorr`, `picca`, future
  `onecorr`. Pillar 2 is where cosmology enters.
  See [`2026-05-28-pillar2-external-interfaces.md`](2026-05-28-pillar2-external-interfaces.md).

- **Pillar 3 — Simulation / Digital Twin.** Constrained Bayesian
  forward modelling of the actual Universe; per-survey observation
  models; mini-simulation zoom-ins; incremental updates.
  See [`2026-05-28-pillar3-simulation-digital-twin.md`](2026-05-28-pillar3-simulation-digital-twin.md).

## Pillar 1 phase plans

### Stabilisation (Phases 1–15, complete 2026-05-22)

- [`2026-04-15-stabilisation-roadmap.md`](2026-04-15-stabilisation-roadmap.md) — master phase-level roadmap with rationale, design decisions, and scope per phase.
- [`2026-04-15-phase1-ouf-v2.md`](2026-04-15-phase1-ouf-v2.md) — Phase 1 (OUF 2.0 manifest + atomic writes + content hashing).
- [`2026-04-15-phase3-healpix-partitioning.md`](2026-04-15-phase3-healpix-partitioning.md) — Phase 3 (HEALPix spatial partitioning + cone / SkyPatch cell pruning).
- [`2026-04-15-phase4-unified-oneuid.md`](2026-04-15-phase4-unified-oneuid.md) — Phase 4 (CrossMatchRules policy, relocated cross-matcher, audit columns, named on-disk ONEUID indices, `WeightedCatalog.from_oneuid`).
- [`2026-04-16-phase5-streaming-hydration.md`](2026-04-16-phase5-streaming-hydration.md) — Phase 5 (row-level pushdown via `_original_row_index`, `OneuidQuery.iter_partial`).
- [`2026-04-16-phase6-housekeeping-combine.md`](2026-04-16-phase6-housekeeping-combine.md) — Phase 6 (`oneuniverse.weight` → `oneuniverse.combine` redesign, `default_weight_for`). Final `oneuniverse.weight` deprecation shim deleted 2026-05-28.
- [`2026-04-20-temporal-subobject-roadmap.md`](2026-04-20-temporal-subobject-roadmap.md) — joint roadmap for Phase 7 + 8.
- [`2026-04-20-phase7-temporal.md`](2026-04-20-phase7-temporal.md) — Phase 7 (temporal).
- [`2026-04-20-phase8-subobject.md`](2026-04-20-phase8-subobject.md) — Phase 8 (sub-object).
- [`2026-04-23-phase9-desi-dr1-onboarding.md`](2026-04-23-phase9-desi-dr1-onboarding.md) — Phase 9 (DESI DR1 QSO onboarding + fragility audit).
- [`2026-04-23-phase10-probabilistic-redshifts.md`](2026-04-23-phase10-probabilistic-redshifts.md) — Phase 10 (photo-z PDFs).
- [`2026-04-23-phase11-selection-weights.md`](2026-04-23-phase11-selection-weights.md) — Phase 11 (selection / completeness weight family).
- [`2026-05-22-phase12-carried-debt.md`](2026-05-22-phase12-carried-debt.md) — Phase 12 (carried-over debt cleanup).
- [`2026-05-22-phase14-performance-footprint.md`](2026-05-22-phase14-performance-footprint.md) — Phase 14 (performance + footprint).
- [`2026-05-22-phase15-docs-stability.md`](2026-05-22-phase15-docs-stability.md) — Phase 15 (docs + stability).

### Generalisation (Phases 16–22, planned 2026-05-28)

Driven by [`../research/survey_landscape_review.md`](../research/survey_landscape_review.md)
and [`../research/schema_generalisation_audit.md`](../research/schema_generalisation_audit.md).
Per-phase detailed plans land as we start each.

| # | Name | Driver |
|---|------|--------|
| 16 | Observational metadata expansion (`CoordinateSpec`, `SpectrumSpec`, `z_type` registry, `ColumnDef.frame/epoch/λ-convention/nullable`) | GAIA epoch, SDSS air vs BOSS vacuum, multi-z columns |
| 17 | Variable-length columns + generic partition stats | Lyα δ, ZTF/Rubin lightcurves, GAIA XP, DESI BITWEIGHTS, multi-filter photometry, S/N pushdown |
| 18 | PDF polymorphism + tomographic n(z) + classification PDFs | RAIL / `qp` alignment, KiDS/DES/HSC tomographic bins |
| 19 | Shear group + `ShearWeight` + `PipBitweightWeight` | DES Y3/Y6, KiDS-1000, HSC-Y3, Rubin shapes, DESI PIP |
| 20 | Map-based ONEUID + multi-level sub-object chains | GW × galaxy, cluster→galaxy→spec, deblender trees |
| 21 | Cleanup of deferred sub-object items (composite-ID `galaxy_id`, `CrossMatchRules.attribute_filters`, `mocpy` multi-order MOC) | GWTC native ingest, composite-ID surveys, colour-aware cross-match |
| 22 | Data-driven geometry expansion: `CUBE` (observed IFU/HI/21cm) + `GW_SKYMAP` (event probability maps) — **no mocks** (`PARTICLE` is Pillar 3) | IFU cubes, HI cubes, 21cm intensity maps, GW skymaps |
| 23 | Real-survey loader writes (rolled-up Phase 13) | All loaders, depends on 16–20 |

## Phase status (Pillar 1)

| # | Name | Status |
|---|------|--------|
| 1 | OUF 2.0 (typed manifest, hashes, atomic writes) | **complete (2026-04-15, 130/130)** |
| 2 | DatasetView + pyarrow.dataset backend | **complete (2026-04-15, 145/145)** |
| 3 | HEALPix spatial partitioning | **complete (2026-04-15, 156/156)** |
| 4 | Unified ONEUID (z-type rules, subsets, named indices) | **complete (2026-04-16, 190/190)** |
| 5 | Streaming hydration | **complete (2026-04-16, 198/198)** |
| 6 | Housekeeping + `weight/` → `combine/` redesign | **complete (2026-04-20, 211/211)** |
| 7 | Temporal data (t_obs + LIGHTCURVE + bitemporal database + versioned ONEUID) | **complete (2026-04-21, 265/265)** |
| 8 | Sub-object hierarchy (bitemporal link sidecars) | **complete (2026-04-22, 292/292)** |
| 9 | DESI DR1 QSO onboarding end-to-end + fragility audit | **complete (2026-04-23, 299/299; F1+F2 fixed, F3 deferred)** |
| 10 | Probabilistic redshifts (photo-z PDFs) | **complete (2026-04-23, 326/326)** |
| 11 | Generic selection / completeness weight family | **complete (2026-04-23, 345/345)** |
| 12 | Carried-over debt cleanup | **complete (2026-05-22, 361/361)** |
| 14 | Performance + footprint | **complete (2026-05-22, 364/364; suite 277s → 205s)** |
| 15 | Docs + stability hardening | **complete (2026-05-22, 365/365)** |
| — | `oneuniverse.weight` deprecation shim deleted | **complete (2026-05-28, 365/365)** |
| 16 | Observational metadata expansion (`CoordinateSpec`, `SpectrumSpec`, extensible `z_type` registry, `ColumnDef.frame/epoch/wavelength_convention/nullable`, OUF → 2.2.0) | **complete (2026-05-28, 406/406 tests green)** |
| 17 | Variable-length columns + generic `PartitionStats.extra_ranges` (dtype mini-language, `column_dtypes` writer kwarg, `extra_stats_columns`, `DatasetView.extra_filters`, OUF → 2.3.0) | **complete (2026-05-29, 428/428 tests green)** |
| 18 | PDF polymorphism (`sample` / `hist`) + column aliases + `TomographicNzSpec` + `ClassificationPdfSpec` (OUF → 2.4.0) | **complete (2026-05-29, 450/450 tests green)** |
| 19 | Shear column group + `ShearWeight` + `PipBitweightWeight` + sub-species registry key | **complete (2026-05-29, 472/472 tests green)** |
| 20 | Map-based sub-object (point × HEALPix probability map) + multi-level chain walker + `relation_type` / `next_level` on `SubobjectRules` | **complete (2026-05-29, 487/487 tests green)** |
| 21 | Cleanup of deferred sub-object items (`CrossMatchRules.attribute_filters`, CORE `composite_id`, `mocpy` MOC rasteriser) | **complete (2026-05-29, 499/499 tests green; 2 skipped — mocpy optional)** |
| 22 | Data-driven geometry expansion (CUBE + GW_SKYMAP only; PARTICLE → Pillar 3) | optional |
| 23 | Real-survey loader writes (rolled-up Phase 13) | planned (after 16–20) |

## Pillar 2 / Pillar 3 phase plans

Pillar 2 begins with Phase 0: standing up the new `onemeasure` package
(MeasurementSet + builders + adapters) against the current OUF 2.4
format. Pillar 3 follows once `onemeasure` ships its first adapter.
Per-phase detailed plans get written when work begins.

- Pillar 2 phases A–F: see
  [`2026-05-28-pillar2-external-interfaces.md`](2026-05-28-pillar2-external-interfaces.md).
- Pillar 3 phases α–η: see
  [`2026-05-28-pillar3-simulation-digital-twin.md`](2026-05-28-pillar3-simulation-digital-twin.md).
