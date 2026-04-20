# Temporal & Sub-object Extension Roadmap

**Started:** 2026-04-20 (post Pillar 1 stabilisation)
**Applies to:** `oneuniverse` package, post-Phase-6.

## Motivation

Pillar 1 stabilisation (Phases 1–6) assumed a **static** database: every
row is a point in (ra, dec, z) space with no time dimension and no
structural relationship to other rows. Two real-world use-cases break
that assumption:

1. **Temporal data** — supernovae, tidal disruption events, variable
   stars, image-differencing alerts. Every row has an observation epoch
   (typically MJD). Some surveys produce multi-epoch photometric
   **lightcurves**: one object, N observations over time.
2. **Sub-object hierarchy** — a galaxy *hosts* one or more supernovae; a
   cluster *contains* member galaxies; a sightline *passes through* a
   set of absorbers. The catalogs are distinct rows in distinct datasets
   but carry a physical parent/child relation.

Neither is served by the existing abstractions. The ONEUID index
unifies rows that describe *the same object*; a supernova is not the
same object as its host galaxy, so an ONEUID merge is the wrong tool.
Partitioning and `DatasetView.scan` know nothing about time.

This roadmap adds both capabilities as two independent phases that can
ship sequentially.

## Scope boundary (non-negotiable)

`oneuniverse` is a **data/interface** layer. It does not fit
lightcurves, does not classify transients, does not model host-galaxy
environments, does not estimate SN host associations from scratch. The
temporal and sub-object work is about **storage, inventory, indexing,
and query** — surface the information cleanly so downstream packages
(`flip`, forward-model code, notebooks) can consume it.

Out of scope for every task in this roadmap:

- Any statistical inference beyond deterministic coordinate/tolerance
  matching.
- Any transient classification, light-curve fitting, peak-brightness
  estimation.
- Any host-galaxy property modelling (stellar mass, SFR, metallicity).
- Any forward-model or simulation code.

## Fixed design decisions

These decisions are locked in; they came out of the 2026-04-20 design
pass. Do not re-debate them during implementation without raising with
the user first.

### Temporal

1. **Ride on existing geometries first.** Any POINT or SIGHTLINE
   dataset can carry an optional `t_obs` column (float64, MJD). A new
   `LIGHTCURVE` geometry is added for genuinely multi-epoch data where
   storing one row per (object, epoch) is the right shape.
2. **Time unit: MJD, float64, TDB-referenced by default.** Other time
   references (TAI, UTC) are opt-in via a `TemporalSpec.time_reference`
   manifest field. No survey converter should emit JD or BJD silently.
3. **No migration of existing converted datasets.** OUF 2.x is
   forward-compatible: manifests without a `temporal` block behave
   exactly as before. Loaders opt in when they add `t_obs` to their
   schema.
4. **Partition stats carry `t_min` / `t_max`.** `DatasetView.scan(t_range=…)`
   prunes partitions before opening Parquet, mirroring the existing
   ra/dec/z pruning path.
5. **`LIGHTCURVE` geometry is a structural twin of `SIGHTLINE`.**
   `objects.parquet` holds per-object metadata (object_id, ra, dec, z,
   n_epochs, mjd_min, mjd_max, …). `part_*.parquet` holds per-epoch
   rows keyed by `object_id` and `mjd`. Same disk shape, same writer
   pattern, same reader pattern. `SIGHTLINE` proves the design.
6. **No per-survey `t_obs` column for `SIGHTLINE`.** A SIGHTLINE is
   fundamentally a single-epoch coadd; if a survey does want temporal
   sightlines, it ships as `LIGHTCURVE` instead.
7. **Writer version bump: OUF format → `2.1.0`, schema → `2.1.0`.** The
   bump is additive — `2.0.0` readers still work on `2.1.0` datasets
   that omit temporal metadata. Manifest reader treats `2.0.x` and
   `2.1.x` as compatible; only `3.x` would break.

### Sub-object

1. **Sidecar tables, not per-survey manifest entries.** Sub-object
   links live at `{root}/_subobject/<name>.parquet` with a companion
   `.manifest.json`, mirroring the named, versioned ONEUID index
   layout. Multiple link sets (e.g. `sne_in_hosts`, `galaxies_in_clusters`)
   coexist.
2. **Link currency is ONEUID, not per-survey row index.** Building
   sub-object links requires a covering ONEUID index for the involved
   datasets. This keeps the link stable under re-partitioning and
   survives database rebuilds.
3. **Children keep their own ONEUID.** A supernova is a distinct
   astrophysical object from its host galaxy; merging them into one
   ONEUID would be physically wrong. The sub-object link records a
   **relation**, not an identity.
4. **Driven by a `SubobjectRules` dataclass**, symmetric in spirit to
   `CrossMatchRules` but separate: different physics, different
   tolerances, different failure modes. Frozen dataclass, content-hash
   for audit, `parent_survey_type` / `child_survey_type` enforced.
5. **Many-to-many is natural.** One host can have multiple SNe; one SN
   could in principle have ambiguous hosts (only 1 accepted by
   default; exposed via `accept_ambiguous` rule flag).
6. **Confidence scores persist.** Every link row carries
   `confidence` (float32), derived from the matching algorithm
   (sky-separation + redshift-consistency). Downstream code may filter.

## Phase index

| Phase | Scope | Plan | Status |
|-------|-------|------|--------|
| 7     | Temporal data (epoch-tagged POINT/SIGHTLINE + new LIGHTCURVE geometry) | [`2026-04-20-phase7-temporal.md`](2026-04-20-phase7-temporal.md) | pending |
| 8     | Sub-object hierarchy (named, versioned link tables) | [`2026-04-20-phase8-subobject.md`](2026-04-20-phase8-subobject.md) | pending |

Phase 7 and Phase 8 are **independent**: neither structurally depends
on the other. Phase 7 is recommended first because it exercises the
manifest/partition-stats extension path that will be touched again in
Phase 8. Ship them separately.

## Memory & graph touchpoints consulted while writing this roadmap

- Memory records:
  - `project_oneuniverse_stabilisation.md` (phase status, design commitments)
  - `project_oneuniverse.md` (three-pillar vision, current architecture)
  - `project_oneuniverse_scope.md` (data/interface boundary)
  - `project_digital_twin_vision.md` (longer-term SNe/transient use-case)
  - `feedback_visual_testing.md` (diagnostic figure requirement)
- Graphify communities (`graphify-out/GRAPH_REPORT.md`):
  - **Format Spec & Geometry** (`DataGeometry` as 92-edge god node)
  - **OUF Manifest Schema** (`Manifest`, `PartitionSpec`, `PartitionStats`, `PartitioningSpec`)
  - **DatasetView Query** (`_select_partitions`, `_range_expr`, `_spatial_expr`)
  - **ONEUID Index** (`OneuidIndex`, `OneuidQuery`, `build_oneuid_index`)
  - **Cross-Match Rules** (`CrossMatchRules` as structural reference for `SubobjectRules`)

## Success criteria per phase

Phase 7: (a) an existing SN loader can emit a POINT dataset with a
`t_obs` column, a `TemporalSpec` in its manifest, and partition-level
`t_min`/`t_max`; (b) `DatasetView.scan(t_range=(t0, t1))` returns only
rows within the window without reading pruned partitions; (c) a
synthetic LIGHTCURVE dataset can be written end-to-end, with
`objects.parquet` + per-epoch `part_*.parquet`, and `DatasetView` can
return either the objects table or the epoch rows with time-range
pruning. Full test suite green (≥ +20 new tests). One diagnostic
figure showing MJD distribution and sky map of a temporal POINT
dataset.

Phase 8: (a) `SubobjectRules` declares matching policy between parent
and child survey types with sky + redshift tolerances; (b)
`database.build_subobject_links(rules, parent_datasets, child_datasets,
name=…)` writes a `_subobject/<name>.parquet` sidecar keyed on ONEUIDs;
(c) `database.load_subobject_links(name).children_of(parent_oneuid)`
and `.parent_of(child_oneuid)` work; (d) ambiguous matches are either
rejected or exposed via `accept_ambiguous` with a confidence score.
Full test suite green (≥ +15 new tests). One diagnostic figure showing
host/SN link lines on a skymap.
