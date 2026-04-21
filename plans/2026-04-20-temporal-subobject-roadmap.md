# Temporal & Sub-object Extension Roadmap

**Started:** 2026-04-20 (post Pillar 1 stabilisation)
**Applies to:** `oneuniverse` package, post-Phase-6.

## Motivation

The Pillar-1 stabilisation (Phases 1–6) delivered a **static** database:
every row is a point in (ra, dec, z) with no time dimension, every
dataset is considered permanent, and the ONEUID / sub-object graph has
no notion of *when* it was true. Two real-world use-cases break that
assumption.

### Use-case 1 — observation time per row

Transient and time-domain data has a physical time axis that the schema
must carry explicitly:

- **Supernovae (Pantheon+, DES-SN, ZTF-BTS, LSST-DR1)** — each row is a
  discovery/peak epoch on MJD.
- **Tidal disruption events, optical alerts, GRB afterglows** — each row
  is an alert epoch.
- **Multi-epoch photometry (ZTF, LSST lightcurves, Kepler, TESS)** —
  one object has N measurements at N epochs; today there is no
  geometry to describe this shape.

### Use-case 2 — the database itself is dynamic

The database is not a frozen snapshot. Surveys publish data releases
(eBOSS DR14 → DR16; DESI DR1 → DR2; Pantheon → Pantheon+). Old releases
are superseded but not deleted — reproducibility demands that a paper
written against "the database as it was on 2026-01-15" remains
queryable. Likewise:

- An ONEUID index is **a function of which datasets were available when
  it was built**. A DR17 ONEUID index differs from a DR16 ONEUID index.
- A sub-object linkage between SN rows and host galaxies depends on
  which SN catalogue and which galaxy catalogue were available.
- Cross-survey combined measurements cached to disk inherit that
  vintage.

The database therefore has **two** orthogonal time axes, which
temporal-database literature calls *valid time* and *transaction time*
(bitemporal model):

- **Observation time (t_obs)** — physical time of the measurement, the
  one astronomers plot on a light-curve.
- **Transaction time (db_time)** — when the database knew this fact;
  the one reviewers want when they ask "what did you use for the
  paper?".

This roadmap adds both to `oneuniverse` as two phases.

## Scope boundary (non-negotiable)

`oneuniverse` is a **data/interface** layer. It stores, indexes, and
queries; it does not infer, fit, or model. The temporal and sub-object
work is about **storage, inventory, validity, and query** — surface
the information cleanly so downstream packages (`flip`,
forward-modelling code, analysis notebooks) can consume it.

Out of scope for every task in this roadmap:

- Any statistical inference beyond deterministic coordinate / epoch /
  tolerance matching.
- Light-curve fitting, transient classification, peak-brightness
  inference, K-corrections.
- Host-galaxy property modelling (stellar mass, SFR, metallicity,
  environment).
- Causal or dynamical modelling (peculiar-velocity evolution, Hubble
  diagram residuals).
- Any forward-model or simulation code.

## Fixed design decisions

These are locked in; came out of the 2026-04-20 design pass. Do not
re-debate without raising with the user first.

### Observation-time axis (per-row)

1. **Ride on existing geometries first.** Any POINT or SIGHTLINE
   dataset may carry an optional `t_obs` column (float64, MJD). The new
   `LIGHTCURVE` geometry is added for genuinely multi-epoch data where
   the natural shape is one row per (object, epoch).
2. **Time unit: MJD, float64, TDB-referenced by default.** Other time
   references (TAI, UTC, TT) are opt-in via
   `TemporalSpec.time_reference`. No survey converter should silently
   emit JD or BJD.
3. **Partition stats carry `t_min` / `t_max`.**
   `DatasetView.scan(t_range=…)` prunes partitions before opening
   Parquet, mirroring the existing ra/dec/z pruning.
4. **`LIGHTCURVE` geometry is a structural twin of `SIGHTLINE`.**
   `objects.parquet` (per-object metadata: object_id, ra, dec, z,
   n_epochs, mjd_min, mjd_max) + `part_*.parquet` (per-epoch rows keyed
   by object_id and mjd).
5. **Writer adds `t_min`/`t_max` stats automatically** when a temporal
   column is present; schema 2.0.x datasets continue to work unchanged
   (temporal block is optional).

### Transaction-time axis (database-wide)

1. **Every `Manifest` carries a `DatasetValidity` block** with
   `valid_from_utc`, `valid_to_utc` (None = current), `version` (free
   string), and `supersedes` (list of superseded dataset names).
2. **Versioning is expressed at the filesystem level** as sibling
   directories. A survey can ship multiple releases:
   ```
   {DB_ROOT}/spectroscopic/eboss_qso/v_dr16/oneuniverse/manifest.json
   {DB_ROOT}/spectroscopic/eboss_qso/v_dr17/oneuniverse/manifest.json
   ```
   The walker finds both, keys each by a distinct dataset name derived
   from the path (or by an explicit `survey_name` in the manifest).
3. **`OneuniverseDatabase.as_of(timestamp)`** returns a view of the
   database whose entries are the ones with
   `valid_from_utc <= timestamp < (valid_to_utc or +∞)`. The default
   database view is `as_of(now)`.
4. **`OneuniverseDatabase.versions_of_root(path)`** lists all recorded
   versions of a dataset under a given subpath, ordered by
   `valid_from_utc`, including superseded ones.
5. **ONEUID indices are bitemporal too.** The ONEUID manifest JSON
   carries its own `DatasetValidity` block. `db.load_oneuid(name,
   as_of=T)` selects the index that was valid at T. Building an index
   with the same name creates a new version and closes out the
   previous one (`valid_to_utc = now`).
6. **Sub-object links (Phase 8) are bitemporal too** — same mechanism,
   same manifest block.
7. **No migration of existing OUF 2.0 manifests.** A 2.0 manifest read
   by a 2.1 reader is implicitly assigned
   `valid_from_utc = created_utc`, `valid_to_utc = None`, `version =
   "1.0"`, `supersedes = []`. The next rewrite bumps it to 2.1.
8. **Manifest format bump: `2.0.0` → `2.1.0`; schema
   `2.0.0` → `2.1.0`.** The reader accepts `2.0.x` and `2.1.x`; an
   explicit `3.x` write in the future will break compatibility.

### Sub-object hierarchy (Phase 8)

1. **Sidecar tables, not per-survey manifest entries.** Sub-object
   links live at `{root}/_subobject/<name>.parquet` with a companion
   `<name>.manifest.json`, mirroring the ONEUID-index layout. Multiple
   link sets (e.g. `sne_in_hosts`, `galaxies_in_clusters`) coexist.
2. **Link currency is ONEUID, not per-survey row index.** Building
   links requires a covering ONEUID index. The link is stable under
   re-partitioning and database rebuilds.
3. **Children keep their own ONEUID.** A supernova is a distinct
   astrophysical object from its host galaxy; merging them into one
   ONEUID is physically wrong. The link records a *relation*, not an
   identity.
4. **Driven by a `SubobjectRules` dataclass** — symmetric in spirit to
   `CrossMatchRules`, different physics and tolerances. Frozen
   dataclass, content-hashed for audit,
   `parent_survey_type`/`child_survey_type` enforced.
5. **Many-to-many is natural.** A host can have multiple SNe;
   ambiguous SN→host matches are either rejected or exposed with a
   confidence score via an `accept_ambiguous` rule flag.
6. **Confidence scores persist.** Every link row carries `confidence`
   (float32), derived from the matching algorithm. Downstream filters.
7. **Bitemporal.** Sub-object-links manifests carry the same
   `DatasetValidity` block; old link sets remain queryable via
   `db.load_subobject_links(name, as_of=T)`.

## Phase index

| Phase | Scope | Plan | Status |
|-------|-------|------|--------|
| 7 | Temporal data model — observation-time axis (POINT/SIGHTLINE t_obs + LIGHTCURVE geometry) **and** transaction-time axis (bitemporal database snapshots + versioned ONEUID) | [`2026-04-20-phase7-temporal.md`](2026-04-20-phase7-temporal.md) | pending |
| 8 | Sub-object hierarchy (bitemporal named link tables) | [`2026-04-20-phase8-subobject.md`](2026-04-20-phase8-subobject.md) | pending |

Phase 7 is large but cohesive: observation-time and transaction-time
share the manifest-extension / partition-stats / `DatasetView`
machinery. Splitting them would require most of the extension
infrastructure to be built twice.

Phase 8 depends on Phase 7 only to the extent that its own sidecar
manifests re-use `DatasetValidity`. Conceptually independent.

## Memory & graph touchpoints consulted while writing this roadmap

- Memory records:
  - `project_oneuniverse_stabilisation.md` (phase status, design commitments)
  - `project_oneuniverse.md` (three-pillar vision, architecture)
  - `project_oneuniverse_scope.md` (data/interface boundary)
  - `project_digital_twin_vision.md` (longer-term SNe/transient use-case)
  - `feedback_visual_testing.md` (diagnostic figure requirement)
- Graphify communities (`graphify-out/GRAPH_REPORT.md`):
  - **Format Spec & Geometry** (`DataGeometry` as 92-edge god node)
  - **OUF Manifest Schema** (`Manifest`, `PartitionSpec`, `PartitionStats`, `PartitioningSpec`)
  - **DatasetView Query** (`_select_partitions`, `_range_expr`, `_spatial_expr`)
  - **ONEUID Index** (`OneuidIndex`, `OneuidQuery`, `build_oneuid_index`)
  - **Cross-Match Rules** (`CrossMatchRules` as structural reference)
  - **Database & Config Build** (`OneuniverseDatabase`, `DatasetEntry`, `_name_from_path`)

## Success criteria per phase

Phase 7:
- A SN-style loader emits a POINT dataset with `t_obs`, a
  `TemporalSpec` in its manifest, and per-partition `t_min`/`t_max`.
- `DatasetView.scan(t_range=(t0, t1))` returns only rows in the window
  without reading pruned partitions.
- A synthetic LIGHTCURVE dataset can be written end-to-end; `DatasetView`
  can return either the per-object table or the epoch rows with
  time-range pruning.
- Every `Manifest` read through the new path has a
  `DatasetValidity` block (default-filled for 2.0.x).
- `OneuniverseDatabase.as_of(timestamp)` returns the correct subset
  when the folder contains a `v1` and a `v2` of the same survey and
  one supersedes the other.
- `db.load_oneuid(name, as_of=T)` resolves to the correct ONEUID
  manifest.
- Full test suite green (≥ +35 new tests). Two diagnostic figures:
  (a) transient MJD window vs full distribution, (b) database contents
  at two snapshot times.

Phase 8:
- `SubobjectRules` declares matching policy between parent and child
  survey types with sky and redshift tolerances; its `hash()` is
  stable across Python sessions.
- `db.build_subobject_links(rules, parent_datasets, child_datasets,
  name=…)` writes a `_subobject/<name>.parquet` sidecar keyed on
  ONEUIDs and a bitemporal manifest.
- `db.load_subobject_links(name, as_of=T).children_of(parent_oneuid)`
  and `.parent_of(child_oneuid)` work.
- Ambiguous matches are rejected by default or surfaced with
  `accept_ambiguous=True`.
- Full test suite green (≥ +18 new tests). One diagnostic figure
  showing host/SN link lines on a skymap.
