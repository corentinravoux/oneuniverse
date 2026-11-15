# oneuniverse — External Review & Issue Register

**Date:** 2026-06-05 · **Reviewer pass:** full-package, external-reviewer style
· **Suite at review:** 780 passed / 2 skipped (+ this pass's additions).

This is an honest, critical review of the whole `oneuniverse` package (Pillars
1 + 3 in-package; Pillar 2 = the new `measure` layer). Severity: **C**ritical /
**H**igh / **M**edium / **L**ow / **O**bservation. Each item: *where · what ·
impact · recommendation*. Items marked ✅ were fixed during this review.

---

## Summary verdict

The package is **internally consistent, well-tested (783 tests, 0 TODO/FIXME),
and honest about its toy components**. The architecture (3-pillar, cosmology-free
Pillar 1, Universal DataProduct) is coherent and the discipline (Rule-1 import
guard, cosmology-free invariant) is real. The principal weaknesses are
**over-advertised survey loaders**, **no on-disk form for the `MeasurementSet`
handoff object**, and a **window inferred from data rather than the survey
mask**. None are architectural; all are addressable. One latent version bug was
found and fixed.

---

## Resolution status (post-review fixes, 2026-06-05)

| Item | Status | What was done |
|---|---|---|
| F1 OUF version drift | ✅ fixed | LIGHTCURVE writer uses canonical `FORMAT_VERSION` + regression test |
| F2 `summary()` not JSON-safe | ✅ fixed | tuple pair keys stringified + `json.dumps` test |
| H1 loader over-advertising | ✅ fixed | `SurveyConfig.status` (ready/planned); `list_surveys(status=)`, `survey_status()` |
| H2 no on-disk MeasurementSet | ✅ fixed | `to_dir`/`from_dir` (parquet+npy+JSON); round-trips all 3 subtypes |
| M1 window inferred from data | ✅ fixed | `window_from_mask(completeness)`; `footprint_from_positions` = stop-gap |
| M2 symbolic cosmology guard | ✅ fixed | `check_invariants` scans catalog columns for cosmology-derived names |
| M4 stale root CLAUDE.md | ✅ fixed | OUF 2.5 / 781 tests / measure built |
| L1 map_cross randoms | ✅ fixed | `build_map_cross(randoms="generate")` |
| L2 unused `attributes` | ✅ fixed | `build_cosmic_shear` records shape columns |
| **M3 real-data for other probes** | ⏳ open | only clustering validated on real data |
| L3 3×2pt metadata redundancy · L4 plan residue | ⏳ open | cosmetic |

---

## Fixed during this review

- **✅ F1 — OUF format-version drift (bug).**
  `oneuniverse/data/_converter_lightcurve.py` hard-coded
  `oneuniverse_format_version="2.1.0"` / `schema_version="2.1.0"` while
  POINT/SIGHTLINE use the canonical `FORMAT_VERSION=SCHEMA_VERSION="2.5.0"`.
  Every LIGHTCURVE dataset was mislabelled, and **no test caught it** (the
  existing `test_format_version_is_2_5_0` only checks the constants, not the
  writer output). *Fixed*: use the constants + a regression test on the written
  manifest (`test/test_lightcurve_version_fix.py`).

- **✅ F2 — `summary()` not JSON-serialisable.** `MeasurementSet.summary()`
  returned `pair_statistics` with **tuple keys**, so `json.dumps(ms.summary())`
  raised. *Fixed*: stringify pair keys (`"a×b"`).

---

## High

- **H1 — Survey registry over-advertises capability.**
  *Where:* `oneuniverse/data/surveys/*`. *What:* `list_surveys()` returns **10**
  surveys, but **7 of 10 loaders `raise NotImplementedError`** in `load()`
  (`sixdfgs, pantheonplus, des_dr2, desi_bgs, desi_pv, sdss_mgs, cosmicflows4`).
  Only `eboss_qso`, `desi_qso`, `dummy` actually load. *Impact:* a user who
  calls `load_catalog("desi_bgs")` gets a crash, not data; the registry promises
  coverage it does not have. *Recommendation:* add a `status ∈ {ready, planned}`
  field to the loader registry; `list_surveys()` should mark/segregate planned
  ones; docs must say "3 functional loaders (eBOSS, DESI, dummy); 7 scaffolds."

- **H2 — `MeasurementSet` has no on-disk form.**
  *Where:* `oneuniverse/measure/measurement_set.py`. *What:* the object is billed
  as *the P1→P2 handoff*, but there is no `save`/`load` — it lives only in memory.
  *Impact:* the separate estimator package cannot actually receive a
  `MeasurementSet` across a process boundary; the handoff story is incomplete.
  *Recommendation:* `to_dir(path)` / `from_dir(path)` — catalog/randoms →
  parquet, region_map/maps → npy, spec/metadata/provenance/links → JSON sidecar
  (cosmology-free by construction). This is the natural next feature.

---

## Medium

- **M1 — Window is inferred from data, not the survey mask.**
  *Where:* `measure/window.py::footprint_from_positions`. *What:* the angular
  window is "pixels that contain ≥1 object" — i.e. the footprint is defined *by
  the data*. This is circular: the true selection mask (mangle / HEALPix
  completeness / veto) is an input, not an output, of the objects. *Impact:* for
  real clustering the window is biased (empty-but-observed regions are dropped;
  edge effects mis-modelled). The slots exist (`Window.systematics`,
  `polygon_path`) but no builder populates them. *Recommendation:* a
  `window_from_mask(healpix_completeness)` / `from_mangle(path)` ingest; treat
  the data-inferred footprint as an explicit stop-gap in docs.

- **M2 — Cosmology-free guard is symbolic.** *Where:*
  `measurement_set.py::check_invariants`. *What:* it checks
  `hasattr(metadata, "cosmology")` (impossible on the frozen dataclass) + a
  test-only flag. It does **not** detect cosmology leaking into *catalog
  columns* (e.g. a `comoving_distance` column). *Impact:* the load-bearing
  "no cosmology in the output" rule is enforced only structurally, not on
  contents. *Recommendation:* a forbidden-column scan
  (`comoving_distance`, `r_comoving`, `dist_mpc_h`, …) or a documented column
  allowlist.

- **M3 — Real-data validation is clustering-only.** *Where:*
  `test/test_measure_real_desi_eboss.py`. *What:* only `build_galaxy_clustering`
  runs on real data (eBOSS/DESI QSO). WL / PV / SN / Lyα / map builders are
  **synthetic-only**. *Impact:* the other five builders are structurally tested
  but never met a real catalog. *Recommendation:* add real DESI BGS/LRG
  clustering (with real weights), a real shear catalog through
  `build_cosmic_shear`, real Lyα δ through `build_lya`.

- **M4 — Root workspace `CLAUDE.md` stale.** *Where:*
  `/home/ravoux/Documents/Python/CLAUDE.md`. *What:* says "OUF 2.1", "365/365
  tests", "future `measure/` subpackage". Reality: OUF 2.5, ~783 tests, measure
  built. *Recommendation:* refresh the three facts (done in this pass for the
  package-level docs; the root file is workspace-wide).

---

## Low

- **L1 — `build_map_cross` drops galaxy randoms.** Builds the galaxy side with
  `randoms="none"`; a real galaxy×κ still needs the galaxy mask/randoms. Minor
  (cross-spectra often use the field's mask), but inconsistent with the
  clustering path. *Recommendation:* allow `randoms="generate"` here too.

- **L2 — `PointSet.attributes` declared-not-used.** The role→columns map is set
  only in a generality test, never by a builder. *Recommendation:* populate it
  in `build_cosmic_shear` (shapes) / `build_*` (distances) or drop it.

- **L3 — 3×2pt metadata redundancy.** `build_3x2pt` uses the lens product's
  `ProductMetadata` for the set while the source product carries an equal-NSIDE
  copy. Harmless (invariant holds), but two sources of truth.

- **L4 — Doc/plan residue.** `plans/2026-05-28-pillar2-external-interfaces.md`
  is superseded (banner present, OK); `plans/2026-06-02-twin-coupling-roadmap.md`
  still references `onemeasure`. README was 741 lines (rewritten this pass).

- **L5 — Notebooks referenced removed/old APIs.** Replaced from scratch this pass.

---

## Observations (design honesty — not defects)

- **O1 — The measure layer computes no estimator.** It *builds and validates*
  the `MeasurementSet` (data + randoms + n(z) + window + weights + regions); it
  does **not** compute P(k)/ξ/C_ℓ. That is by design (estimator-side adapters
  are a separate package). Stated prominently in the new README.

- **O2 — Pillar 3 is dummy/toy end-to-end.** Linear sim + fast-PM + TreePM-split
  resim + Wiener twin — no real N-body or Bayesian inference. The storage/IO/
  orchestration substrate is real; the physics is a stand-in. Documented.

- **O3 — n(z)/window provenance.** Clustering n(z) = weighted data histogram
  (fine for spec-z); tomographic n(z) = photo-z kernel stack. Provenance method
  is recorded (`Nz.method`), which is correct practice.

---

## Strengths (for balance)

- 783 tests, 0 TODO/FIXME; Rule-1 (`simulation` ⊥ `data/combine`) import guard
  green; cosmology-free discipline enforced.
- **Real eBOSS DR16Q + DESI DR1 QSO** clustering validated end-to-end (genuine
  NGC+SGC footprint, real n(z)).
- **Generality coverage** across 12 probe classes (one container expresses
  clustering, WL, clusters, strong-lens time-delay, radio z-absent, SN, LIM, GW
  siren, Lyα+DLA, …).
- Multi-backend OUF-Sim storage with **index-only wrap-in-place** (≈14% of
  re-encode); TreePM-split resimulation beats the buffered baseline.

---

## Prioritised fix order

1. H1 loader registry `status` (stop crashing on advertised surveys).
2. H2 `MeasurementSet` on-disk form (complete the handoff).
3. M1 window-from-mask ingest (correctness for real clustering).
4. M3 real-data validation of the other five builders.
5. M2 stronger cosmology-free guard; L1–L4 cleanups.
