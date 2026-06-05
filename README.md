# oneuniverse

**One sky, every survey, one queryable Universe.** `oneuniverse` ingests and
standardises astronomical survey catalogs, combines them, builds analysis-ready
**measurements** for cosmology estimators, and provides a storage + simulation
substrate for a constrained digital twin of the cosmic web.

Pure-Python (numpy / pandas / pyarrow / healpy). Cosmology-free where it counts:
H₀ / Ωₘ / distance models never enter the data — they enter only at the
estimator/inference call.

---

## Three pillars

| Pillar | What it does | Package surface | Cosmology |
|---|---|---|---|
| **1 — Data** | ingest → standardise (OUF 2.5 parquet, HEALPix-partitioned) → cross-match (ONEUID) → weight → sub-object links | `oneuniverse.data`, `oneuniverse.combine` | **no** |
| **2 — Measure** | build the **MeasurementSet** (the general P1→P2 output) for every probe | `oneuniverse.measure` | **no** (enters at the estimator) |
| **3 — Simulation** | OUF-Sim storage substrate + fast-PM + resimulation + data↔sim twin | `oneuniverse.simulation`, `oneuniverse.twin` | in IC/forward model |

Estimators/likelihoods (P(k), ξ, C_ℓ, f σ₈ — `flip`, `pycorr`, `picca`, …) are
**external**, downstream tools that consume the MeasurementSet. The converters
to those tools are a **separate package** (not in this repo).

---

## Install

```bash
pip install -e ".[dev]"     # editable + test deps
pytest -q                   # 783 tests (~5 min; real-data tests auto-skip)
```

---

## Pillar 1 — data

```python
from oneuniverse.data import load_catalog, DatasetView
df = load_catalog("eboss_qso")                       # standardised DataFrame
view = DatasetView.from_path(survey_path)             # partial-access OUF reader
df = view.read(z_range=(0.8, 2.2), cone=cone)         # only the partitions you need
```

- **OUF 2.5** on disk: `manifest.json` + HEALPix-NSIDE32-NEST parquet.
- CORE columns: `ra, dec, z, z_type, z_err, galaxy_id, survey_id, _healpix32`.
- Geometries: `POINT`, `SIGHTLINE` (Lyα δ), `HEALPIX`/`GW_SKYMAP`, `CUBE`,
  `LIGHTCURVE`.
- ONEUID bitemporal cross-match + sub-object links; weights in
  `oneuniverse.combine` (FKP, completeness, systematics, shear, PIP bitwise).

**Loader status (honest):** `eboss_qso`, `desi_qso`, `dummy` load real data;
the other 7 registered loaders are scaffolds (`NotImplementedError`) pending the
survey files + a round-trip test. See [REVIEW.md](REVIEW.md) H1.

## Pillar 2 — measure (the P1→P2 output)

A **MeasurementSet** is the cosmology-free, analysis-ready handoff object. One
**Universal DataProduct** (3 geometry subtypes) carries every atom a probe
needs:

```python
from oneuniverse.measure import build_galaxy_clustering
ms = build_galaxy_clustering(view, z_range=(0.8, 2.2),
        weights=[...], nz_edges=..., randoms="generate")   # or randoms=<view>
ms.summary()        # JSON-safe description of products + atoms (no cosmology)
```

| Builder | Subtype | Probe |
|---|---|---|
| `build_galaxy_clustering` | PointSet | 3D clustering / RSD / BAO |
| `build_cosmic_shear`, `build_3x2pt` | PointSet | weak lensing (shapes + photo-z kernel + tomographic n(z)) |
| `build_peculiar_velocity`, `build_sn_hubble` | PointSet | PV, SN (distance atoms + covariance handle) |
| `build_lya` | Sightline | Lyα forest P₁D/P₃D |
| `build_map_cross` | FieldMap | galaxy × CMBκ / tSZ / HI |

The container also expresses probes that have no builder yet (clusters,
strong-lens time delays, radio z-absent tracers, GW sirens, line-intensity
mapping) via optional atom slots — proven by a 12-class coverage test. **It
builds and validates the MeasurementSet; it does not compute the estimator.**

**Validated on real data:** `build_galaxy_clustering` runs end-to-end on real
eBOSS DR16Q + DESI DR1 QSO (genuine footprint + n(z)) —
`test/test_measure_real_desi_eboss.py`.

## Pillar 3 — simulation + twin

```python
from oneuniverse.simulation.oufsim import write_oufsim_store, SimStore
store = write_oufsim_store(native_dir, out, sim_name="run",
                           field_projection="reference")     # wrap-in-place
SimStore(store).read_box("snapshots", z=0.0, cube=cube)      # partial access
```

- **OUF-Sim** storage substrate: manifest + parquet/memmap tiles + sidecar
  index; **multi-backend** (adapter registry), **index-only wrap-in-place**
  (≈14% of re-encode), MPI/GPU read hooks, budget-bounded streaming.
- **Resimulation:** fast-PM + **TreePM-split** coupling (beats the buffered
  baseline at every buffer).
- **Twin** (`oneuniverse.twin`): mock-challenge → Wiener reconstruction →
  data-driven resimulation; field-validation estimators (r(k), transfer,
  stochasticity).

**Honest scope:** Pillar 3 is a **dummy/toy** end-to-end (linear sim + fast-PM +
Wiener) — the storage/IO/orchestration substrate is real, the physics is a
stand-in for real N-body / Bayesian inference. See [REVIEW.md](REVIEW.md) O2.

---

## Status

- **783 tests** green, 0 TODO/FIXME, cosmology-free + Rule-1 import guards.
- Notebooks: `notebooks/` (executed with embedded plots, built by the
  `_build_*` generators). Diagnostic figures in `test/test_output/`.
- Honest known issues + review: **[REVIEW.md](REVIEW.md)**.
- Roadmaps + design: **[plans/README.md](plans/README.md)**,
  `plans/2026-06-05-pillar2-definition.md`,
  `research/2026-06-05-p1-to-p2-measurement-requirements.md`.

## Layout

```
oneuniverse/
  data/         Pillar 1 — schema, manifest, converter, DatasetView, ONEUID, loaders
  combine/      Pillar 1 — weights + WeightedCatalog
  measure/      Pillar 2 — DataProduct + MeasurementSet + per-probe builders
  simulation/   Pillar 3 — OUF-Sim store, fast-PM, resimulation
  twin/         Pillar 3 — data↔sim coupling
test/  plans/  research/  notebooks/  scripts/  docs/
```

Author: Corentin Ravoux. Companion to `flip` (arXiv:2501.16852).
