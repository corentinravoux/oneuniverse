# Pillar 3 — Simulation Storage & Orchestration (Digital Twin Substrate)

**Date:** 2026-05-28 (rewritten 2026-06-01 — scope narrowed to
storage + orchestration of existing simulation outputs; mini-
simulation runs deferred indefinitely; partial-access + MPI/GPU
reads promoted to first-class design constraints).
**Scope:** Standalone package that **stores, indexes, and serves**
cosmological simulations of every kind (N-body, SPH, AMR, PM, full
GR, phase-space, constrained-realisation chains, differentiable
checkpoints). Its job is **partial access**: a user must be able to
read 1 Gpc/h sub-region of a 2 Gpc/h snapshot, one HEALPix shell of
a full-sky lightcone, or one progenitor branch of a merger tree —
without ever loading the parent dataset into memory.

This document is a **large-scope roadmap**, not a task plan.
Implementation lives in a future package (provisional name
`oneuniverse.sim` or sibling `onesim`); concrete phases are scoped
once architectural prototypes validate the partial-access model.

---

## 1. Mission

Be the data + orchestration layer for **every cosmological
simulation a downstream user might compare to a survey, condition
an IC sampler on, or use as a forward-model template**. Pillar 3
does **not** run simulations. It stores, indexes, and serves them
through a partial-access API: spatial cube, HEALPix tile, redshift
shell, halo branch, IC chain link, particle subsample.

The digital twin emerges later: once Pillar 3 reliably serves
arbitrary sub-regions of arbitrary simulations at high I/O
throughput, downstream packages (a future BORG-like sampler, a
JaxPM-based forward modeller, a constrained-realisation builder)
can plug in. They are **out of scope** for the current roadmap.

## 2. Boundary clarity

| In scope | Out of scope |
|---|---|
| Store + index existing simulation outputs in their native formats | Run simulations (no Gadget/AREPO/RAMSES/HACC integration) |
| Wrap native readers via thin adapters | Re-encode petabyte archives into a new format |
| Partial spatial / temporal / structural access | Whole-snapshot loads |
| MPI-collective + GPU-direct read paths | Single-threaded serial I/O only |
| Cross-snapshot / cross-representation indexes (halo ↔ particles, halo → progenitor chain, snapshot → lightcone shell) | Cross-survey cross-match (Pillar 1) |
| Cosmology + unit-frame declaration per dataset | Cosmology conversion / theory P(k) (Pillar 2) |
| Suite-level orchestration (AbacusSummit grid, CAMELS LHC, BORG posterior chain) | IC sampler / HMC / NUTS / VI implementation (future) |
| Constrained-realisation chain *storage* | Constrained-realisation chain *sampling* |
| Differentiable-forward-model checkpoint *storage* | The forward-model itself (lives in `pmwd`, `JaxPM`, `flip_simulation`, etc.) |
| Mini-simulation outputs *storage* (if ever produced upstream) | Mini-simulation *running* (deferred indefinitely) |

## 3. Design principles

Three load-bearing constraints, each derived from the user's
2026-06-01 guidance:

### 3.1 Minimal cross-pillar coupling

Pillar 3 is a **standalone** subpackage / package. It depends on
`numpy`, `pyarrow`, `h5py`, `healpy`, plus native-format readers
(`yt`, `abacusutils`, `swiftsimio`, `genericio`, …) — but **not**
on `oneuniverse.data` or `oneuniverse.combine`. Communication with
Pillar 1 / Pillar 2 happens through external file artefacts (OUF
parquet, MeasurementSet contracts), not Python imports.

Practical consequences:
- `oneuniverse.sim` (or `onesim`) can be split into its own
  repository without code surgery once mature.
- No `from oneuniverse.data import …` lines in Pillar 3 code.
- Cosmology / unit-frame declarations are duplicated rather than
  shared (small price for decoupling).
- Tests can run without installing Pillar 1.

### 3.2 Partial access first

Simulations are routinely TB–PB per snapshot. A whole-snapshot load
is **never** the default API; it is at most an explicit escape hatch
(`scan_full()` with a loud docstring warning).

Every public reader returns either:
- a **lazy view** (xarray / Dask / pyarrow Dataset / generator), or
- a **chunked iterator** (`for chunk in sim.iter_particles(cone=…, batch=10_000_000)`), or
- a **sub-region materialiser** (one HEALPix tile, one octree node,
  one halo's particle subsample).

API design principle: **make `read_everything` impossible to type by
accident.** Every reader takes a mandatory selector
(spatial / temporal / structural). Loading a whole 2 Gpc/h snapshot
must be a multi-step opt-in.

Selector taxonomy (uniform across backends):
- **Spatial**: cube `(x_min, x_max, y_min, y_max, z_min, z_max)`,
  cone (lon, lat, half-angle), sky-patch (lon_min, lon_max, lat_min,
  lat_max), HEALPix tile list, octree-node id.
- **Temporal**: redshift `z`, snapshot id, lightcone shell range
  `(z_min, z_max)`.
- **Structural**: halo id (returns member particles), tree branch
  (returns progenitor chain), IC chain link id (returns derived
  snapshot path), refinement-level range (AMR).
- **Field projection**: column subset
  (`fields=("Coordinates","Velocities")` skips the rest at I/O time).

### 3.3 MPI and GPU-direct reads

Pillar 3 must support both:

- **MPI-collective I/O.** Native parallel-HDF5 (TNG, MTNG, EAGLE,
  FLAMINGO), GenericIO (HACC), BigFile (FastPM), and AMReX plotfile
  (Nyx) all support `H5Pset_fapl_mpio` or equivalent. Pillar 3's
  reader API exposes an `mpi_comm=` kwarg; backends that support it
  return a per-rank-local view, those that don't fall back to
  rank-0 broadcast.
- **GPU-direct reads** (NVIDIA GPUDirect Storage / cuFile / kvikIO /
  cuDF for parquet). For backends that admit GPU staging (parquet
  via `kvikIO`, Zarr v3 + S3 via `cucim`), Pillar 3's reader API
  exposes a `device="cuda:0"` kwarg; reads land in GPU memory
  without a CPU bounce. Backends without GPU paths warn + fall back.

Both paths are **opt-in**, default is single-process CPU read.
Backends declare their capabilities via `BackendCapabilities`
dataclass; the reader API does not promise features the backend
cannot deliver.

### 3.4 Deferred (placeholder only — not in dev plan)

- **Mini-simulation runs.** Zoom-in hydrodynamics on regions of
  interest from the IC posterior. Architectural placeholder: Pillar 3
  can store mini-sim outputs (same as any other sim), but never
  invokes a simulation code.
- **IC sampling / forward modelling.** BORG-like HMC, JaxPM gradient
  pipelines, constrained-realisation builders. Architectural
  placeholder: Pillar 3 stores their inputs (cosmology), outputs
  (IC chain links, derived snapshots), and indexes, but does not
  implement the sampler.
- **Incremental update on new data.** Re-running forward models when
  a new Pillar-1 survey lands. Future, after sampler exists.

## 4. Architecture sketch

```
┌─────────────────────────────────────────────────────────────────┐
│  oneuniverse.sim  (standalone subpackage; minimal deps)         │
│                                                                  │
│  ┌─────────────────┐    ┌──────────────────┐                     │
│  │ Manifest layer  │    │ Index layer      │                     │
│  │  manifest.yaml  │    │  HEALPix tiles   │                     │
│  │  cosmology.yaml │    │  octree per snap │                     │
│  │  unit_frame.yaml│    │  KD-tree per snap│                     │
│  │  provenance.yaml│    │  halo→part join  │                     │
│  └────────┬────────┘    │  tree pointer DB │                     │
│           │             └─────────┬────────┘                     │
│           │                       │                              │
│  ┌────────▼───────────────────────▼────────┐                     │
│  │      Selector API (spatial / temporal / structural)            │
│  └────────┬───────────────────────┬────────┘                     │
│           │                       │                              │
│  ┌────────▼──────┐  ┌─────────────▼──────┐  ┌─────────────────┐  │
│  │ Backend: HDF5 │  │ Backend: ASDF/pack9│  │ Backend: native │  │
│  │ (Gadget/AREPO │  │ (AbacusSummit)     │  │ (RAMSES/yt/etc.)│  │
│  │  /SWIFT/TNG)  │  │                    │  │                 │  │
│  │ MPI ✓ GPU ✗   │  │ MPI ✓ GPU ✗        │  │ MPI ?  GPU ✗    │  │
│  └───────────────┘  └────────────────────┘  └─────────────────┘  │
│           │                       │                  │           │
│  ┌────────▼──────┐  ┌─────────────▼──────┐  ┌────────▼────────┐  │
│  │ Backend: Zarr │  │ Backend: parquet   │  │ Backend: FITS   │  │
│  │ (regular grid │  │ (halo / lightcone  │  │ (HEALPix shells)│  │
│  │  v3 + S3)     │  │  galaxy tables)    │  │                 │  │
│  │ MPI ✓ GPU ✓   │  │ MPI ✓ GPU ✓ (kvikIO│  │ MPI ✗ GPU ✗     │  │
│  └───────────────┘  └────────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.1 Manifest layer

`OUFSimManifest` dataclass — a manifest-of-manifests pointing at
native files + sidecar indexes (see `research/cosmological_simulation_landscape.md`
§6). Schema-versioned. Per-sim (or per-suite-member) directory
contains:
- `manifest.yaml` — contract; `oufsim_format_version` pinned.
- `cosmology.yaml` — Ω_m, σ_8, h, n_s, w_0, w_a, T_CMB.
- `unit_frame.yaml` — canonical unit-frame declaration
  (length, mass, velocity, time, h-factor, comoving/proper).
- `provenance.yaml` — code version, git hash, original file paths,
  ingest timestamp.
- Per-product subdirectories (`snapshots/`, `merger_tree/`,
  `lightcone/`, `ic_posterior/`, `checkpoints/`) each containing an
  `index.parquet` plus paths/symlinks to native files.

### 4.2 Index layer

For each snapshot / lightcone shell, OUF-Sim builds **partial-access
indexes** alongside the native files. Native files are read-only;
indexes are sidecar parquet / HDF5 / Zarr.

Per snapshot:
- **HEALPix tile index** at fixed NSIDE (suggest NSIDE=64 for
  snapshot-level → 49152 tiles → ~50k particles/tile at 2 Gpc/h
  6912³). Columns: `(tile_id, file_offset, n_particles, sha256)`.
- **Octree node index** for AMR (Enzo / RAMSES / FLASH / Nyx /
  AthenaPK) — per-level patch bounding boxes + native file pointers.
- **Halo → particle pointer** — `(halo_id, particle_offset, n_particles)`
  joining halo catalog to particle table.

Per merger tree:
- **Branch index** — for each halo, depth-first start/end indices
  into the tree's flat representation (`first_progenitor_id`,
  `last_progenitor_id`).

Per lightcone:
- **Shell index** — `(shell_id, z_min, z_max, NSIDE, ordering, path)`.
- **Per-pixel z-vs-shell lookup** — HEALPix-indexed parquet for
  stitching onion-shell snapshots.

Per IC posterior chain:
- **Chain manifest** — `(chain_link_id, log_posterior, log_likelihood,
  IC_path, derived_snapshot_paths)`.

### 4.3 Selector API

Public reader API. Every method takes a mandatory selector + an
optional MPI/GPU directive:

```python
sim.snapshot(z=0.5).iter_particles(
    cone=Cone(ra=120.0, dec=0.0, radius_deg=5.0),
    type="DM",
    fields=("Coordinates", "Velocities"),
    batch=10_000_000,
    mpi_comm=None,                  # opt-in MPI
    device="cpu",                   # or "cuda:0"
) -> Iterator[pa.Table]
```

```python
sim.snapshot(z=0.5).halos("CompaSO_L1").get_particles(
    halo_id=12345, fields=("Coordinates",),
) -> pa.Table
```

```python
sim.lightcone.shell(z=0.3).read(
    tiles=[42, 43, 44], fields=("kappa", "gamma1", "gamma2"),
) -> np.ndarray   # one HEALPix array per field, sub-tiles only
```

```python
sim.merger_tree.branch(start_halo_id=12345) -> pa.Table
```

```python
sim.ic_posterior.chain_link(step=1234).realised_snapshot
  -> SnapshotView   # lazy; further selectors needed before read
```

### 4.4 Backend capabilities

Each backend declares its capabilities up front:

```python
@dataclass(frozen=True)
class BackendCapabilities:
    name: str
    native_format: str          # "Gadget HDF5", "ASDF/pack9", …
    supports_mpi: bool
    supports_gpu_direct: bool
    supports_random_access: bool # KD-tree / Hilbert key range
    supports_streaming: bool     # chunked iterator
    requires_extra: tuple       # ("abacusutils",), ("genericio",), …
```

Reader API checks capabilities at call time; raises informatively
if asked for something the backend cannot do.

### 4.5 Indices vs caches

Pillar 3 distinguishes:
- **Indexes** (built once at ingest, stored alongside the OUF-Sim
  manifest, version-controlled with the manifest).
- **Caches** (built on first use, local to the user's environment,
  not shipped with the manifest).

Indexes are mandatory for partial access. Caches accelerate
repeated reads.

## 5. Roadmap (large strokes, no per-phase commitments yet)

Pillar 3 design depends on validating the manifest-of-manifests +
partial-access model. Three exploratory milestones, each
proof-of-concept:

### Milestone σ — Architectural prototype (no code yet)

- Stand up the design doc (this file + research landscape doc).
- Define `OUFSimManifest`, `BackendCapabilities`, `Selector` types
  on paper.
- Decide subpackage vs sibling repo (final answer pending; current
  bias: subpackage `oneuniverse.sim` for ease of co-development,
  but no imports from `oneuniverse.data`).

### Milestone τ — Partial-access proof-of-concept

- Single backend: **AbacusSummit ASDF/pack9 via `abacusutils`** (most
  varied representations, well-documented format, NERSC POSIX
  access).
- One sim, one snapshot. Build:
  - manifest + indexes (HEALPix tile, halo→particle pointer);
  - `Snapshot.iter_particles(cone=…, batch=…)` returning sub-region
    chunks at a measured throughput floor (target: > 1 GB/s on
    Perlmutter login node, single process);
  - MPI variant returning per-rank-local views;
  - GPU-direct variant via `kvikIO` parquet (after converting the
    Abacus particle subsample to an indexed parquet sidecar; do NOT
    re-encode the native ASDF).
- Validation: cone read + halo → particles query yield byte-
  identical results to `abacusutils` direct calls.

### Milestone υ — Multi-backend + suite orchestration

- Add second backend: **Gadget HDF5** (covers TNG, EAGLE, FLAMINGO,
  MTNG, FIRE, SIMBA — one reader for ~80% of the field).
- Suite-level manifest — AbacusSummit grid (97 cosmologies); CAMELS
  Latin-Hypercube (~1000 sims); query: "all (Ω_m, σ_8) pairs at
  this box size".
- Cross-snapshot index: merger-tree branch walker delegating to
  `ytree`.
- Document the BackendCapabilities matrix.

### Future placeholders (not in any current dev plan)

- Backends for AMR (RAMSES / Enzo / Nyx via `yt`), GenericIO
  (HACC), BORG IC chain, BigFile (FastPM), differentiable
  checkpoint (pmwd / JaxPM).
- Wave-1 ingest: FLAMINGO, MTNG, Buzzard, Flagship-2.
- Wave-2 ingest: Uchuu, MANTICORE, SIBELIUS, BORG, THESAN.
- Mini-simulation run integration. **Out of scope until further notice.**
- IC sampler integration (BORG-style HMC, JaxPM gradient flow).
  **Out of scope until further notice.**

## 6. Dependencies posture

**Hard requirements** (mandatory for any Pillar 3 import):
- numpy, pyarrow, h5py, healpy, pyyaml.

**Optional extras** (declared per backend, loaded lazily):
- `abacusutils` — AbacusSummit ASDF/pack9.
- `yt` — universal AMR + many particle formats.
- `swiftsimio` — SWIFT / EAGLE-XL / FLAMINGO.
- `illustris_python` — TNG / Illustris / EAGLE Subfind.
- `ytree` — universal merger trees.
- `halotools` — HOD wrappers.
- `genericio` — HACC.
- `bigfile` — FastPM / MP-Gadget.
- `mocpy` — multi-order MOC HEALPix (already a Pillar 1 dep).
- `mpi4py` — MPI-collective I/O.
- `kvikIO` / `cucim` — GPU-direct reads.
- `zarr` — modern chunked array store (cloud-friendly).

**No dependency** on `oneuniverse.data`, `oneuniverse.combine`, or
any Pillar-1 internals.

## 7. Open questions

- Subpackage vs sibling repo? Subpackage simpler for co-development;
  sibling enforces zero coupling. Decide after milestone τ.
- Index format: parquet vs Arrow IPC vs Zarr v3? Parquet good for
  small indexes (catalogs, pointers); Zarr v3 better for large
  sidecar arrays (full-snapshot HEALPix tile maps). Likely both.
- MPI: hard requirement at backend level or universal optional?
  Likely backend-by-backend.
- GPU paths: how much code do we write before NVIDIA cuFile is
  production-ready in `kvikIO` for HDF5? Currently parquet + Zarr
  are well-supported; HDF5 GPU-direct via NVIDIA-HDF5 plugin is
  alpha.
- Constrained-realisation chains: store the chain as an "ensemble"
  (one OUFSim per link) or a single OUFSim with `ic_posterior/`
  group + many samples? Tentative answer: single OUFSim with chain
  manifest, but expose iter-over-links for downstream consumers.

## 8. References

- [`../research/cosmological_simulation_landscape.md`](../research/cosmological_simulation_landscape.md)
  — codes, suites, representations, OUF-Sim format proposal.
- [`../research/survey_landscape_review.md`](../research/survey_landscape_review.md)
  — Pillar 1 survey landscape for cross-matching survey ↔ sim
  pairings.
- [`2026-05-28-pillar1-data-combine-measure.md`](2026-05-28-pillar1-data-combine-measure.md)
  — Pillar 1 (sister roadmap).
- [`2026-05-28-pillar2-external-interfaces.md`](2026-05-28-pillar2-external-interfaces.md)
  — Pillar 2 (`MeasurementSet` contract).
- [[oneuniverse-pillars]] — three-pillar architecture memory entry.

Implementation notes (to remember when work begins):
- **Whole-snapshot loads are an anti-pattern.** Every public reader
  takes a mandatory selector.
- **Re-encoding the petabytes is forbidden.** Wrap, never duplicate.
- **MPI + GPU paths are first-class.** Single-process reads must
  remain functional but must not block the parallel paths.
- **No `oneuniverse.data` imports.** Pillar 3 stands alone.
- **Mini-simulation runs are deferred indefinitely.** Architecture
  leaves room; dev plan does not include them.
