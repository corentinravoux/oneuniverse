# Phase S1 — OUF-Sim Architecture Proposal

**Date:** 2026-06-01
**Pillar:** 3 (simulation storage + orchestration; digital-twin substrate).
**Phase type:** Architecture proposal. **No code in this phase** —
this document is the deliverable. It defines the format, the
database, the converter model, the partial-access view layer, and
the orchestration layer that later phases implement.

**Status:** First Pillar-3 phase. Supersedes the earlier exploratory
"Milestone σ" in
[`2026-05-28-pillar3-simulation-digital-twin.md`](2026-05-28-pillar3-simulation-digital-twin.md).

**Constraints (pinned, non-negotiable).** From
[[pillar3-partial-access-and-minimal-deps]]:
1. **Minimal cross-pillar coupling** — standalone package; no
   imports from `oneuniverse.data` / `oneuniverse.combine`.
2. **Partial access is the load-bearing API** — sims are TB–PB;
   whole-sim loads never the default.
3. **MPI-collective + GPU-direct reads first-class.**
4. **Mini-simulation runs deferred indefinitely** — Pillar 3 stores +
   indexes + selects regions; it does **not** run simulations.

---

## 1. Mission of this phase

Produce the architecture for **OUF-Sim**: a database of cosmological
simulation products, structured so that

- any simulation's **output** is stored in an OUF-Sim format with
  partial access (twin of the OUF data format in Pillar 1);
- the OUF-Sim **database** is queryable with easy access (twin of
  the OUF survey database);
- the database is the **basis for orchestration** — selecting which
  regions to (re-)simulate and emitting simulation *requests*
  (without running anything);
- both the **input** (IC, config, cosmology) and the **output**
  (snapshots, halos, trees, lightcones) of a simulation can be
  **converted** into one or more OUF-Sim databases, with
  user-chosen projections (particle-centric, halo-centric,
  lightcone-centric, field-centric), because there are nearly
  infinite ways to simulate and to represent a simulation.

The digital twin is the eventual payoff: a queryable OUF-Sim
database of constrained realisations + region-selection
orchestration is the substrate a future sampler / forward modeller
plugs into. That sampler is **not** built here.

## 2. Three-layer architecture

OUF-Sim mirrors the OUF data stack but for simulations:

| Layer | OUF data (Pillar 1) | OUF-Sim (Pillar 3) |
|---|---|---|
| **Format** | OUF 2.x parquet + `Manifest` | OUF-Sim manifest-of-manifests + native-file wrappers + sidecar indexes |
| **Database** | survey database (`Database`, ONEUID, sub-object) | `SimDatabase` (sim index, cosmology grid, region index, lineage links) |
| **View / access** | `DatasetView` (cone / sky-patch / pushdown) | `SimDatasetView` (cube / cone / HEALPix tile / octree node / halo / tree branch; MPI + GPU) |
| **Converter** | survey loaders → OUF | sim converters (input + output) → OUF-Sim |
| **Orchestration** | weights + combine + (Pillar 2) MeasurementSet | region selection → simulation request |

The symmetry is deliberate: a user who knows the OUF data workflow
(load → convert → view → combine) recognises the OUF-Sim workflow
(ingest → convert → partial-view → select-region → request).

```
┌──────────────────────────────────────────────────────────────────┐
│  oneuniverse.sim  (standalone; minimal hard deps)                  │
│                                                                    │
│  ┌────────────┐  ingest   ┌──────────────────┐                     │
│  │ INPUT      │──────────▶│  OUF-Sim format  │                     │
│  │ converters │           │  (one sim)       │                     │
│  │ (IC/config)│           │  manifest +      │                     │
│  └────────────┘           │  native wrappers │                     │
│  ┌────────────┐  ingest   │  + sidecar       │                     │
│  │ OUTPUT     │──────────▶│  indexes         │                     │
│  │ converters │           └────────┬─────────┘                     │
│  │ (snap/halo/│                    │ register                      │
│  │  tree/lc)  │                    ▼                               │
│  └────────────┘           ┌──────────────────┐                     │
│                           │  SimDatabase     │  query              │
│                           │  - sim index     │◀──────────┐         │
│                           │  - cosmology grid│           │         │
│                           │  - region index  │           │         │
│                           │  - lineage links │           │         │
│                           └────────┬─────────┘           │         │
│                                    │ open                │         │
│                                    ▼                      │         │
│                           ┌──────────────────┐           │         │
│                           │ SimDatasetView   │           │         │
│                           │ (partial access; │           │         │
│                           │  MPI + GPU)      │           │         │
│                           └────────┬─────────┘           │         │
│                                    │ select region       │         │
│                                    ▼                      │         │
│                           ┌──────────────────┐           │         │
│                           │ Orchestration    │───────────┘         │
│                           │ region → request │  (request feeds      │
│                           └──────────────────┘   external sim;      │
│                                                   output re-ingests)│
└──────────────────────────────────────────────────────────────────┘
```

The dashed loop (request → external sim → re-ingest output) is the
incremental-zoom mechanism. Pillar 3 closes everything *except* the
"external sim" box, which stays out of scope.

## 3. OUF-Sim format (one simulation)

Refines the §6 sketch in
[`../research/cosmological_simulation_landscape.md`](../research/cosmological_simulation_landscape.md).

### 3.1 Principle

**Manifest of manifests.** OUF-Sim never re-encodes native petabytes.
It stores a manifest pointing at native files + sidecar partial-access
indexes + provenance + unit-frame + cosmology. Native readers stay
authoritative.

### 3.2 On-disk layout

```
oufsim_<sim_name>/
├── manifest.yaml              # contract; oufsim_format_version pinned
├── cosmology.yaml             # Ω_m, σ_8, h, n_s, w_0, w_a, T_CMB, …
├── unit_frame.yaml            # length/mass/velocity/time, h-factor,
│                              # comoving|proper, frame, endianness
├── provenance.yaml            # code, git hash, original paths, ingest ts
├── input/                     # the simulation's INPUT side
│   ├── ic_index.parquet       # (ic_id, kind, native_path, sha256)
│   ├── config_native/         # parameter files, makefile flags
│   └── ic_native/             # IC field / white-noise / transfer fn
├── output/                    # the simulation's OUTPUT side
│   ├── snapshots/
│   │   ├── index.parquet      # (snap_id, z, a, native_path, format)
│   │   └── snap_NNN/
│   │       ├── particles_native/   # path/symlink to native files
│   │       ├── halos_native/
│   │       └── ouf_index/          # SIDECAR partial-access indexes
│   │           ├── healpix_tiles.parquet
│   │           ├── octree_nodes.parquet      # AMR only
│   │           └── halo_particle_ptr.parquet
│   ├── merger_tree/
│   │   ├── tree_native/
│   │   └── branch_index.parquet
│   ├── lightcone/
│   │   ├── shells.parquet
│   │   ├── healpix_shells/             # FITS / Zarr
│   │   └── pixel_shell_lookup.parquet
│   ├── fields/                # regular Cartesian grids (PM δ, κ cubes,
│   │   └── *.zarr             #   GR tensor cubes) — Zarr v3 + sharding
│   ├── phase_space/           # tessellation / sheet (ColDICE, GAMER ψ)
│   │   ├── index.parquet      # (patch_id, bbox6d, native_path, n_simplex)
│   │   └── native/            # simplex vertices + connectivity, or ψ-AMR
│   ├── gr_fields/             # full-GR 4-tensor on (3+1) mesh
│   │   ├── index.parquet      # (slice_id, t, gauge, field, native_path)
│   │   └── native/            # Carpet/Chombo HDF5 per time-slice/level
│   ├── checkpoints/           # differentiable forward-model state
│   │   ├── index.parquet      # (ckpt_id, step, a, native_path, framework)
│   │   └── native/            # orbax (pmwd/JaxPM) / BigFile (FastPM)
│   └── ic_posterior/          # constrained-realisation chain (optional)
│       ├── chain_manifest.parquet
│       └── samples_native/
└── regions/                   # region catalog (drives orchestration)
    └── regions.parquet        # (region_id, kind, bbox/cone, z, refs)
```

`input/` + `output/` split: an OUF-Sim record captures both sides of
a simulation. Input drives reproducibility + re-launch; output is the
science product. The `output/` subdirectories cover **all eleven
storage primitives** from the research landscape (Section 3.5);
`snapshots/` carries both DM-only and hydro particle tables, `fields/`
carries regular Cartesian grids, `phase_space/` / `gr_fields/` /
`checkpoints/` / `ic_posterior/` carry the four non-particle, non-grid
representations. Any subdirectory absent ⇒ that product not present
for this sim (recorded in `manifest.products`).

### 3.3 Manifest contract

`OUFSimManifest` (dataclass; YAML-serialised):

```python
@dataclass(frozen=True)
class OUFSimManifest:
    oufsim_format_version: str          # pinned; reject incompatible
    sim_name: str
    sim_kind: str                       # "nbody" | "sph" | "amr" | "pm"
                                        # | "gr" | "phase_space"
                                        # | "constrained" | "differentiable"
    code: str                           # "Gadget-4" | "AREPO" | "ABACUS" …
    code_version: Optional[str]
    layout_schema: str                  # one of the 5 hierarchy patterns
    backends: Tuple[str, ...]           # native formats present
    has_input: bool
    has_output: bool
    products: Tuple[str, ...]           # "snapshots","halos","tree",
                                        # "lightcone","fields",
                                        # "phase_space","gr_fields",
                                        # "checkpoints","ic_posterior"
    n_snapshots: int
    redshifts: Tuple[float, ...]
    box_size: Optional[float]           # in unit_frame length units
    n_particles: Optional[int]
    cosmology_ref: str                  # → cosmology.yaml
    unit_frame_ref: str                 # → unit_frame.yaml
    provenance_ref: str
```

Pinned `oufsim_format_version`; reader rejects incompatible majors
(same discipline as OUF data `Manifest`).

### 3.4 Sidecar indexes (the partial-access enabler)

Built once at ingest, stored under `ouf_index/`, version-controlled
with the manifest. **Mandatory** — without them there is no partial
access.

- **HEALPix tile index** — `(nside, tile_id, file_offset, n_rows, sha256)`.
  Maps a sky tile (or, for 3D, an Nside-projected angular tile) to a
  byte range in the native file. Default NSIDE configurable per
  snapshot.
- **Octree node index** (AMR) — `(level, node_id, bbox, native_path,
  refinement_flag)`.
- **Halo → particle pointer** — `(halo_id, particle_offset, n_particles,
  native_path)`. Join key between halo catalog + particle table.
- **Merger-tree branch index** — `(halo_id, first_prog_id, last_prog_id,
  depth_first_start, depth_first_end)`.
- **Lightcone shell + pixel lookup** — `(shell_id, z_min, z_max, nside,
  ordering, path)` plus `(pixel_id, shell_id, z)` for onion-shell
  stitching.
- **Field-grid chunk index** — `(field, chunk_id, bbox, zarr_key)` for
  regular Cartesian grids + GR tensor cubes (Zarr-native chunking +
  this manifest = sub-region grid reads).
- **Phase-space patch index** — `(patch_id, bbox6d, native_path,
  n_simplex)` for ColDICE tessellation; bbox6d is the 6-D
  (x, v) bounding box so a 3-D spatial cone maps to candidate patches.
- **GR slice index** — `(slice_id, t, gauge, refinement_level, field,
  native_path, bbox)` for Carpet/Chombo per-time-slice tensor output.
- **Checkpoint index** — `(ckpt_id, step, a, framework, native_path)`
  for differentiable forward-model state (orbax / BigFile); no
  spatial sub-index (checkpoints reload whole-state by design).

Sidecar indexes are parquet (small, queryable) or Zarr (large
per-pixel / per-chunk maps). Native files are **read-only**. A given
backend builds only the indexes its products need; the converter
declares which (Section 5).

### 3.5 Coverage of all simulation shapes

The format must represent **every** storage primitive + code class
from [`../research/cosmological_simulation_landscape.md`](../research/cosmological_simulation_landscape.md)
(§4 representations, §2 codes). Audit:

| Research primitive (§4) | OUF-Sim product | Partial-access index | Primary native readers |
|---|---|---|---|
| 4.1 Particle table (DM-only) | `snapshots/` | HEALPix tile + KD-tree | Gadget HDF5, ASDF/pack9, GenericIO, BigFile |
| 4.2 Particle table (hydro/SPH/Voronoi) | `snapshots/` (extra fields) | HEALPix tile + KD-tree + halo-ptr | Gadget HDF5 (swiftsimio / illustris_python) |
| 4.3 AMR hierarchical mesh | `snapshots/` (AMR) | Octree node | RAMSES Fortran, AMReX, Enzo HDF5 (via yt) |
| 4.4 Regular Cartesian grid | `fields/` (Zarr) | Field-grid chunk | Zarr, NumPy, HDF5 |
| 4.5 Halo catalog | `snapshots/halos_native/` | Halo→particle pointer | ROCKSTAR, Subfind, CompaSO, AHF, VELOCIraptor |
| 4.6 Merger-tree graph | `merger_tree/` | Branch index | SubLink, Consistent Trees, HBT+, LHaloTree (via ytree) |
| 4.7 HEALPix lightcone shell | `lightcone/healpix_shells/` | Shell + pixel lookup | FITS HEALPix, Zarr |
| 4.8 Galaxy/halo lightcone catalog | `lightcone/` (parquet) | Shell + HEALPix tile | parquet (HEALPix-partitioned), FITS |
| 4.9 Phase-space sheet | `phase_space/` | Phase-space patch (bbox6d) | ColDICE binary, GAMER HDF5 |
| 4.10 Full-GR 4-tensor on mesh | `gr_fields/` | GR slice (t, gauge, level) | Carpet/Chombo HDF5, gevolution HDF5 |
| 4.11 IC posterior chain | `ic_posterior/` | Chain manifest | BORG HDF5, MANTICORE HDF5 |
| 4.12 Differentiable checkpoint | `checkpoints/` | Checkpoint index | orbax (pmwd/JaxPM), BigFile (FastPM) |

| Research code class (§2) | `sim_kind` | Covered by |
|---|---|---|
| 2.1 Pure N-body (tree/TreePM/FMM/P3M) | `nbody` | snapshots + halos + tree + lightcone |
| 2.2 N-body + SPH | `sph` | snapshots (hydro fields) + halos + tree |
| 2.3 Moving-mesh (AREPO) | `sph` (Voronoi flag) | snapshots (cell-as-particle) + halos + tree |
| 2.4 AMR hydro | `amr` | snapshots (octree) + halos |
| 2.5 PM / fast / forward-model | `pm` | snapshots + fields + (BORG → ic_posterior) |
| 2.6 Full GR | `gr` | gr_fields + snapshots (particles) |
| 2.7 Hybrid / radiative-transfer | `sph` / `amr` (rt flag) | snapshots + fields (photon bands) |
| 2.8 Constrained / forward-model | `constrained` | ic_posterior + snapshots + lightcone |
| 2.9 Phase-space / Vlasov / ψDM | `phase_space` | phase_space + fields (ψ grid) |
| 2.10 Emulators / surrogate suites | `nbody`/`sph` (suite bundle) | snapshots + fields + P(k) tables (as fields) |
| 2.11 Differentiable / inverse | `differentiable` | checkpoints + fields |

**Result: every primitive + every code class has a product + index +
reader.** No simulation shape in the research landscape falls outside
the format. The four non-particle / non-grid representations (phase-
space, GR tensor, IC posterior chain, differentiable checkpoint) get
dedicated `output/` subdirectories + index types rather than being
forced into the particle model.

## 4. SimDatabase

Twin of the OUF survey `Database`. A collection of OUF-Sim records
with an index + cross-record links.

### 4.1 Responsibilities

- **Sim index** — `(sim_name, sim_kind, code, box_size, n_particles,
  cosmology_id, layout_schema, path)`. Query: "all sims at this box +
  resolution"; "all sims of cosmology c000".
- **Cosmology grid** — `(cosmology_id, Ω_m, σ_8, h, n_s, w_0, w_a, …)`.
  Suite-level orchestration (AbacusSummit 97 cosmologies; CAMELS
  Latin-Hypercube). Query: "all (Ω_m, σ_8) pairs run at box X".
- **Region index** — aggregate of every sim's `regions.parquet`:
  `(region_id, sim_name, kind, bbox/cone, z, mass, refs)`. The
  substrate for region selection (Section 7).
- **Lineage links** — directed edges between OUF-Sim records:
  `parent_sim → child_sim` with a relation
  (`zoom_in`, `resimulate_higher_res`, `posterior_sample`,
  `forward_model_step`, `converted_projection`). This is how the
  database tracks the incremental-zoom history and the
  multiple-projection conversions (Section 8). Structurally analogous
  to Pillar 1 sub-object link sidecars + ONEUID lineage.

### 4.2 Bitemporal + versioned (reuse Pillar 1 lessons)

OUF-Sim records and their indexes should be **bitemporal** (valid-from
/ valid-to + ingest time) and **versioned** (rebuild archives prior
versions), mirroring the Pillar 1 ONEUID + sub-object design. A
constrained-realisation chain that grows as new survey data lands is
a versioned, append-only lineage — the bitemporal model fits exactly.

### 4.3 Easy access

The "database with easy access" requirement means:
- A single entry point `SimDatabase(root)` that lazily indexes
  child OUF-Sim records.
- Query by cosmology, box, resolution, code, region, lineage.
- `db.sim(name)` → OUF-Sim handle; `db.open(name).snapshot(z=…)`
  → `SimDatasetView`.
- No whole-DB scan on open; the index is parquet, read on demand.

## 5. Converters — the modularity core

Twin of survey loaders, but the converter layer is the **single most
important extensibility surface** in Pillar 3. Adding a new simulation
code must be straightforward: drop in one converter module, register
it, done. Everything else (database, view, orchestration) is
code-agnostic and never needs touching.

Design goal: **a new-code author writes only the format-specific glue;
all index-building, manifest-writing, lineage, partial-access wiring
is provided by the framework.**

### 5.1 Layered converter design (separation of concerns)

Three decoupled layers so a new converter reuses ~90% of existing
machinery:

```
┌──────────────────────────────────────────────────────────────┐
│ Layer 3 — Converter (per code, the ONLY thing a new code      │
│           author writes)                                       │
│   detect() + read_native() + declare products/units/cosmology │
└───────────────────────────┬──────────────────────────────────┘
                            │ uses
┌───────────────────────────▼──────────────────────────────────┐
│ Layer 2 — NativeReaderAdapter (per native FORMAT, shared      │
│           across codes that share a format)                    │
│   GadgetHDF5Reader, ASDFPack9Reader, GenericIOReader,          │
│   RamsesFortranReader, AMReXPlotfileReader, BigFileReader,     │
│   ZarrReader, FitsHealpixReader, OrbaxReader, CarpetHDF5Reader,│
│   ColdiceReader, ConsistentTreesReader, SubLinkReader, …       │
│   → delegates to yt / abacusutils / swiftsimio / ytree /       │
│     genericio / bigfile (lazy optional deps)                   │
└───────────────────────────┬──────────────────────────────────┘
                            │ feeds
┌───────────────────────────▼──────────────────────────────────┐
│ Layer 1 — IndexBuilder toolkit (format-agnostic, framework-   │
│           provided; a converter calls these, never reimplements)│
│   HealpixTileIndexer, OctreeNodeIndexer, KDTreeIndexer,        │
│   HaloParticlePointerIndexer, MergerTreeBranchIndexer,         │
│   LightconeShellIndexer, FieldChunkIndexer,                    │
│   PhaseSpacePatchIndexer, GrSliceIndexer, CheckpointIndexer    │
│   + ManifestWriter + ProvenanceWriter + UnitFrameWriter        │
└──────────────────────────────────────────────────────────────┘
```

**The key insight:** most new codes share a *native format* with an
existing code (Section 5.5). A new code on Gadget HDF5 (e.g. a new
SWIFT-physics variant) needs **only** a thin Layer-3 converter that
reuses the existing `GadgetHDF5Reader` (Layer 2) and the existing
index builders (Layer 1). Even a genuinely new format only requires
one new Layer-2 reader; Layers 1 + 3 mostly compose.

### 5.2 Two converter families

- **Input converters** — `(native IC + config + cosmology) →
  OUF-Sim input/`. Read a parameter file + IC field + transfer
  function → standardised `input/` record. Enables reproducibility +
  re-launch.
- **Output converters** — `(native snapshot / halo / tree /
  lightcone / field / phase-space / GR / checkpoint) → OUF-Sim
  output/`. Wrap native files + build sidecar indexes + emit manifest.

A code typically ships both (paired by `code` name), but either may
exist alone (some public products have output only).

### 5.3 Converter ABC (Layer 3 — the minimal contract)

```python
@dataclass(frozen=True)
class ProductDecl:
    product: str                    # "snapshots" | "halos" | "tree" | …
    native_format: str              # "Gadget HDF5" | "ASDF/pack9" | …
    indexes: Tuple[str, ...]        # which Layer-1 indexers to run
    fields: Tuple[str, ...]         # canonical field names exposed

class SimConverter(abc.ABC):
    code: ClassVar[str]                     # "Gadget-4", "AREPO", "ABACUS"
    sim_kind: ClassVar[str]                 # "nbody" | "sph" | "amr" | …
    capabilities: ClassVar[BackendCapabilities]

    @abc.abstractmethod
    def detect(self, path: Path) -> bool:
        """Sniff a directory; return True if this converter handles it."""

    @abc.abstractmethod
    def declare_products(self, src: Path) -> Tuple[ProductDecl, ...]:
        """List products found at src + which indexes each needs."""

    @abc.abstractmethod
    def read_cosmology(self, src: Path) -> dict: ...
    @abc.abstractmethod
    def read_unit_frame(self, src: Path) -> dict: ...

    # Provided by the framework — a converter rarely overrides:
    def convert(self, src: Path, out: Path, *, projection: str = "native",
                build_indexes: bool = True) -> OUFSimManifest:
        """Default: for each declared product, wrap native files,
        run the declared Layer-1 indexers, write manifest +
        cosmology + unit_frame + provenance. Override only for
        format quirks."""
```

`@register class AbacusSummitOutputConverter(SimConverter): …` — same
registry idiom as Pillar 1 survey loaders. The base `convert()` is
**concrete**: it iterates `declare_products()`, dispatches each to its
Layer-2 reader + Layer-1 indexers, and assembles the manifest. A
converter author writes `detect`, `declare_products`,
`read_cosmology`, `read_unit_frame` — four small methods — and
inherits everything else.

Converters **never re-encode** native particle/cell data; they wrap +
index.

### 5.4 Adding a new simulation code — the straightforward path

For a new code **on an existing native format** (the common case):

1. Subclass `SimConverter`; set `code`, `sim_kind`, `capabilities`.
2. Implement `detect()` (e.g. check for a signature file/header).
3. Implement `declare_products()` returning `ProductDecl`s that
   reference **existing** Layer-2 readers + Layer-1 indexers.
4. Implement `read_cosmology()` + `read_unit_frame()` (parse the
   header / parameter file).
5. `@register`. Done — no database / view / orchestration changes.

For a new code **with a genuinely new native format**, add **one**
Layer-2 reader (a `NativeReaderAdapter` exposing
`iter_chunks(selector) -> Iterator[pa.Table]` + `read_region(...)`),
delegating to the code's own Python reader where one exists. Then
follow steps 1–5. The Layer-1 index builders are format-agnostic and
do not change.

A unit-test harness (`SimConverterContractTest`) validates any new
converter against the contract: detect round-trip, manifest
schema-validity, index completeness, partial-access correctness
(a cone read through OUF-Sim must equal the native reader's cone
read), unit-frame declaration present. New code authors run this to
prove conformance.

### 5.5 Native-format sharing → converter reuse

Most public codes cluster onto a few native formats (from the
research landscape §5.5). Each format = one Layer-2 reader, shared:

| Native format (Layer 2 reader) | Codes / suites sharing it |
|---|---|
| Gadget HDF5 | Gadget-3/-4, GIZMO, SWIFT, AREPO, TNG, EAGLE, SIMBA, FIRE, MTNG, FLAMINGO (via swiftsimio), BAHAMAS, Magneticum |
| ASDF / pack9 | AbacusSummit, AbacusCosmos |
| GenericIO | HACC: Outer Rim, Last Journey, Q-Continuum, Mira-Titan |
| RAMSES Fortran multi-file | Horizon-AGN, NewHorizon, OBELISK, SPHINX |
| AMReX plotfile | Nyx, CASTRO, ERF |
| Enzo HDF5 hierarchy | Enzo, Renaissance, RomulusC |
| BigFile | FastPM, MP-Gadget |
| orbax checkpoint | pmwd, JaxPM |
| Carpet / Chombo HDF5 | Einstein Toolkit, GRChombo, gevolution |
| ColDICE binary | ColDICE (phase-space) |
| GAMER HDF5 | GAMER-2 (ψDM) |
| Subfind HDF5 | TNG, EAGLE, MTNG, Auriga, FABLE (halo product) |
| CompaSO ASDF | AbacusSummit (halo product) |
| ROCKSTAR + Consistent Trees | Quijote, Aemulus, Uchuu, UNIT (halo + tree) |
| SubLink / LHaloTree HDF5 | TNG, MTNG, EAGLE (tree product) |
| FITS HEALPix | Buzzard, FLAMINGO maps, GLASS (lightcone) |
| parquet (HEALPix-partitioned) | CosmoDC2, Flagship-2, Uchuu mocks (lightcone catalog) |
| BORG HDF5 chain | BORG family, MANTICORE, SIBELIUS-DARK (ic_posterior) |

≈ 18 Layer-2 readers cover the **entire** public landscape. Adding
the 100th simulation code is, in the overwhelming majority, a
4-method Layer-3 converter against a reader that already exists.

### 5.6 Projection choice (the convertibility requirement)

A single native simulation can be converted into **multiple
OUF-Sim databases** with different projections:
- `projection="particle"` — particle-centric; HEALPix tile + KD-tree
  indexes for spatial cone reads.
- `projection="halo"` — halo-centric; halo catalog primary, particle
  subsamples as sub-objects.
- `projection="lightcone"` — lightcone-centric; HEALPix-shell index
  primary.
- `projection="field"` — gridded-field-centric; Zarr cubes primary.
- `projection="phase_space"` — sheet-centric (ColDICE / ψDM).

Each projection is a distinct OUF-Sim record, linked back to the
source via a `converted_projection` lineage edge. Mirrors the OUF
data freedom to convert a survey into the format with different
column / weight / partition choices.

## 6. SimDatasetView — partial access

Twin of `DatasetView`. The load-bearing API. Every reader takes a
mandatory selector + optional MPI/GPU directive.

### 6.1 Selector taxonomy (uniform across backends)

- **Spatial**: `cube(xlo,xhi,ylo,yhi,zlo,zhi)`, `cone(lon,lat,radius)`,
  `skypatch(...)`, `tiles=[...]` (HEALPix), `octree_node=id` (AMR),
  `bbox6d=...` (phase-space patch), `grid_region=...` (field chunk).
- **Temporal**: `z=`, `snapshot=`, `shell_z_range=(zlo,zhi)`,
  `gr_slice_t=...` (GR time-slice), `step=...` (checkpoint).
- **Structural**: `halo_id=` (→ member particles via pointer),
  `tree_branch(start_halo_id=)` (→ progenitor chain),
  `chain_link=` (→ realised snapshot), `level_range=(lo,hi)` (AMR),
  `gauge=...` (GR field).
- **Projection**: `fields=(...)` — column subset at I/O time.

The selector set is **extensible per product**: each product type
declares the selectors its index supports (a particle table supports
spatial + structural; a GR field supports `gr_slice_t` + `gauge` +
`grid_region`; a checkpoint supports only `step`). The view raises
informatively if a product is asked for an unsupported selector.

### 6.2 Return contract

Never a whole-snapshot array by default. One of:
- lazy view (pyarrow Dataset / xarray / Dask);
- chunked iterator (`iter_particles(cone=…, batch=N) -> Iterator[pa.Table]`);
- sub-region materialiser (one tile / one node / one halo).

`scan_full()` exists but is an explicit, loud, documented escape
hatch.

### 6.3 MPI + GPU

```python
view.iter_particles(cone=…, fields=…, batch=N,
                    mpi_comm=comm,        # per-rank-local view
                    device="cuda:0")      # GPUDirect / kvikIO landing
```

Backends declare `BackendCapabilities(supports_mpi, supports_gpu_direct,
…)`. Reader raises informatively if asked beyond capability. Default:
single-process CPU. The API must expose `mpi_comm` + `device` from
day one even when every backend declares `False`.

## 7. Orchestration — region selection → simulation request

The new headline requirement: **the OUF-Sim database is the basis to
orchestrate the launch and selection of regions to simulate.**

### 7.1 What orchestration does (and does not)

- **Does**: query the SimDatabase (region index + lineage + cosmology)
  to *select* a region of interest, then emit a standardised
  **SimulationRequest** describing what to (re-)simulate.
- **Does not**: run the simulation. The request is handed to an
  external code (Gadget / AREPO / RAMSES / GIZMO / a future
  `flip_simulation`). Pillar 3 never submits a Slurm job or calls a
  sim binary. (Rule 4.)

### 7.2 Region selection inputs

- **Existing simulation regions** — overdensities, clusters, voids,
  filaments catalogued in `regions.parquet`.
- **Observed structures** — a region can be *pinned to data* by
  cross-referencing a Pillar-1 artefact on disk (a cluster catalog,
  a void catalog, a peculiar-velocity reconstruction). This is the
  only Pillar-1 touch-point and it is **file-based, not import-based**
  (Rule 1): the region selector reads an OUF parquet path, it does
  not `import oneuniverse.data`.
- **Lineage** — avoid re-simulating a region already zoomed; the
  lineage graph records prior `zoom_in` edges.

### 7.3 SimulationRequest (the output artefact)

```python
@dataclass(frozen=True)
class SimulationRequest:
    request_id: str
    parent_sim: Optional[str]           # the sim this region came from
    region: RegionSpec                  # bbox / cone / Lagrangian patch
    target_resolution: float            # mass or spatial resolution
    physics: Tuple[str, ...]            # "dm","hydro","mhd","rt","cr"
    cosmology_ref: str
    ic_strategy: str                    # "zoom_from_parent_ic" |
                                        # "constrained_from_posterior" |
                                        # "fresh"
    code_hint: Optional[str]            # suggested code, non-binding
    provenance: dict
```

The request is itself stored in the database (a pending-request
table) so the eventual external run + re-ingest closes the loop and
records a `resimulate_higher_res` / `zoom_in` lineage edge when the
output returns.

### 7.4 Why this is the digital-twin substrate

A constrained realisation of the local Universe (BORG / MANTICORE)
ingested as OUF-Sim → its regions catalogued → orchestration selects
the Coma cluster / the Boötes void / an observed TDE host → emits a
zoom request → external code runs → output re-ingests as a child
OUF-Sim with `zoom_in` lineage. Iterating this *is* incremental
digital-twin construction. Pillar 3 owns every step except the run.

## 8. Convertibility — many databases from one simulation

The requirement: "since a nearly infinite way of simulating is
possible, the user will be able to convert the input and output of a
simulation into different databases if needed."

### 8.1 Mechanisms

1. **Projection converters** (Section 5.3) — one native sim →
   N OUF-Sim records (particle / halo / lightcone / field-centric),
   linked by `converted_projection` lineage.
2. **Database composition** — a `SimDatabase` can federate OUF-Sim
   records from heterogeneous codes (Gadget + AREPO + RAMSES) under a
   common index, because the manifest abstracts the native format.
3. **Re-indexing** — the same native files can carry *different*
   sidecar index sets tuned to different access patterns (cone-read
   index vs halo-join index vs lightcone-shell index) without
   duplicating the native data.
4. **Cross-database export** — a region selected in DB-A can be
   exported as a fresh OUF-Sim seed (IC patch + cosmology) into
   DB-B, supporting different downstream workflows.

### 8.2 Invariant

All conversions are **lossless wrappers + sidecar indexes**, never
re-encodings of native particle/cell data. The lineage graph records
every conversion so provenance is never lost.

## 9. Mapping to the OUF data stack (recognisability)

| Concept | OUF data | OUF-Sim |
|---|---|---|
| On-disk record | `{survey}/oneuniverse/` (manifest + parquet) | `oufsim_{sim}/` (manifest + native wrappers + indexes) |
| Format version | `oneuniverse_format_version` | `oufsim_format_version` |
| Typed manifest | `Manifest` | `OUFSimManifest` |
| Partitioning | HEALPix NSIDE=32 NEST | HEALPix tile index + octree + KD-tree (per backend) |
| View | `DatasetView` (cone / sky-patch / pushdown) | `SimDatasetView` (+ cube / octree / halo / branch; MPI + GPU) |
| Loader | survey loaders (`@register`) | sim converters (`@register`, input + output) |
| Cross-record links | ONEUID + sub-object sidecars | lineage graph (zoom / posterior / projection edges) |
| Bitemporal | dataset validity + versioned ONEUID | versioned OUF-Sim records + lineage validity |
| Combine / orchestrate | weights + (Pillar 2) MeasurementSet | region selection → SimulationRequest |
| Convertibility | survey → OUF with column / weight choices | sim → OUF-Sim with projection / index choices |

## 10. Dependency posture

**Hard deps:** numpy, pyarrow, h5py, healpy, pyyaml.
**Optional extras (lazy):** abacusutils, yt, swiftsimio,
illustris_python, ytree, halotools, genericio, bigfile, zarr,
mpi4py, kvikIO/cucim, mocpy.
**Forbidden:** any `import oneuniverse.data` / `oneuniverse.combine`.
Pillar-1 interaction is file-path-based only.

## 11. Open design questions (resolve before coding)

1. **Subpackage vs sibling repo.** `oneuniverse.sim` (co-development
   ease) vs `onesim` (enforced zero coupling). Bias: subpackage now,
   extractable later — but enforce the no-import rule with a lint
   check from day one.
2. **Index store: parquet vs Zarr vs Arrow IPC.** Parquet for small
   indexes (catalogs, pointers); Zarr v3 + sharding for large
   per-pixel sidecar arrays + gridded fields. Likely both.
3. **Lineage graph store.** Reuse the Pillar-1 sub-object sidecar
   pattern (parquet edge table + bitemporal manifest) vs a proper
   graph store. Bias: parquet edge table, since the Pillar-1 idiom
   is proven and dependency-light.
4. **Region catalog schema.** What is the canonical RegionSpec —
   Eulerian bbox, Lagrangian patch (for zoom ICs), or both? Zoom
   re-simulation needs the Lagrangian patch; observed-structure
   pinning needs the Eulerian region. Probably store both.
5. **SimulationRequest lifecycle.** Pending → dispatched → running →
   ingested. Who owns the state machine? Pillar 3 stores the states;
   the external runner updates them out-of-band (file-based).
6. **MPI capability granularity.** Per-backend `supports_mpi` vs a
   universal collective-read shim. Bias: per-backend, declared in
   `BackendCapabilities`.
7. **GPU read maturity.** parquet + Zarr GPU-direct via kvikIO is
   production; HDF5 GPUDirect is alpha. Ship GPU paths only for
   backends where the underlying lib is stable; declare `False`
   elsewhere.
8. **Cosmology declaration duplication.** Rule 1 forbids importing
   Pillar-1 cosmology metadata helpers. OUF-Sim `cosmology.yaml`
   duplicates the small schema. Accept the duplication.

## 12. What Phase S1 delivers, and what comes next

**Phase S1 delivers (this document):**
- The three-layer architecture (format / database / orchestration).
- The OUF-Sim on-disk format with input + output split, covering
  **all eleven storage primitives + all code classes** from the
  research landscape (§3.5 coverage matrix — no sim shape left out).
- The **3-layer modular converter design** (Converter / NativeReader
  / IndexBuilder), the 4-method `SimConverter` contract, the
  "add a new code in 5 steps" path, and the native-format sharing
  matrix (~18 Layer-2 readers cover the entire public landscape).
- The `OUFSimManifest`, `ProductDecl`, `SimConverter`,
  `SimDatasetView`, `SimDatabase`, `SimulationRequest`,
  `BackendCapabilities`, `RegionSpec` type sketches.
- The per-product extensible selector taxonomy + MPI/GPU posture.
- The lineage + convertibility model.
- The mapping to the OUF data stack.
- Eight open design questions.

**Phase S1 does NOT deliver:** any code. No package skeleton, no
backend, no converter. Implementation is later phases.

**Candidate next phases (no commitment yet):**
- **Phase S2 — package skeleton + types.** Stand up
  `oneuniverse.sim` (or `onesim`) with the dataclass types from this
  doc + a no-Pillar-1-import lint guard. Still no backend.
- **Phase S3 — first backend + partial-access proof-of-concept.**
  AbacusSummit ASDF/pack9 via abacusutils; manifest + sidecar
  indexes; `iter_particles(cone=…, batch=…)` with a measured
  throughput floor; MPI variant; GPU-direct variant via kvikIO
  parquet sidecar. Validate byte-identical results vs abacusutils.
- **Phase S4 — SimDatabase + region index + lineage.**
- **Phase S5 — orchestration: region selection → SimulationRequest.**
- **Phase S6 — second backend (Gadget HDF5) + projection converters.**
- **Future** — AMR / HACC / BORG / BigFile backends; Wave-1 ingest
  (FLAMINGO, MTNG, Buzzard, Flagship-2); mini-sim run integration
  (**deferred indefinitely**); IC sampler integration (**deferred
  indefinitely**).

This is a very long-term, very challenging build. The architecture
above is deliberately conservative: wrap don't re-encode, partial
access first, minimal deps, MPI/GPU-ready, reuse every Pillar-1
lesson (manifest discipline, bitemporal lineage, pluggable
converters, sidecar indexes). The hard work is in the backend
converters and the partial-access index builders; the format +
database + orchestration scaffolding is well-served by the patterns
this document lays out.

## 13. References

- [`2026-05-28-pillar3-simulation-digital-twin.md`](2026-05-28-pillar3-simulation-digital-twin.md)
  — Pillar 3 large-scope roadmap.
- [`../research/cosmological_simulation_landscape.md`](../research/cosmological_simulation_landscape.md)
  — codes, suites, representations, OUF-Sim §6 sketch.
- [[pillar3-partial-access-and-minimal-deps]] — pinned constraints.
- [[oneuniverse-pillars]] — three-pillar architecture.
- [`2026-05-28-pillar1-data-combine-measure.md`](2026-05-28-pillar1-data-combine-measure.md)
  — OUF data stack (the recognisability twin).
