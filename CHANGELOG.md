# Changelog

## 1.0.0 — 2026-11-15

Complete rewrite of the package.

### Breaking

- New public API: `oneuniverse.data`, `oneuniverse.measure`,
  `oneuniverse.simulation`, `oneuniverse.reference`, `oneuniverse.twin`.
- OUF 3.0 on-disk format. There is no reader for OUF 2.x; reconvert from the
  survey files.
- OUF-Sim 1.0 on-disk format. There is no reader for OUF-Sim 0.1.
- Python 3.11 or newer.
- `oneuniverse/_registry.py` (compatibility shim) is gone; use
  `oneuniverse._core.registry`.

### Fixed

- ONEUID hydration returned the wrong objects (the row index did not match the
  source row). Hydration now uses explicit source rows and refuses a dataset
  changed since the index was built.
- Zel'dovich velocities were off by the factor `E(z)·h`; they follow
  `v = a·100·E(a)·f·ψ`.
- Growth rate used the approximation `f = Ω_m^0.55`, inconsistent with the
  growth factor D; D and f now come from one integral.
- The constrained-realisation engine reused the truth seed and so reproduced
  the truth at small scales. The seed is now explicit, with no default.
- Resimulation buffers were not periodic: targets near the box edge lost
  particles. Buffer selection now wraps.
- The Wiener filter ignored the survey mask; it now renormalises inside the
  mask and ignores the outside.

### Removed

- Stub loaders without a real reader.
- SQLite export (replaced by the DuckDB SQL face).
- Dead-end physics: COLA attempts, AMR, `gr_fields`, phase space, merger tree,
  zoom, merge and far-field.

### Added

- DuckDB SQL face over datasets and simulation stores, with `as_of` time travel.
- Reference engine (`oneuniverse.reference`): EH98 spectrum, Gaussian fields,
  Zel'dovich, particle-mesh, periodic buffered resimulation, TreePM,
  mask-aware Wiener filter, constrained realisation, lognormal mock.
- Twin contracts (`oneuniverse.twin`): engine interfaces, registry,
  `observe_from_view`, `run_engine_to_store`.
- Streaming bucketed conversion with bounded memory.
- Versioned, crash-safe sidecars (ONEUID, links) with `as_of`.
- `SimDatabase`: requests, lineage and crash reconciliation for simulation
  stores.
