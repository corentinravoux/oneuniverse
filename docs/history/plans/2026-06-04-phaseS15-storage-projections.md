# Phase S15 — OUF-Sim storage projections (wrap-in-place vs re-encode)

> **STATUS 2026-06-04 — CORE DONE (fields).** `field_projection='reference'`
> wraps the native `field.npy` (index-only, no copy; <20% of re-encode); reads
> memmap-match the re-encode path via `NumpyFieldAdapter`. Both modes coexist
> per product. ⏳ remaining: particle `reference` (row-id sidecar) + the
> chunk-sorted-native wrap demo + full benchmark — secondary.

> Develop **both** ways an OUF-Sim store can hold a simulation's bulk data, as
> a per-product choice. linear+PM dummy stands in for real codes; structure
> first. Closes audit gap "re-encode, not wrap" (notebook 04).

## The two projections

| | **`reencode`** (current) | **`reference`** (wrap-in-place) |
|---|---|---|
| bulk data | copied into parquet/`.npy` tiles | **stays in the native files**; the store holds only manifest + sidecar index |
| store size | ≈ native (duplicated) | ≈ index only (tiny) |
| native files | can be deleted | **must persist** |
| reader | uniform (parquet/npy) | per-format `NativeReaderAdapter` |
| compression | parquet snappy | native's own |
| portability | self-contained, movable | tied to native files |
| **petabyte scale** | ❌ never copy PB | ✅ the only viable mode |

Both coexist. **Convention:** bulk products (particles, fields) of a real sim →
`reference`; small/derived products (halos, lightcone catalogs, checkpoints) →
`reencode` (cheap, portable, queryable). The choice is per product, recorded in
the manifest.

This realises the S1 three-layer design: **Layer 1** format-agnostic
`IndexBuilder`, **Layer 2** per-format `NativeReaderAdapter`, **Layer 3**
`SimConverter`. `reference` is where Layer 2 pays off.

---

## Tasks (TDD; dummy npy as the "native format")

### T1 — Manifest carries a per-product `projection`
- `store_layout[product][...]["projection"] ∈ {"reencode","reference"}`.
- `write_oufsim_store(..., projection="reencode"|"reference"|{product:mode})`.
- **Test:** a re-encoded store records `projection="reencode"`; round-trip unchanged.

### T2 — `NativeReaderAdapter` ABC + a dummy-`npy` adapter
- `oufsim/native.py`: `NativeReaderAdapter` with `read_field_region(path, cell_slice)`
  (memmap a `.npy` field tile / sub-array) and `read_rows(path, row_slice)`
  (memmap particle rows). The contract a real backend (HDF5/ASDF/BigFile)
  implements later.
- Dummy adapter = numpy memmap.
- **Test:** the adapter reads a sub-region of a native `.npy` without loading
  the whole array (memmap), matching the full-load result.

### T3 — `reference` index builder (no data copy)
- For **fields**: index rows = `{tile bbox, native_file, cell_ranges}` pointing
  at the native `field.npy` (memmap by cell range) — **no tile copy**.
- For **particles**: the Zel'dovich/native order is *not* spatially sorted, so a
  reference index needs either (a) the native to be spatially pre-sorted (real
  codes: AbacusSummit cells / Hilbert order), or (b) a per-chunk **row-id list**
  sidecar (a permutation — smaller than re-encoding the floats). Implement (b)
  for the dummy and document (a) as the real-backend path.
- **Test:** reference store size ≪ native size (index-only for fields;
  index+row-ids for particles); no float data duplicated.

### T4 — `SimStore` reference-mode reads
- `read_box`/`read_field_box` branch on `projection`: `reference` → resolve the
  native file + region via the adapter (memmap), `reencode` → current path.
- **Test:** reference and re-encode reads return **identical** rows/fields for
  the same selector.

### T5 — "wrapped real-ish backend" demo
- Write the dummy native in a **chunk-sorted** layout (particles ordered by
  coarse cell — mimicking a real spatially-ordered sim), then **wrap** it
  (`reference`) so each chunk = a contiguous native row range (no row-id list
  needed). This is the real-backend ideal.
- **Test:** the wrapped store is index-only (≈ a few % of native) and reads
  match.

### T6 — Comparison + projection-choice convention + docs
- Benchmark: store size + read correctness for `reencode` vs `reference` vs
  `reference`-on-sorted-native.
- Document the convention; update notebook 04's "not working" → "now both
  modes"; memory.

---

## Success criteria
- `reference` store of a field is **index-only** (no `.npy` copy); reads via
  memmap match the re-encode path exactly.
- Particle `reference` (row-id sidecar) ≪ re-encode; chunk-sorted-native wrap is
  index-only.
- Projection is a per-product manifest choice; both modes coexist; the convention
  (bulk→reference, derived→reencode) is documented.

## Notes / honesty
- `reference` for **particles** only becomes index-only when the native is
  spatially sorted (real codes do this). The dummy's Zel'dovich order is not,
  so it needs the row-id sidecar — a real but smaller cost. T5 shows the ideal.
- Real native adapters (HDF5 parallel, ASDF/pack9, BigFile) are `future`; T2's
  ABC is the contract they implement.
