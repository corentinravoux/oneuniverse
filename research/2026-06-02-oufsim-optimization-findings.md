# OUF-Sim optimisation findings — linsim_demo run (2026-06-02)

Empirical hotspots from converting a small dummy linear simulation into
the OUF-Sim store, used to drive the Phase-S4/S5 optimisation work.

## Run

| Quantity | Value |
|---|---|
| Box / grid | 512 Mpc/h, 128³ (2,097,152 particles/snapshot) |
| Snapshots | z = 0.0, 0.5, 1.0, 2.0 |
| Products | snapshots, fields, halos, lightcone |
| Native size | 479 MB |
| Store size | 578 MB (re-encoded) |
| `generate` wall | 4.0 s |
| `convert` wall | 4.5 s |
| `convert` peak RAM | 375 MB (tracemalloc) |
| Parquet files written | 349 |

## Partial-access read (the design pays off)

| Query | Partitions touched | Reduction |
|---|---|---|
| Cube 64³ → particles | 1 / 64 chunks | 64× |
| Cube 64³ → field | 1 / 64 tiles | 64× |
| Cone r=25° → lightcone | 8 / 48 pixels | 6× |

Box read = 4.4 ms. The sidecar `_index.parquet` pruning works; reads are
**not** the bottleneck. The optimisation frontier is the **write/convert
path** and its **memory ceiling**.

## Hotspots (cProfile, cumulative)

| Rank | Cost | Source | Why |
|---|---|---|---|
| 1 | **2.30 s (52%)** | `pyarrow.parquet.write_table` ×349 | Many small partition files; serial write |
| 2 | 0.46 s | `cartesian_chunk_ids` | floor-div + clip over full N array |
| 3 | 0.40 s | `bbox_of` (×288, min+max) | two separate reductions per chunk |
| 4 | 0.33 s | `{k: v[order]}` column gather | full sorted **copy of every column** |
| 5 | 0.15 s | global `argsort` (×18) | sorts entire snapshot to group by chunk |

## Where optimisation must go (→ Phase S4/S5)

1. **Parallelise partition writes (biggest win).** Parquet write is 52% of
   convert and is embarrassingly parallel per chunk. Declare
   `parquet_write` / `particle_chunking` as MPI- and thread-capable in
   `BackendCapabilities.heavy_step_modes`; **MPI rank-per-chunk** or a
   thread pool writes chunks concurrently. GPU path: cuDF + GPU-direct
   storage.
2. **Bound memory by streaming, not global sort.** Current path loads a
   whole 100 MB snapshot, `argsort`s it, then materialises a sorted copy
   of *every* column → ~3–4× a snapshot live at peak (375 MB). Replace
   with a **bucket/counting pass** keyed on the known chunk count, writing
   one chunk at a time within `ExecutionPlan.memory_budget_bytes`. This is
   the Rule-5 "bounded-memory streamed" requirement made concrete.
3. **Fuse the index pass.** Compute chunk id + per-chunk bbox in a single
   streaming pass over positions (min & max together; or use analytic cell
   bounds) instead of 288 separate `min`/`max` reductions after the fact.
4. **Wrap, don't re-encode, at scale.** Store (578 MB) > native (479 MB)
   because the demo re-encodes a trivial native format into parquet. Real
   backends (AbacusSummit ASDF, Gadget HDF5) must **reference native files
   in place** and write only the sidecar index + manifest — the
   "manifest-of-manifests" from the S1 architecture. Re-encoding petabytes
   is never acceptable; the linear demo re-encodes only because its native
   files are tiny.
5. **Field tiles are already read-optimal.** `.npy` memmap tiles give 64×
   read pruning with `mmap_mode="r"` and no full-grid decode. Keep; the
   only write-side win is parallel `np.save`.
6. **Tune partition granularity.** 349 files at this size is write-heavy;
   for production, size chunks to a target rows-per-file (as OUF does with
   `MIN_ROWS_PER_PARTITION` + HEALPix auto-coarsening) and consider
   `pyarrow.dataset.write_dataset` to batch metadata.

## Concrete `heavy_step_modes` to declare (S4)

```
"particle_chunking": (SEQUENTIAL, MPI, GPU)   # rank/thread per chunk
"parquet_write":     (SEQUENTIAL, MPI)        # rank per partition file
"field_tiling":      (SEQUENTIAL, MPI)        # rank per tile
"index_build":       (SEQUENTIAL,)            # cheap, fused into the above
```

Each heavy step consults `ExecutionPlan(mode, memory_budget_bytes,
batch_rows)` and refuses a mode the backend cannot honour rather than
silently falling back to an unbounded in-memory path.
