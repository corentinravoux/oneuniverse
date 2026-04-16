# oneuniverse Stabilisation Plans

Implementation plans for the Pillar 1 stabilisation effort (architecture
audit done 2026-04-15).

- [`2026-04-15-stabilisation-roadmap.md`](2026-04-15-stabilisation-roadmap.md) — master phase-level roadmap with rationale, design decisions, and scope per phase.
- [`2026-04-15-phase1-ouf-v2.md`](2026-04-15-phase1-ouf-v2.md) — detailed task-by-task plan for Phase 1 (OUF 2.0 manifest + atomic writes + content hashing). **Starting here.**
- [`2026-04-15-phase3-healpix-partitioning.md`](2026-04-15-phase3-healpix-partitioning.md) — detailed task-by-task plan for Phase 3 (HEALPix spatial partitioning on disk + Cone/SkyPatch cell pruning).

Phases 2–6 each get their own detailed plan document as we reach them
(written using `superpowers:writing-plans`).

## Phase status

| # | Name                                    | Status       |
|---|-----------------------------------------|--------------|
| 1 | OUF 2.0 (typed manifest, hashes, atomic writes) | **complete (2026-04-15, 130/130 tests green)** |
| 2 | DatasetView + pyarrow.dataset backend   | **complete (2026-04-15, 145/145 tests green)** |
| 3 | HEALPix spatial partitioning            | **complete (2026-04-15, 156/156 tests green)** |
| 4 | Unified ONEUID (z-type rules, subsets, named indices) | pending |
| 5 | Streaming hydration                     | pending      |
| 6 | Housekeeping + `weight/` → `combine/` redesign | pending     |
