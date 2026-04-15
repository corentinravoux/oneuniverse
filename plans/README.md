# oneuniverse Stabilisation Plans

Implementation plans for the Pillar 1 stabilisation effort (architecture
audit done 2026-04-15).

- [`2026-04-15-stabilisation-roadmap.md`](2026-04-15-stabilisation-roadmap.md) — master phase-level roadmap with rationale, design decisions, and scope per phase.
- [`2026-04-15-phase1-ouf-v2.md`](2026-04-15-phase1-ouf-v2.md) — detailed task-by-task plan for Phase 1 (OUF 2.0 manifest + atomic writes + content hashing). **Starting here.**

Phases 2–6 each get their own detailed plan document as we reach them
(written using `superpowers:writing-plans`).

## Phase status

| # | Name                                    | Status       |
|---|-----------------------------------------|--------------|
| 1 | OUF 2.0 (typed manifest, hashes, atomic writes) | in progress |
| 2 | DatasetView + pyarrow.dataset backend   | pending      |
| 3 | HEALPix spatial partitioning            | pending      |
| 4 | Unified ONEUID (z-type rules, subsets, named indices) | pending |
| 5 | Streaming hydration                     | pending      |
| 6 | Housekeeping (dict consolidation, dtypes)| pending     |
