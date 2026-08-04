# oneuniverse — whole-package analysis + improvement plan

*2026-08-04. Sources: the code-only knowledge graph (`graphify-out/`, 3647
nodes / 10543 edges / 88 communities), grep ground-truth, the test suite (838
tests / 213 files), and project memory. Lens: the endgame.*

---

## 0 · The endgame (the yardstick every improvement is measured against)

**A digital twin of the Universe:** combine every cosmological observation into
a *single Bayesian posterior on the matter-density + velocity field*. oneuniverse
is the **data + orchestration substrate** for that — not the estimator (that is
`flip`, Pillar 2), not the physics engine (pluggable). The chain the endgame
needs to run end-to-end:

```
   real surveys ─► ingest (P1 data/combine) ─► measure (P1→P2 MeasurementSet)
        ─► estimate / CONSTRAIN (P2 flip · P3 twin) ─► posterior field
```

Everything below is ranked by **how directly it unblocks that chain on real
data** — because the architecture is already sound; what is missing is *flow*.

---

## 1 · Purpose & structure (what the graph + grep show)

| Layer | LOC | graph nodes | role | state |
|---|--:|--:|---|---|
| `data/` | 9017 | 586 | Pillar 1 ingest, OUF, ONEUID, DatasetView | mature, densest module (1808 intra-edges) |
| `simulation/` | 4786 | ~330 | Pillar 3 storage substrate + toy physics | broad, uneven test depth |
| `measure/` | 1563 | 166 | P1→P2 MeasurementSet (the handoff) | built, **real-data-unvalidated** |
| `combine/` | 1253 | 133 | weights, WeightedCatalog | mature |
| `twin/` | 623 | 70 | data↔sim coupling (the digital twin MVP) | **only half-wired** |

**Boundaries (grep-verified, not just graph):**
- **Rule-1 holds:** `simulation/` imports *nothing* from `data`/`combine`;
  `data`/`combine` import *nothing* from `simulation`/`twin`. The 50 `data↔oufsim`
  graph edges are **AST symbol-name collisions** (both layers own `Manifest`,
  `converter.py`, `database.py`), not real coupling — a caveat for reading the
  graph: trust it for *relative* structure, verify any specific edge with grep.
- **`CosmologySpec` is the #1 hub (236 edges)** and lives entirely in
  `simulation/*`+`twin/` — zero references in Pillar 1. The cosmology-free rule
  is empirically enforced.
- **`twin/` bridges only the simulation side.** It imports 9 `simulation.*`
  modules and **zero** `data`/`combine`. Its "data" is synthetic
  (`mock_observe`, `mock_survey`, `mock_challenge`); the code says so:
  *"no oneuniverse.data import yet (that is the next…)"*. **This open socket is
  the single missing span between the two halves of the package.**

**Test suite (838 tests):** `data/` ≈ 450 tests (incl. the 214 infra tests my
keyword map first mis-bucketed) — very well covered. Thinner where the endgame
lives: `measure/` 69, `twin/` 35 (all synthetic), `simulation` ~130 for 4786 LOC
(the thinnest per-LOC), and **`packed` backend = 2 tests** despite being the
storage-generality proof.

---

## 2 · The gap analysis (endgame lens)

The substrate is built; **real observations cannot yet reach the twin.** Three
concrete breaks, in dependency order:

1. **Only 2 of ~10 loaders are real-data-validated** (DESI + eBOSS). The rest are
   schema scaffold ([[project_p1_real_ingestion_status]]). The ingest end of the
   chain is narrow.
2. **The measure builders (WL / PV / SN / Lyα) are unvalidated on real catalogs.**
   The P1→P2 handoff — the `MeasurementSet` — only has synthetic-fixture tests
   for 4 of its 6 probes.
3. **The twin's data socket is empty.** No path takes a real `DatasetView` +
   selection into `mock_observe`'s place. The digital twin can only constrain
   against synthetic tracers.

Break 3 depends on 2 depends on 1. This is one span, built in three segments.

---

## 3 · Improvement plan (prioritised)

### P0 — Close the real-data span to the twin (the endgame-critical work)

**P0.1 — Validate the measure builders on real data (the M3 item).**
Take one real catalog already ingested (eBOSS QSO, or a DESI PV / SN sample) and
run the corresponding builder (`build_peculiar_velocity` / `build_sn_hubble` /
`build_cosmic_shear`) end-to-end, asserting `check_invariants()` + physically
sane n(z)/window/weights. *Why first:* the `MeasurementSet` is the P1→P2 contract;
until it is proven on real data it cannot be trusted as the twin's or flip's
input. *Effort:* M (needs the real catalogs on disk — the standing blocker).

**P0.2 — Wire a real observation model into the twin. ✅ done 2026-08-04 (dummy realization).**
`twin/observe_from_view.py` grids a Pillar-1 catalog's positions into an
`Observation` — the first real `data→twin` edge. See
`research/2026-08-04-dev-plan-twin-data-socket-dummy.md`. Real sky→comoving
variant still pending real catalogs.
Add `twin/observe_from_view.py`: a `DatasetView` (+ its window + selection) →
the same `Observation` object `mock_observe` produces. Keep `mock_observe` for
tests; the new path is the first real `data`→`twin` edge (Rule-1 still lets
`twin` import both). *Why:* this is the literal missing span the graph found —
it turns the twin from "permitted bridge" into "connected bridge". *Effort:* M.
Gated by P0.1 (needs a validated MeasurementSet to consume).

**P0.3 — One real-data twin smoke test. ✅ done 2026-08-04 (dummy realization).**
Closed-loop test: dummy catalog → observe → wiener → `recover_metrics` recovers
the known truth field (r>0.6 at large k). Swaps to real data with no code change.
After P0.2: a single end-to-end test (real view → observe → wiener_reconstruct →
`recover_metrics`) so the real path is regression-locked. *Respect the
"don't bloat tests" preference — one meaningful test, not a suite.*

### P1 — Shore up the endgame-critical substrate (targeted, low-bloat)

**P1.1 — Parity-test the `packed` backend (2 → ~5 tests). ✅ done 2026-08-04.** Storage generality
is load-bearing for the endgame (real sims arrive in many formats — Gadget/
Abacus/BigFile). The 2nd backend proving the pattern has 2 tests; give it the
same read-parity + wrap-in-place checks `oufsim` has. *Effort:* S.

**P1.2 — A 3rd engine on the plug-in contract. ✅ done 2026-08-04.** `twin/engine.py` proves
generality with 2 engines (linear-forward, wiener-recon). The contract stays
honest only if a 3rd, differently-shaped engine fits without changing it — e.g. a
lognormal or constrained-realization forward engine registered via
`register_engine`. *Why:* the endgame swaps in real engines (BORG/SBI); the
socket must be proven to flex now, cheaply. *Effort:* S–M.

### P2 — Structural hygiene (only as it impedes work — not for its own sake)

**P2.1 — Consider sub-packaging `data/` (9 kLOC / 586 nodes).** The review
already split `converter.py`; the next densest single files are `database.py` and
`dataset_view.py`. *Do not do a wide refactor for tidiness* — only split a file
when you are next editing it and it no longer fits in context. Track, don't
schedule.

**P2.2 — Leave the boundaries alone.** Rule-1 and cosmology-free are enforced and
correct; the graph confirmed it. No action — just don't regress them (the
existing guard tests cover this).

### Explicitly *not* now
- No mass test additions (respecting the "you added too much test" preference —
  every P0/P1 test above is one targeted case, not a suite).
- No `onemeasure`/`onecorr` second-consumer build — real, but far past the P0 span.
- No production inference engine (BORG/SBI) — the substrate must carry real data
  *first*; the engine is meaningless without it.

---

## 4 · The one-sentence verdict

The architecture is finished and correct; the **only thing between oneuniverse
and its endgame is real observational flow** — validate the measure handoff on
real catalogs (P0.1), open the twin's data socket (P0.2), lock it with one test
(P0.3). Everything else is optional hygiene. The graph's most important finding
is also the simplest: the digital twin is a bridge with one end not yet attached.
