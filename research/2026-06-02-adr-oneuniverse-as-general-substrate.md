# ADR — oneuniverse as a general substrate for forward-modelling / inference engines

**Status:** Accepted (2026-06-02, C. Ravoux + critical review).
**Type:** Architecture Decision Record. Supersedes the implicit "oneuniverse
builds the digital twin" framing with an explicit substrate-vs-engine split.

---

## Context

The long-term goal is a "digital twin": a posterior on the matter
density+velocity field combining all observations, with a forward-modelling
side. A critical review (this session) flagged that the hard novel core is
the *inverse* problem, while what is built is a *substrate + targeted-
resimulation* engine, and that the data↔sim coupling — the only genuinely
novel part — was the least specified.

The owner's resolution: **oneuniverse is a *general substrate*, not a
committed inference method.** It owns **data content (Pillar 1)** and
**simulation storage + orchestration (Pillar 3)**; the **forward-
modelling / inference** is **pluggable** — an external engine (e.g. BORG),
a home-grown engine, or none. Generality (many solutions can be built on
it) is the explicit design value.

## Decision

### D1 — Two layers, one clean line

- **Neutral substrate** (forward-model-agnostic): data store + selection/
  window/n(z) (Pillar 1); sim store + region/ensemble orchestration + IC
  extraction + far-field provision + output ingest + lineage (Pillar 3).
- **Reference engine** (one plugin among many): the linear full-sim +
  fast-PM resimulation. Shipped to prove the contract and provide a default
  — explicitly *a* solution, **not** part of the neutral substrate.

### D2 — "No forward-model commitment in the substrate" rule

Analogous to the load-bearing *no-cosmology-in-Pillar-1* rule. The neutral
substrate must not bake in a gravity model, IC power spectrum, bias model,
or growth convention. Those belong to the engine. The substrate provides
*inputs* (IC field, far-field, data, cosmology spec) and *stores outputs*;
it does not assume how the engine evolves them. (The linear+PM far-field is
part of the **reference engine**, not the substrate.)

### D3 — The plug-in contract (illustrative ABC)

```python
class ForwardEngine(abc.ABC):
    """Pluggable forward-model / inference engine. The substrate provides
    inputs + ingests outputs; the engine owns the physics + inference."""
    name: ClassVar[str]
    requires: ClassVar[Tuple[str, ...]] = ()      # optional deps, e.g. ("borg",)

    @abc.abstractmethod
    def consume(self, *, ic, farfield, cosmo, region, data=None,
                plan) -> "EngineRun": ...
    #   ic       : substrate-extracted IC field for the region
    #   farfield : large-scale potential / displacement (long-range force)
    #   cosmo    : CosmologySpec (engine owns how it is used)
    #   data     : Pillar-1 data + selection + window  (for likelihood engines)
    #   plan     : ExecutionPlan (sequential / MPI / GPU; memory budget)

    @abc.abstractmethod
    def run(self, run: "EngineRun") -> "ProductBundle": ...
    #   returns snapshots / fields / posterior samples for ingest
```

Substrate side provides: `extract_ic(store, region)`,
`farfield(store, region, scale_factors)`, `ingest(store, bundle, lineage)`,
and (Pillar 1) `data + selection + window + n(z)` for likelihood-based
engines.

### D4 — Two orchestration modes (one schema)

`SimulationRequest` / `SimDatabase` must serve both, from day one of the
schema (even if only mode A is implemented first):
- **A — zoom/resim:** one region, higher fidelity (S8 as planned).
- **B — ensemble:** N realisations over a parameter/phase prior (for SBI,
  covariance, training sets). Same `SimulationRequest`, `kind="ensemble"`.

### D5 — Two products (portfolio de-risking)

- **P1 + P2** = cross-survey constraint facilitator (multi-tracer, joint
  f·σ8 via flip etc.). Near-term, low-risk, *must produce published
  science* — it justifies the substrate.
- **P1 + P3 + coupling** = the generalised digital twin. High-risk,
  high-novelty. The coupling is the keystone.

### D6 — Prove generality incrementally; let the contract emerge (YAGNI)

Generality is a **hypothesis tested by integrations**, not a design done up
front. Build the contract minimally for the reference engine; **stretch it
one consumer at a time, refactoring when each reveals a gap.** Do not
pre-abstract for engines never run. Generality is *demonstrated* only when
≥2 engines satisfy the same contract.

## Feasibility progression (the order to prove it)

| # | Consumer | What it stresses | Note |
|---|---|---|---|
| 1 | Linear + fast-PM (own) | the reference engine + ingest | the default |
| 2 | **External forward model** | the *neutral-substrate* claim | **do early — cheapest decisive test** |
| 3 | Mid-complexity own FM (2LPT/COLA + bias) | richer IC/data coupling | constrained-realisation level |
| 4 | Simulation-based inference (SBI) | **all 3 pillars + ensemble mode + summary stats + statistical fidelity** | last; gated by Gate-2/3 error budget |

## Validation (the falsifiable tests)

- **Mock challenge:** inject a known IC field → mock-observe via Pillar 1
  (sample tracers + selection + noise) → constrain (linear constrained
  realisation on large scales) → resimulate small scales → recover truth
  within errors. Validates the coupling end-to-end on the dummy where truth
  is known.
- **Generality test:** ≥2 engines (reference PM + a deliberately-different
  toy/external engine) satisfy the same `ForwardEngine` contract.
- **SBI-fidelity gate:** the Gate-2/3 resimulation error budget must be
  small enough that summary statistics of the cheap sims are unbiased;
  otherwise cheap sims bias the SBI posterior.

## Consequences

**Positive.** De-risks the science bet (no need to beat BORG); achievable,
infrastructure-class contribution (CosmoSIS/Cobaya/Astropy-class); P1+P2
delivers near-term science; the scale-separation unifies data constraint
(large scales, linear CR) with resimulation (small scales).

**Costs / risks (owned).** Generality *increases* surface area + maintenance
for a small team — discipline-per-feature must stay high. Value proposition
shifts from "we produce the twin" to "we enable twins" (state honestly per
audience; the own-engine option keeps a foot in discovery). Generality is
unproven until ≥2 integrations; the *convention/adapter layer* (units, IC
representation, growth/RSD conventions) is where generality is real work and
where it most easily leaks. Risk now concentrates in: a clean adopted
contract + a reference engine that matters — a more tractable risk than
"build a better BORG", but not free.

## Cross-references

- Feasibility study: `research/2026-06-02-resimulation-orchestration-feasibility.md`
- Orchestration plan: `plans/2026-06-02-phaseS8-resimulation-orchestration.md`
- Pillar rules (Rule 4 revised): memory `pillar3-partial-access-and-minimal-deps`
- Pillar 1 no-cosmology rule (the analogue for D2): memory
  `no-cosmology-in-pillar1`
