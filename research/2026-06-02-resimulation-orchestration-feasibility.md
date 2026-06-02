# Resimulation orchestration — critical feasibility research

**Author context:** Pillar-3 digital-twin core. Question posed by C. Ravoux:
can we run a cheap **full-volume** simulation that carries the large scales,
then selectively **re-simulate sub-volumes** at higher fidelity (mini-sims),
splice them back, and thereby build/update a digital twin without ever
running an expensive full nonlinear box? This document maps the idea onto
established methods, decomposes the machinery, defines the verification
protocol, and gives an honest physicist's verdict on what is achievable,
what is only approximate, and what is impossible.

**TL;DR verdict.** The proposal is **feasible as a controlled
approximation** — it is, almost exactly, **sCOLA** (spatially-split COLA)
combined with **phase-consistent zoom initial conditions** and a
**separate-universe external tidal background**. Each ingredient exists and
is validated in the literature to ~few-% on large scales. It is **not** a
route to *exact* full-nonlinear reconstruction, and crucially it **cannot
update the large-scale field from the mini-sims alone** — the small→large
back-reaction is not deterministically recoverable (EFT-of-LSS
stochasticity). The realistic deliverable is *"locally high-fidelity,
large-scale-consistent resimulation of selected sub-volumes, with a
quantified error budget,"* not a replacement for evolving the large scales.

---

## 1. The orchestration, decomposed into operations

| Operation | What it needs | Established method | Status |
|---|---|---|---|
| Pick a region | Lagrangian patch selection | zoom-sim region masking (Katz+; Oñorbe+) | solid |
| Extract its IC | phase-consistent sub-field of the parent | MUSIC, Panphasia, Hoffman–Ribak, 2LPT-IC | **solid (exact at linear order)** |
| Carry large-scale forces | far-field / long-range potential each step | COLA/sCOLA LPT far-field; TreePM long-range | solid (approx in time) |
| External tidal field on the box | anisotropic background expansion | separate-universe (Sirko; Schmidt+18; Stücker+21) | **approximate (uniform tide only)** |
| Avoid border effects | buffer/overlap + isolated BC | sCOLA buffer + exact LPT boundary | solid (cost ∝ buffer) |
| Run the mini-sim | fast small-box solver | COLA / FastPM / PM | solid |
| Merge back | stitch inner regions | field-level blending in overlap | **approximate (no exact N-body merge)** |
| Update large scales | small→large back-reaction | — (EFT stochastic term) | **impossible (deterministically)** |

The orchestration is the *composition* of the first seven rows. The eighth
row is the ambition that must be dropped or heavily qualified.

---

## 2. This is not new — the method map

**COLA** (Tassev, Zaldarriaga & Eisenstein 2013, JCAP, arXiv:1301.0322).
Split the displacement into an analytic **2LPT large-scale** part and a
**PM-computed small-scale residual**, integrated in a frame *comoving with
the LPT trajectory*. Large scales are exact *by construction*; the PM only
has to get the near field right. ~10³× faster than full N-body while
matching large-scale observables to a few %. → This is exactly "large-scale
forces handled cheaply on the full volume, mini-sim solves the rest."

**sCOLA** (Tassev & Eisenstein 2015, arXiv:1502.07751; parallel/tiled
version: Leclercq et al. 2020, arXiv:2003.04925). Extends COLA to the
**spatial domain**: tile the volume into sub-boxes, **evolve each tile
independently**, capturing the **far field perturbatively (LPT)** and
letting the N-body solve only the **near field**, with far/near *decoupled
to localise gravity*. The known failure mode is the **boundary condition**;
the fixes are a **buffer region around each tile** + **exact LPT boundary
conditions**. → **This is precisely the user's proposal.** The "mini-sim
with overlap to avoid border effects" is the sCOLA buffer.

**Phase-consistent zoom ICs.** A resimulation IC must share the parent's
large-scale modes exactly. Achieved by Fourier padding, the **Hoffman–Ribak**
constrained-realisation algorithm (Hoffman & Ribak 1991; constrained peaks
van de Weygaert & Bertschinger 1996, arXiv:astro-ph/9507024), **MUSIC**
(Hahn & Abel 2011, MNRAS 415, 2101), **Panphasia** (Jenkins 2013,
arXiv:1306.5968 — an octree phase field spanning >10¹⁵ in scale, so any
sub-region's phases are *by construction* consistent with the parent and
with each other), **2LPT resimulation ICs** (Jenkins 2010, MNRAS 403, 1859),
and **GenetIC** (Stopyra et al. 2021, ApJS, "genetically modified" zooms).
→ Extracting "a specific IC linked to a region" is a **solved problem**.

**External tidal field on a finite box.** A small periodic box cannot host
super-box modes. The **separate-universe** trick absorbs a *uniform*
overdensity as a modified background (Sirko 2005; Gnedin, Kravtsov & Rudd
2011). The **anisotropic** generalisation puts an external *tidal tensor*
into the integrator as an **anisotropic expansion factor A_ij** (Schmidt et
al. 2018, MNRAS 479, 162, arXiv:1803.03274; Stücker et al. 2021, MNRAS 503,
1473, arXiv:2003.06427, TreePM). → The large-scale tidal field *can* be
imposed, but only its **uniform (box-averaged) part**, and it **requires
modifying the time integrator**.

**Fast small-box solvers.** COLA, **FastPM** (Feng et al. 2016, MNRAS 463,
2273 — a PM with modified kick/drift operators enforcing the correct linear
growth in few steps), **L-PICOLA** (Howlett et al. 2015, Astronomy &
Computing). → The mini-sim solver is off-the-shelf physics; a vanilla
CIC+FFT-Poisson+leapfrog PM (optionally COLA-framed) suffices for a dummy.

---

## 3. The user's intuition about AMR / force-splitting is correct

The user noted AMR/zoom "contains similar ideas regarding scales." It does,
and the link is precise: **every fast gravity solver already splits the
force by scale.** TreePM/P³M compute the **long-range force on a global FFT
mesh** and the **short-range force locally**. AMR/nested-zoom refine a
**coarse global potential** with **local fine grids**. COLA splits **LPT
far-field** from **PM near-field**. All three are the *same idea*: a global,
slowly-varying, cheaply-computed **long-range potential**, plus a local,
expensive **short-range correction**.

**Design consequence for OUF-Sim:** the full-sim's **global potential mesh
φ(x; a)** (the `gr_fields` product, ∇²φ = δ, already in the S5 plan) is
*the* long-range-force provider. Every mini-sim consumes the same φ as its
far field and computes only the residual. The infrastructure for "large
scale forces computed on the full volume" = storing and serving φ(x; a) at a
few scale factors. The full-sim therefore **does not need many steps** — in
the COLA/linear limit the far field is analytic.

---

## 4. The verification protocol (the user's test, formalised)

The user proposes: *before running the mini-sim, the large-scale density
profile of the mini-sim and the full-sim on the same volume should match.*
This is exactly right, and it splits into a **necessary** and a
**sufficient** test.

**Gate 1 — pre-run IC consistency (necessary).** Smooth the mini-sim IC
density and the full-sim density over the sub-volume to a scale ≳ the
inter-tile scale; require
- cross-correlation `r(k) → 1` for `k < k_buffer`,
- power ratio `P_mini(k)/P_full(k) ∈ [1−ε, 1+ε]` at low k.
If phases are inherited correctly this is **automatic** — it is a *unit test
on the extraction*, not a physics result. Failing it means the IC linkage is
wrong.

**Gate 2 — post-run dynamical consistency (sufficient).** After evolving the
mini-sim, compare its **evolved** large-scale modes to the full-sim's on the
**buffer-trimmed inner region**:
- `r(k)` and `P_mini/P_full` at low k after evolution,
- displacement-field cross-correlation (Zel'dovich/2LPT residual).
This tests the **dynamical boundary treatment** (the hard part), not just
the IC. **This is the real feasibility verdict.**

**Gate 3 — convergence / error budget.** Sweep:
- error vs **buffer width** (expect convergence once buffer ≳ a few × rms
  displacement),
- error vs **tidal treatment** (none / uniform separate-universe /
  anisotropic),
- error vs **mini-sim step count**.
Report the buffer needed for a target large-scale accuracy. From the
literature: rms Zel'dovich displacement ≈ 6–10 Mpc/h at z=0, COLA large-scale
P(k) good to a few %, sCOLA buffers of order tens of Mpc/h → expect
**buffer ≳ 15–20 Mpc/h** for %-level large-scale accuracy, so a useful
mini-box must be ≫ that.

---

## 5. Critical assessment — achievable / approximate / impossible

**Achievable (solid).**
- Region IC extraction with exact large-scale phase consistency (linear
  order is *exact*; this is the strong, reliable part).
- Large-scale-correct evolution via the COLA frame (large scales guaranteed
  by construction).
- Buffered tiling to suppress boundary artifacts.
- The pre-run consistency gate (it is a construction property).

**Approximate (works, with quantifiable error).**
- **External tidal field**: only the **uniform** part of the tidal tensor is
  cleanly includable (anisotropic separate universe); tidal **gradients**
  across the box are dropped. Error grows with box size / tidal coherence
  length. Requires integrator modification.
- **Isolated boundary conditions**: a periodic mini-box injects spurious
  periodic self-images; need open BC (zero-padded/James-method FFT) or the
  COLA frame. Extra cost, residual error.
- **Merging nonlinear sub-volumes**: density/potential **fields** can be
  blended in the overlap; the **particle/N-body state** cannot be exactly
  reconciled across the splice (no conservation guarantee). Acceptable for
  field-level products, lossy for particle-level.
- **Buffer cost**: the buffer must be ≳ the displacement coherence length;
  near clusters/strong tides this grows, so the "small" mini-sim is not
  always small. This sets the economic floor of the method.

**Impossible (fundamental — be blunt).**
- **Updating the large-scale field from mini-sims alone.** Nonlinear gravity
  is not separable across scales. The EFT of LSS shows the small-scale
  back-reaction onto large scales enters as a **stochastic term + counter-
  terms** that are *not* a deterministic function of the resolved field.
  You can compute the **response** of small scales to fixed large scales
  (separate universe), but you **cannot invert it** to deterministically
  correct the large scales without evolving them (i.e. without the full-sim).
- **Exact reconstruction of the global nonlinear state** by tiling
  independent mini-sims. Gravity is long-range; a tile's trajectory depends
  formally on the entire mass distribution at all times. Truncation to a
  buffer is *always* an approximation; the error is irreducible, only
  controllable by buffer size.

**Verdict.** Build it — but scope it honestly. The deliverable is a
**large-scale-consistent local zoom engine** with a measured error budget,
not a full-sim replacement and not a large-scale updater. Every claim must
be backed by Gate 2/3 numbers. If Gate 2 fails the tolerance even at large
buffers, the honest outcome is "this region/regime is not resimulable to
target accuracy" — and that negative result is itself a deliverable.

---

## 6. Why the dummy stack is the *ideal* testbed

A subtle but important point: using the **linear sim as the full-sim** makes
the feasibility test **cleaner**, not weaker.

- The linear/2LPT full-sim provides the COLA far-field **analytically and
  exactly** → Gate 1 is exact by construction, so any Gate-1 failure is
  purely an extraction bug.
- The **fast-PM mini-sim** adds the *only* new physics (near-field
  nonlinearity + boundary treatment) → Gate 2 isolates **exactly the
  approximation we care about** (the dynamical boundary), with no confounding
  from an approximate full-sim.
- It is cheap enough to run **many** mini-sims and do the Gate-3 convergence
  sweep.

So the staged tooling the user proposed — *linear full-sim + fast-PM mini-sim
+ orchestrator + the large-scale-match verification* — is not a toy
shortcut; it is the **methodologically correct minimal experiment** to
measure resimulation feasibility before committing to real codes.

---

## 7. Risks / open questions to settle empirically

1. How large a buffer for %-level Gate-2 accuracy as a function of region
   nonlinearity (void vs cluster)? (Expect strong dependence.)
2. Does the uniform-tide (separate-universe) approximation suffice, or are
   tidal gradients across realistic mini-boxes already %-level errors?
3. Field-level merge artifacts in the overlap — magnitude and mitigation
   (feathering window, conservation-corrected blend)?
4. Does COLA-frame double-counting of the long-range force (full-sim φ *and*
   the mini-sim's own PM) introduce a bias if not carefully subtracted?
5. At what point does the buffer + far-field bookkeeping cost approach just
   running a coarse global PM — i.e. when is resimulation *not* worth it?

---

## 8. Key references (grounded)

- Tassev, Zaldarriaga & Eisenstein 2013, *Solving LSS in ten easy steps with
  COLA*, JCAP 06, 036 — arXiv:1301.0322.
- Tassev & Eisenstein 2015, *sCOLA: COLA extended to the spatial domain* —
  arXiv:1502.07751.
- Leclercq et al. 2020, *Perfectly parallel cosmological simulations using
  spatial COLA* — arXiv:2003.04925.
- Schmidt et al. 2018, *Cosmological N-body simulations with a large-scale
  tidal field*, MNRAS 479, 162 — arXiv:1803.03274.
- Stücker et al. 2021, *Anisotropic separate-universe simulations using
  TreePM*, MNRAS 503, 1473 — arXiv:2003.06427.
- Jenkins 2010, *2LPT initial conditions for resimulations*, MNRAS 403, 1859.
- Jenkins 2013, *Panphasia: multiscale Gaussian phases* — arXiv:1306.5968.
- van de Weygaert & Bertschinger 1996; Hoffman & Ribak 1991 (constrained
  realisations) — e.g. arXiv:astro-ph/9507024.
- Hahn & Abel 2011, *MUSIC*, MNRAS 415, 2101.
- Stopyra et al. 2021, *GenetIC*, ApJS 252, 28.
- Feng et al. 2016, *FastPM*, MNRAS 463, 2273.
- Howlett et al. 2015, *L-PICOLA*, Astronomy & Computing 12, 109.

*(Reference list synthesised from domain knowledge + targeted literature
search 2026-06-02; arXiv ids confirmed for the COLA/sCOLA/tidal/Panphasia
entries via search, others by venue.)*
