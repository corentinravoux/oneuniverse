# R6 — reference engine and twin contracts

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-author the toy physics as `oneuniverse.reference` (the one reference plugin of the substrate ADR) with the four verified defects fixed and ensemble-validated, and rebuild `oneuniverse.twin` as contracts only (engine ABCs + registry, `observe_from_view`, `run_engine_to_store`). Dead ends are gone.

**Architecture:** `reference/` depends on `simulation` (specs, store) and numpy only; `twin/engine.py` depends on nothing physical; `reference/engines.py` registers the four reference engines on the twin contract; `reference/fieldstats.py` is the single P(k)/r(k) implementation. Ensemble tests (5 seeds, stated tolerances) are `slow`.

**Tech Stack:** numpy FFT, scipy (`integrate.quad`, `cKDTree`), pyarrow, healpy.

**Spec:** design §6.9, §8 items 1–2, §9b (defect list with tolerances), audit `audit_measure_sim_twin.md` §5–§6. Master plan Global Constraints (`simulation` and `twin/engine.py` import nothing from `reference`).

## Interfaces consumed (R5)

`CosmologySpec`, `UnitFrameSpec`, `ProvenanceSpec`, `ProductDecl`, `OUFSimManifest`, `SimConverter`, `register`, `NativeProduct`, `build_store`, `SimStore`, `ingest_field`, `ExecutionPlan`, `Cube`, `NativeReaderAdapter`, `register_adapter`.

## Interfaces produced

```
oneuniverse.reference.linear
  eh98_transfer(k, cosmo) ; linear_power(k, cosmo, z=0.0) -> P(k) [(Mpc/h)³], sigma8-normalised (normalisation cached per cosmo)
  growth_factor(z, cosmo) -> float (D(z)/D(0), ODE integral, flat ΛCDM) ; growth_rate(z, cosmo) -> float (f = dlnD/dlna from the same integral, NOT γ=0.55)
  k_grid(n_grid, box) -> (kx, ky, kz, kmag) ; gaussian_field(cosmo, *, box, n_grid, z=0.0, seed) -> ndarray (n³, float64), mean 0, variance matches mode sum
  zeldovich(delta_z0, cosmo, *, box, z) -> (pos (n³,3) [Mpc/h], vel (n³,3) [km/s])    # pos = q + D(z) ψ ; vel = a H(a) f ψ = 100 a E(a) f(z) D(z) ψ  (ψ in Mpc/h)
  find_peaks(delta, *, box, threshold=2.0) -> pa.Table (x, y, z, height, mass_proxy)
oneuniverse.reference.pm
  deposit_cic(pos, n_grid, box) -> ndarray (n_grid³ counts, cells centred at (i+0.5)·cell, np.bincount)
  interpolate_cic(grid, pos, box) -> ndarray ; pm_force(delta, box) -> (fx, fy, fz) grids ; pm_force_isolated(delta, box) -> same with zero-padded open Poisson
  run_pm(pos, vel, cosmo, *, box, n_grid, z_start, z_end, n_steps, force="periodic"|"isolated", snapshots: Sequence[float] = ()) -> dict {"pos", "vel", "snapshots": {z: delta}}
  zeldovich_pm_ic(cosmo, *, box, n_grid, z_start, seed) -> (pos, vel, ic_field) ; zeldovich_pm_ic_from_field(cosmo, delta_z0, *, box, z_start) -> (pos, vel)   # growing-mode momentum
oneuniverse.reference.resim
  run_full_reference(cosmo, *, box, n_grid, z_start, z_end, n_steps, seed) -> (delta_full (n_grid³ at z_end), ic_field) ; run_full_reference_from_ic(cosmo, ic_field, *, box, n_grid, z_start, z_end, n_steps) -> (delta_full, ic_field)
  extract_buffered(ic_field: ndarray, *, box, cube: Cube, buffer: float) -> (sub_ic, sub_box)       # PERIODIC: uses np.take with mode="wrap" on index ranges; sub-grid may cross the box boundary
  run_buffered(cosmo, ic_field, *, box, cube, buffer, z_start, z_end, n_steps, resolution_factor=1) -> dict {"inner": delta (inner cube cells), "n_particles": int, "sub_box": float}
  run_treepm(cosmo, ic_field, *, box, cube, buffer, z_start, z_end, n_steps, k_split=None) -> dict (same keys + "k_split")   # long-range = low-pass of the full-box linear force × D(a); short-range = high-pass tile PM force
  inner_reference(delta_full, *, box, cube) -> ndarray ; buffer_convergence(cosmo, ic_field, *, box, cube, buffers: Sequence[float], method="buffered"|"treepm", **run_kw) -> pa.Table (buffer, cell_corr, r_lowk)
oneuniverse.reference.reconstruct
  wiener_reconstruct(delta_g, cosmo, *, box, nbar, bias=1.0, z=0.0, mask: ndarray | None = None) -> ndarray   # mask-aware: delta_g renormalised by the mask mean before filtering; outside the mask the WF mean is 0
  constrained_realization(delta_g, cosmo, *, box, nbar, bias, z, seed: int, mask=None) -> ndarray   # seed REQUIRED (no default); Hoffman–Ribak: δ_CR = δ_WF(δ_g) + (δ_rand − δ_WF(mock(δ_rand)))
oneuniverse.reference.fieldstats
  bin_edges_kf(n_grid, box) -> ndarray (fundamental-mode-spaced to Nyquist)   # the ONE binning
  power_spectrum(delta, *, box) -> (k, P) ; cross_power(a, b, *, box) -> (k, P_ab) ; cross_r(a, b, *, box) -> (k, r) ; transfer(rec, truth, *, box) -> (k, T) ; k_half(k, r, level=0.5) -> float
  cell_corr(a, b) -> float (Pearson correlation of flattened cells)
  @dataclass RecoveryMetrics(k, r, transfer, power_ratio, k_half) ; recover_metrics(rec, truth, *, box) -> RecoveryMetrics
oneuniverse.reference.mock
  lognormal_tracers(delta, *, box, nbar, bias, seed) -> ndarray (counts grid, Poisson of a lognormal field; mean bias within 5 % of `bias` for σ_cell ≲ 1)
  mask_slab(n_grid, axis, lo_frac, hi_frac) ; mask_ball(n_grid, centre_frac, radius_frac) -> bool grids
  run_mock_challenge(cosmo, *, box, n_grid, nbar, bias=1.5, z=0.0, seed) -> dict {"truth", "delta_g", "wiener", "metrics": RecoveryMetrics}
oneuniverse.reference.native
  generate_linear_sim(out: Path, cosmo, *, box, n_grid, redshifts, seed) -> Path   # config.json + particles_z<tag>.npy + field_z<tag>.npy + halos_z<tag>.parquet + lightcone.parquet (sky partition fixture, columns ra, dec, z_shell, radius; documented as NOT a physical lightcone)
  write_packed_native(linear_dir, out_dir, *, n_side) -> Path ; PackedNpyAdapter is registered by simulation.store.native (R5)
  @register class LinearSimConverter(SimConverter): code = "linear"  ; @register class PackedSimConverter(SimConverter): code = "packed_npy"
oneuniverse.reference.engines
  @register_engine LinearForwardEngine (name "linear") ; PMForwardEngine ("fastpm") ; WienerReconstruction ("wiener") ; ConstrainedRealization ("constrained", takes seed in __init__)
oneuniverse.twin.engine
  @dataclass(frozen=True) Observation(delta_g: ndarray, nbar: float, bias: float, mask: ndarray | None = None, box: float = 0.0, meta: dict = {})
  @dataclass ProductBundle(fields: dict[str, ndarray], z: float, provenance: dict)
  class ReconstructionEngine(ABC): name: ClassVar[str] ; reconstruct(observation, *, cosmo, z=0.0) -> ndarray
  class ForwardEngine(ABC): name: ClassVar[str] ; forward(*, cosmo, box, n_grid, z, seed=None, ic_field=None) -> ProductBundle
  register_engine(cls) ; get_engine(name) -> type ; registered_engines() -> tuple[str, ...]
oneuniverse.twin.observe
  observe_from_view(source: DatasetView | Product, *, box, n_grid, position_columns=("x", "y", "z"), nbar=None, bias=1.0, mask=None) -> Observation   # grids CIC counts into delta_g = n/⟨n⟩_mask − 1; positions must already be box coordinates (no sky→comoving here)
oneuniverse.twin.store_boundary
  run_engine_to_store(engine: ForwardEngine, *, store_root, name, cosmo, box, n_grid, z, ic_from: tuple[SimStore, str, float] | None = None, seed=None, plan=None) -> SimStore   # ic read via SimStore.read_field_box over the whole box ONLY through scan-like explicit call; documented
oneuniverse.twin (__init__) exports: Observation, ProductBundle, ForwardEngine, ReconstructionEngine, register_engine, get_engine, registered_engines, observe_from_view, run_engine_to_store
```

Tolerances for ensemble tests (seeds `[0, 1, 2, 3, 4]`, assert on the ensemble mean unless stated):

| Test | Quantity | Tolerance |
|---|---|---|
| Zel'dovich velocity | ratio of `div(v)` power to `(a H f δ)²` power at k < 0.1 h/Mpc | within 2 % |
| PM growth | `P_pm / P_lin` at k < 0.05 h/Mpc, z=9→0, 64³, 20 steps | within 5 %; `r > 0.98` |
| Periodic buffer | `n_particles == n_buf³` for `cube.lo < buffer` (every seed) and `|cell_corr(edge) − cell_corr(centre)| < 0.03` | exact / 0.03 |
| CR seed independence | `cross_r(CR(seed=s'), truth)` at k > 0.5 k_Nyq < 0.5 for s' ≠ s; low-k `cross_r(CR, WF) > 0.95` | both |
| Masked Wiener | `cross_r(WF_masked, truth)` inside a 50 % slab vs unmasked: `|Δr| < 0.05` at k < 0.2 h/Mpc | 0.05 |
| TreePM gain | `cell_corr(treepm, buffer=8) ≥ cell_corr(buffered, buffer=8) + 0.1` at box 128, n_grid 64 | 0.1 |

---

### Task 1: `twin.engine` (contracts first, no physics)

**Files:**
- Create: `oneuniverse/twin/__init__.py`, `oneuniverse/twin/engine.py`
- Test: `test/twin/__init__.py`, `test/twin/test_engine.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_registry_and_abstracts():
    @register_engine
    class Dummy(ForwardEngine):
        name = "dummy"
        def forward(self, *, cosmo, box, n_grid, z, seed=None, ic_field=None): return ProductBundle({"delta": np.zeros((n_grid,) * 3)}, z, {"engine": "dummy"})
    assert get_engine("dummy") is Dummy and "dummy" in registered_engines()
    with pytest.raises(ValueError): register_engine(Dummy)
    with pytest.raises(TypeError): ForwardEngine()
    obs = Observation(np.zeros((4, 4, 4)), nbar=1e-3, bias=1.0); assert obs.mask is None
```

- [ ] **Step 2–5:** Run (fail) → implement → run (pass) → commit `feat(twin): engine contracts and registry`.

---

### Task 2: `reference.linear` and `reference.fieldstats`

**Files:**
- Create: `oneuniverse/reference/__init__.py`, `linear.py`, `fieldstats.py`
- Test: `test/reference/__init__.py`, `test/reference/test_linear.py`, `test/reference/test_fieldstats.py`

- [ ] **Step 1: Write the failing tests**

```python
COSMO = CosmologySpec(0.6736, 0.3153, 0.0493, 0.8111, 0.9649)

def test_linear_power_sigma8_normalised():
    k = np.logspace(-4, 2, 2000); P = linear_power(k, COSMO); W = lambda x: 3 * (np.sin(x) - x * np.cos(x)) / x**3
    s8 = np.sqrt(np.trapezoid(k**2 * P * W(k * 8.0)**2, k) / (2 * np.pi**2)); assert s8 == pytest.approx(0.8111, rel=1e-3)

def test_growth_consistent_f_and_D():
    z = 0.5; eps = 1e-3; a = 1 / (1 + z); Dp = growth_factor(1 / (a * (1 + eps)) - 1, COSMO); Dm = growth_factor(1 / (a * (1 - eps)) - 1, COSMO)
    f_num = (np.log(Dp) - np.log(Dm)) / (2 * eps); assert growth_rate(z, COSMO) == pytest.approx(f_num, rel=1e-3) and growth_factor(0.0, COSMO) == 1.0

def test_gaussian_field_variance_matches_power():
    box, n = 500.0, 64; d = gaussian_field(COSMO, box=box, n_grid=n, seed=0); k, P = power_spectrum(d, box=box); kk = k[(k > 0.05) & (k < 0.3)]
    assert abs(d.mean()) < 1e-10 and np.allclose(P[(k > 0.05) & (k < 0.3)] / linear_power(kk, COSMO), 1.0, atol=0.25)

@pytest.mark.slow
def test_zeldovich_velocity_units_ensemble():
    box, n, z = 500.0, 64, 0.5; a = 1 / (1 + z); E = np.sqrt(COSMO.omega_m * a**-3 + 1 - COSMO.omega_m); f = growth_rate(z, COSMO); D = growth_factor(z, COSMO)
    ratios = []
    for seed in range(5):
        d0 = gaussian_field(COSMO, box=box, n_grid=n, seed=seed); pos, vel = zeldovich(d0, COSMO, box=box, z=z)
        # divergence of the Lagrangian velocity field vs −a H f δ(z): compare power at k<0.1 using the displacement-field divergence computed on the grid
        div_v = divergence_on_grid(vel.reshape(n, n, n, 3), box); k, Pv = power_spectrum(div_v, box=box); k2, Pd = power_spectrum(-(100 * a * E * f) * D * d0, box=box)
        sel = k < 0.1; ratios.append(np.mean(Pv[sel] / Pd[sel]))
    assert abs(np.mean(ratios) - 1.0) < 0.02

def test_fieldstats_binning_and_r_of_identical_fields():
    d = np.random.default_rng(0).normal(size=(32, 32, 32)); k, r = cross_r(d, d, box=100.0); assert np.allclose(r, 1.0) and k[0] > 0 and np.all(np.diff(k) > 0)
    assert k_half(np.array([0.1, 0.2, 0.3]), np.array([0.9, 0.5, 0.1])) == pytest.approx(0.2)
    m = recover_metrics(d, d, box=100.0); assert np.allclose(m.transfer, 1.0) and np.allclose(m.power_ratio, 1.0)
```

`divergence_on_grid(v, box)` is a test helper in `test/reference/_helpers.py` (FFT divergence; the builder writes it).

- [ ] **Step 2–5:** Run (fail) → implement → run (pass, run the slow test once) → commit `feat(reference): linear theory with consistent growth and correct Zel'dovich velocities; single fieldstats`.

---

### Task 3: `reference.pm`

**Files:**
- Create: `oneuniverse/reference/pm.py`
- Test: `test/reference/test_pm.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_cic_deposit_conserves_mass_and_is_centred():
    pos = np.array([[0.5 * (100 / 16)] * 3]); g = deposit_cic(pos, 16, 100.0); assert g.sum() == pytest.approx(1.0) and g[0, 0, 0] == pytest.approx(1.0)   # a particle at the centre of cell 0 lands entirely in cell 0
    pos2 = np.random.default_rng(0).uniform(0, 100, (10_000, 3)); assert deposit_cic(pos2, 16, 100.0).sum() == pytest.approx(10_000)

def test_pm_force_is_gradient_of_potential_for_single_mode():
    n, box = 32, 100.0; x = (np.arange(n) + 0.5) * box / n; d = np.sin(2 * np.pi * x / box)[:, None, None] * np.ones((1, n, n)); fx, fy, fz = pm_force(d, box)
    kf = 2 * np.pi / box; assert np.allclose(fx, -np.cos(2 * np.pi * x / box)[:, None, None] / kf, rtol=1e-6) and np.allclose(fy, 0) and np.allclose(fz, 0)

@pytest.mark.slow
def test_pm_linear_growth_ensemble():
    box, n = 256.0, 64; ratios, rs = [], []
    for seed in range(5):
        pos, vel, ic = zeldovich_pm_ic(COSMO, box=box, n_grid=n, z_start=9.0, seed=seed); out = run_pm(pos, vel, COSMO, box=box, n_grid=n, z_start=9.0, z_end=0.0, n_steps=20)
        d = deposit_cic(out["pos"], n, box); d = d / d.mean() - 1; k, P = power_spectrum(d, box=box); sel = k < 0.05
        ratios.append(np.mean(P[sel] / linear_power(k[sel], COSMO))); rs.append(np.mean(cross_r(d, ic, box=box)[1][sel]))
    assert abs(np.mean(ratios) - 1.0) < 0.05 and np.mean(rs) > 0.98
```

- [ ] **Step 2–5:** Run (fail) → implement (KDK leapfrog in `a`; kick/drift factors by `quad`; growing-mode momentum `p = a² E f D ψ`-consistent with `zeldovich`; `np.bincount` CIC) → run (pass) → commit `feat(reference): fast particle-mesh with bincount CIC`.

---

### Task 4: `reference.resim` (periodic buffers, TreePM)

**Files:**
- Create: `oneuniverse/reference/resim.py`
- Test: `test/reference/test_resim.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_extract_buffered_wraps_periodically():
    ic = np.arange(64**3, dtype=float).reshape(64, 64, 64); sub, sub_box = extract_buffered(ic, box=128.0, cube=Cube((0, 0, 0), (32, 32, 32)), buffer=16.0)
    assert sub.shape == (32, 32, 32) and sub_box == 64.0 and sub[0, 0, 0] == ic[-8, -8, -8] and sub[8, 8, 8] == ic[0, 0, 0]

@pytest.mark.slow
def test_run_buffered_edge_equals_centre_ensemble():
    box, n = 128.0, 64; diffs = []
    for seed in range(5):
        ic = gaussian_field(COSMO, box=box, n_grid=n, seed=seed)
        edge = run_buffered(COSMO, ic, box=box, cube=Cube((0, 0, 0), (32, 32, 32)), buffer=16.0, z_start=9.0, z_end=0.0, n_steps=10)
        cen = run_buffered(COSMO, ic, box=box, cube=Cube((48, 48, 48), (80, 80, 80)), buffer=16.0, z_start=9.0, z_end=0.0, n_steps=10)
        assert edge["n_particles"] == cen["n_particles"] == 32**3
        full, _ = run_full_reference_from_ic(COSMO, ic, box=box, n_grid=n, z_start=9.0, z_end=0.0, n_steps=10)   # helper exported by resim
        c_e = cell_corr(edge["inner"], inner_reference(full, box=box, cube=Cube((0, 0, 0), (32, 32, 32)))); c_c = cell_corr(cen["inner"], inner_reference(full, box=box, cube=Cube((48, 48, 48), (80, 80, 80))))
        diffs.append(abs(c_e - c_c))
    assert np.mean(diffs) < 0.03

@pytest.mark.slow
def test_treepm_beats_buffered_at_small_buffer():
    box, n = 128.0, 64; gains = []
    for seed in range(3):
        ic = gaussian_field(COSMO, box=box, n_grid=n, seed=seed); cube = Cube((32, 32, 32), (64, 64, 64)); kw = dict(box=box, cube=cube, buffer=8.0, z_start=9.0, z_end=0.0, n_steps=10)
        full, _ = run_full_reference_from_ic(COSMO, ic, box=box, n_grid=n, z_start=9.0, z_end=0.0, n_steps=10); ref = inner_reference(full, box=box, cube=cube)
        gains.append(cell_corr(run_treepm(COSMO, ic, **kw)["inner"], ref) - cell_corr(run_buffered(COSMO, ic, **kw)["inner"], ref))
    assert np.mean(gains) >= 0.1

def test_buffer_convergence_table_shape():
    ic = gaussian_field(COSMO, box=64.0, n_grid=32, seed=0); t = buffer_convergence(COSMO, ic, box=64.0, cube=Cube((16, 16, 16), (32, 32, 32)), buffers=[4.0, 8.0], method="buffered", z_start=9.0, z_end=0.0, n_steps=4)
    assert t.column_names == ["buffer", "cell_corr", "r_lowk"] and t.num_rows == 2
```

- [ ] **Step 2–5:** Run (fail) → implement (port `main:oneuniverse/simulation/resim/{coupling,treepm,bench}.py` logic with periodic `np.take(..., mode="wrap")` buffers; TreePM split scale default `k_split = 2 × 2π/sub_box`) → run (pass; slow ones once) → commit `feat(reference): periodic buffered resimulation and TreePM coupling`.

---

### Task 5: `reference.reconstruct` and `reference.mock`

**Files:**
- Create: `oneuniverse/reference/reconstruct.py`, `oneuniverse/reference/mock.py`
- Test: `test/reference/test_reconstruct.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_constrained_requires_seed():
    with pytest.raises(TypeError): constrained_realization(np.zeros((8, 8, 8)), COSMO, box=100.0, nbar=1e-3, bias=1.0, z=0.0)

@pytest.mark.slow
def test_constrained_realization_independent_of_truth_seed():
    box, n = 256.0, 64; hi, lo = [], []
    for seed in range(5):
        truth = gaussian_field(COSMO, box=box, n_grid=n, seed=seed); dg = lognormal_tracers(truth, box=box, nbar=5e-3, bias=1.5, seed=100 + seed); dg = dg / dg.mean() - 1
        wf = wiener_reconstruct(dg, COSMO, box=box, nbar=5e-3, bias=1.5); cr = constrained_realization(dg, COSMO, box=box, nbar=5e-3, bias=1.5, z=0.0, seed=seed + 1000)
        k, r_t = cross_r(cr, truth, box=box); _, r_w = cross_r(cr, wf, box=box); knyq = np.pi * n / box
        hi.append(np.mean(r_t[k > 0.5 * knyq])); lo.append(np.mean(r_w[k < 0.1]))
    assert np.mean(hi) < 0.5 and np.mean(lo) > 0.95

@pytest.mark.slow
def test_masked_wiener_recovers_inside_footprint():
    box, n = 256.0, 64; diffs = []
    for seed in range(5):
        truth = gaussian_field(COSMO, box=box, n_grid=n, seed=seed); dg = lognormal_tracers(truth, box=box, nbar=5e-3, bias=1.5, seed=200 + seed); dg = dg / dg.mean() - 1
        mask = mask_slab(n, 0, 0.0, 0.5); wf_m = wiener_reconstruct(dg * mask, COSMO, box=box, nbar=5e-3, bias=1.5, mask=mask); wf = wiener_reconstruct(dg, COSMO, box=box, nbar=5e-3, bias=1.5)
        k, r_m = cross_r(wf_m * mask, truth * mask, box=box); _, r = cross_r(wf * mask, truth * mask, box=box); sel = k < 0.2; diffs.append(np.mean(np.abs(r_m[sel] - r[sel])))
    assert np.mean(diffs) < 0.05

def test_mock_challenge_returns_metrics():
    out = run_mock_challenge(COSMO, box=128.0, n_grid=32, nbar=5e-3, seed=0); assert set(out) == {"truth", "delta_g", "wiener", "metrics"} and 0 < out["metrics"].k_half
```

- [ ] **Step 2–5:** Run (fail) → implement → run (pass) → commit `feat(reference): mask-aware Wiener, seed-explicit constrained realisation, lognormal mock`.

---

### Task 6: `reference.native` (fixture sim, packed layout, two converters)

**Files:**
- Create: `oneuniverse/reference/native.py`
- Test: `test/reference/test_native.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_generate_and_convert_linear(tmp_path):
    src = generate_linear_sim(tmp_path / "lin", COSMO, box=128.0, n_grid=32, redshifts=(1.0, 0.0), seed=0)
    assert detect_converter(src).code == "linear"; st = get_converter("linear")().convert(src, tmp_path / "store", plan=ExecutionPlan())
    assert set(st.products) >= {"particles", "field", "halos", "lightcone"} and st.manifest.cosmology == COSMO and st.manifest.product("lightcone").extra.get("physical") is False

def test_packed_converter_reads_identically(tmp_path):
    src = generate_linear_sim(tmp_path / "lin", COSMO, box=128.0, n_grid=32, redshifts=(0.0,), seed=1); packed = write_packed_native(src, tmp_path / "packed", n_side=4)
    a = get_converter("linear")().convert(src, tmp_path / "a", plan=ExecutionPlan(), n_side=4); b = get_converter("packed_npy")().convert(packed, tmp_path / "b", plan=ExecutionPlan(), projection="reference", n_side=4)
    cube = Cube((10, 10, 10), (50, 50, 50)); assert a.read_box("particles", 0.0, cube).sort_by("x").equals(b.read_box("particles", 0.0, cube).sort_by("x"))
```

- [ ] **Step 2–5:** Run (fail) → implement (port from `main:oneuniverse/simulation/linear/{generate,pack,converter}.py` and `packed/converter.py`, using R5 writers) → run (pass) → commit `feat(reference): linear fixture simulation, packed layout and converters`.

---

### Task 7: `reference.engines`, `twin.observe`, `twin.store_boundary`

**Files:**
- Create: `oneuniverse/reference/engines.py`, `oneuniverse/twin/observe.py`, `oneuniverse/twin/store_boundary.py`
- Test: `test/twin/test_observe_and_store.py`, `test/reference/test_engines.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_four_engines_registered():
    import oneuniverse.reference.engines  # noqa
    assert {"linear", "fastpm", "wiener", "constrained"} <= set(registered_engines())
    b = get_engine("linear")().forward(cosmo=COSMO, box=64.0, n_grid=16, z=0.0, seed=0); assert b.fields["delta"].shape == (16, 16, 16)

def test_observe_from_view_grid_and_mask(tmp_path):
    # synthetic OUF with box columns x, y, z (no sky->comoving conversion in the package)
    tab = synthetic_point_table(20_000, 0); rng = np.random.default_rng(0); tab = tab.append_column("x", pa.array(rng.uniform(0, 100, 20_000))).append_column("y", pa.array(rng.uniform(0, 100, 20_000))).append_column("zb", pa.array(rng.uniform(0, 100, 20_000)))
    write_dataset(tab, tmp_path / "ds", geometry=Geometry.POINT, survey=SurveyMeta("s", "test"), loader=LoaderInfo("t", "1", {})); v = DatasetView.from_path(tmp_path / "ds")
    obs = observe_from_view(v, box=100.0, n_grid=16, position_columns=("x", "y", "zb"), bias=1.2); assert obs.delta_g.shape == (16,) * 3 and abs(obs.delta_g.mean()) < 1e-10 and obs.nbar == pytest.approx(20_000 / 100.0**3)
    mask = mask_slab(16, 0, 0.0, 0.5); obs_m = observe_from_view(v, box=100.0, n_grid=16, position_columns=("x", "y", "zb"), mask=mask); assert abs(obs_m.delta_g[mask].mean()) < 1e-10 and np.all(obs_m.delta_g[~mask] == 0)

def test_closed_loop_recovers_truth(tmp_path):
    truth = gaussian_field(COSMO, box=128.0, n_grid=32, seed=0); counts = lognormal_tracers(truth, box=128.0, nbar=2e-2, bias=1.5, seed=1)
    # fixture: sample positions from counts into a Product catalog with x,y,z extras -> observe_from_view(product)
    prod = product_from_counts(counts, box=128.0, seed=2); obs = observe_from_view(prod, box=128.0, n_grid=32, position_columns=("x", "y", "z"), bias=1.5)
    rec = get_engine("wiener")().reconstruct(obs, cosmo=COSMO); m = recover_metrics(rec, truth, box=128.0); assert np.mean(m.r[m.k < 0.1]) > 0.6

def test_run_engine_to_store_store_to_store(tmp_path):
    st = run_engine_to_store(get_engine("linear")(), store_root=tmp_path / "s", name="lin", cosmo=COSMO, box=64.0, n_grid=16, z=0.0, seed=0)
    st2 = run_engine_to_store(get_engine("fastpm")(), store_root=tmp_path / "s", name="pm", cosmo=COSMO, box=64.0, n_grid=16, z=0.0, ic_from=(st, "delta", 0.0))
    assert "delta" in st2.products and st2.manifest.parent == "lin"
```

`product_from_counts` is a test helper in `test/fixtures/twin_mock.py`.

- [ ] **Step 2–5:** Run (fail) → implement → run (pass) → commit `feat(reference,twin): reference engines on the contract, data→twin socket, store boundary`.

---

### Task 8: Guards, surface, close-out

- [ ] **Step 1:** `test/test_guards.py`: enable the "substrate ⊥ reference" rows (`oneuniverse/simulation`, `oneuniverse/twin/engine.py` import nothing from `oneuniverse.reference`) and the "measure ⊥ reference/twin" row.
- [ ] **Step 2:** `pytest -n 6` (default, < 60 s single-process) and `pytest -m slow -n 6` once (record its wall time in STATUS.md).
- [ ] **Step 3:** STATUS.md + memory; commit + push claude_config. Commit — `docs: R6 close-out`.

**Exit criterion:** the six ensemble tests in the tolerance table pass; store→store engine chain runs; guards green.
