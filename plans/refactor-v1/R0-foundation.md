# R0 — Foundation: branch, packaging, `_core`, guards, CI

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put the shared, science-free helpers and the boundary guards in place on a fresh branch without deleting any old module, so every later step builds on one `_core` and is checked by the same guards.

**Architecture:** `oneuniverse/_core/` holds errors, HEALPix helpers, geometry selectors, atomic filesystem helpers and the existing `Registry`. `test/test_guards.py` scans the package with `ast` and fails on any forbidden import or cosmology identifier. `pyproject.toml` moves to Python ≥ 3.11, drops `pyyaml`, declares extras, and gates `slow` tests.

**Tech Stack:** Python ≥ 3.11, numpy, healpy, pyarrow; pytest markers; GitHub Actions.

**Spec:** `plans/2026-11-12-refactor-v1-design.md` §4, §5 (`_core`), §6.1, §6.10. Master plan: `plans/2026-11-12-refactor-v1-plan.md` (Global Constraints apply).

## Interfaces produced (consumed by every later step)

```
oneuniverse._core.errors:   OneuniverseError(Exception); FormatError, SchemaError, NotFoundError, ValidationError (all subclass OneuniverseError)
oneuniverse._core.healpix:  NSIDE_FINE = 32
                            radec_to_pix(nside: int, ra: ArrayLike, dec: ArrayLike, *, nest: bool = True) -> np.ndarray[int32]
                            coarsen(pix: ArrayLike, nside_from: int, nside_to: int) -> np.ndarray[int32]      # NEST only; nside_to <= nside_from, both powers of 2
                            cone_pixels(nside, ra, dec, radius_deg, *, nest=True) -> np.ndarray[int32]       # inclusive cover
                            patch_pixels(nside, ra_min, ra_max, dec_min, dec_max, *, nest=True) -> np.ndarray[int32]  # inclusive cover, RA wrap aware
oneuniverse._core.geometry: @dataclass(frozen=True, slots=True) Cone(ra: float, dec: float, radius: float)            # degrees
                                .contains(ra, dec) -> np.ndarray[bool]  (haversine)
                            SkyPatch(ra_min, ra_max, dec_min, dec_max)  .wraps -> bool (ra_min > ra_max)  .contains(ra, dec)
                            ZRange(lo: float, hi: float) .contains(z)
                            Cube(lo: tuple[float, float, float], hi: tuple[float, float, float]) .contains(x, y, z) .volume
                            Selection = Cone | SkyPatch
oneuniverse._core.fs:       atomic_write_bytes(path, data: bytes) -> None ; atomic_write_json(path, obj) -> None
                            atomic_dir(path: Path) -> ContextManager[Path]   # yields <path>.tmp; on success replaces <path> (old moved to <path>.old then deleted); on exception removes .tmp, leaves <path> untouched
                            hash_bytes(data: bytes) -> str (16 hex of sha256) ; hash_file(path) -> str
oneuniverse._core.registry: Registry (moved from oneuniverse/_registry.py unchanged; old module becomes `from oneuniverse._core.registry import *` shim until R8)
```

---

### Task 1: Branch and packaging

**Files:**
- Modify: `pyproject.toml`
- Modify: `oneuniverse/__init__.py` (version only)
- Modify: `.github/workflows/tests.yml`

- [ ] **Step 1: Create the branch**

```bash
cd ~/Software/oneuniverse && git checkout -b refactor-v1
git add plans/2026-11-12-refactor-v1-design.md plans/2026-11-12-refactor-v1-plan.md plans/refactor-v1/
git commit -m "docs: refactor v1 design and step plans"
```

- [ ] **Step 2: Edit `pyproject.toml`**

Set: `requires-python = ">=3.11"`; `dependencies = ["numpy>=1.26", "pyarrow>=15.0", "pandas>=2.1", "healpy>=1.16", "scipy>=1.11", "astropy>=6.0"]`; extras `fits = ["fitsio>=1.2"]`, `sql = ["duckdb>=1.0"]`, `mpi = ["mpi4py>=3.1"]`, `dev = ["pytest>=8", "pytest-xdist>=3", "pytest-cov", "matplotlib>=3.8", "duckdb>=1.0", "ruff"]`, `docs = ["mkdocs-material>=9.5", "mkdocstrings[python]>=0.24"]`. Remove `pyyaml`, `mocpy`, `[tool.bumpversion]`. Keep the entry-point group `oneuniverse.survey_loaders`. Pytest: `testpaths = ["test"]`, `addopts = "-q -m 'not slow'"`, `markers = ["slow: ensemble / physics tests, run nightly"]`. Keep `version = {attr = "oneuniverse.__version__"}`; set `__version__ = "1.0.0.dev0"`.

Note: the old `simulation/` still imports `yaml` until R5. Keep `pyyaml` in `dev` only for now (remove in R5 Task 1).

- [ ] **Step 3: Edit CI**

`tests.yml`: matrix `["3.11", "3.12", "3.13", "3.14"]`; test step `pip install -e ".[dev]"` then `pytest -n auto`; add a job `slow` on `schedule: cron "0 3 * * *"` and `workflow_dispatch` running `pytest -m slow -n auto` on 3.12; lint step `ruff check oneuniverse test --select E9,F63,F7,F82` unchanged.

- [ ] **Step 4: Verify the old suite still passes on the branch**

Run: `pip install -e ".[dev]" && pytest -n 6`
Expected: `838 passed, 13 skipped` (same as baseline).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml oneuniverse/__init__.py .github/workflows/tests.yml
git commit -m "chore: python>=3.11, dependency floors, extras, slow marker, CI matrix"
```

---

### Task 2: `_core.errors` and `_core.registry`

**Files:**
- Create: `oneuniverse/_core/__init__.py` (empty docstring only), `oneuniverse/_core/errors.py`, `oneuniverse/_core/registry.py`
- Modify: `oneuniverse/_registry.py` → one-line re-export shim
- Test: `test/core/test_errors_registry.py`, `test/core/__init__.py` (empty)

- [ ] **Step 1: Write the failing tests**

```python
def test_error_hierarchy():
    from oneuniverse._core.errors import OneuniverseError, FormatError, SchemaError, NotFoundError, ValidationError
    for cls in (FormatError, SchemaError, NotFoundError, ValidationError):
        assert issubclass(cls, OneuniverseError)

def test_registry_moved_and_shim_identical():
    from oneuniverse._core.registry import Registry
    import oneuniverse._registry as old
    assert old.Registry is Registry
    r = Registry("thing"); r.register(1, name="a")
    assert r.get("a") == 1 and "a" in r and r.names() == ["a"]
```

- [ ] **Step 2: Run, expect ImportError**

Run: `pytest test/core/test_errors_registry.py -v`

- [ ] **Step 3: Implement**

`errors.py`: the five classes with one-line docstrings. `registry.py`: move the body of `oneuniverse/_registry.py` (git mv), keep behaviour. `oneuniverse/_registry.py` becomes `from oneuniverse._core.registry import Registry  # noqa: F401` plus a comment "shim, removed in R8".

- [ ] **Step 4: Run tests, expect PASS; run the whole suite, expect unchanged**

- [ ] **Step 5: Commit** — `refactor: _core.errors and _core.registry`

---

### Task 3: `_core.fs`

**Files:**
- Create: `oneuniverse/_core/fs.py`
- Test: `test/core/test_fs.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_atomic_write_json_replaces_and_is_readable(tmp_path):
    p = tmp_path / "m.json"; atomic_write_json(p, {"a": 1}); atomic_write_json(p, {"a": 2})
    assert json.loads(p.read_text()) == {"a": 2} and not list(tmp_path.glob("*.tmp*"))

def test_atomic_dir_success_replaces_previous(tmp_path):
    target = tmp_path / "ds"; target.mkdir(); (target / "old").write_text("x")
    with atomic_dir(target) as d:
        (d / "new").write_text("y")
    assert (target / "new").exists() and not (target / "old").exists()
    assert not (tmp_path / "ds.tmp").exists() and not (tmp_path / "ds.old").exists()

def test_atomic_dir_failure_leaves_previous(tmp_path):
    target = tmp_path / "ds"; target.mkdir(); (target / "old").write_text("x")
    with pytest.raises(RuntimeError):
        with atomic_dir(target) as d:
            (d / "new").write_text("y"); raise RuntimeError
    assert (target / "old").exists() and not (target / "new").exists() and not (tmp_path / "ds.tmp").exists()

def test_hash_bytes_is_16_hex():
    assert re.fullmatch(r"[0-9a-f]{16}", hash_bytes(b"abc"))
```

- [ ] **Step 2: Run, expect failure**
- [ ] **Step 3: Implement the five functions** (tmp file + `os.replace`; `fsync` the file before replace; `atomic_dir` refuses if `<path>.tmp` already exists → `FormatError`).
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** — `feat: _core.fs atomic file and directory helpers`

---

### Task 4: `_core.healpix`

**Files:**
- Create: `oneuniverse/_core/healpix.py`
- Test: `test/core/test_healpix.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_radec_to_pix_matches_healpy_nest():
    ra = np.array([0.0, 185.0, 359.9]); dec = np.array([-30.0, 15.0, 89.0])
    exp = hp.ang2pix(32, ra, dec, nest=True, lonlat=True)
    out = radec_to_pix(32, ra, dec)
    assert out.dtype == np.int32 and np.array_equal(out, exp)

def test_coarsen_is_nest_right_shift():
    pix = radec_to_pix(32, np.array([10.0]), np.array([10.0]))
    assert coarsen(pix, 32, 8)[0] == pix[0] >> 4
    with pytest.raises(ValueError): coarsen(pix, 32, 64)

def test_cone_pixels_covers_all_member_points():
    rng = np.random.default_rng(0); n = 5000
    ra = rng.uniform(0, 360, n); dec = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    cone = Cone(ra=0.5, dec=-0.5, radius=3.0); inside = cone.contains(ra, dec)
    cells = cone_pixels(8, cone.ra, cone.dec, cone.radius)
    assert np.all(np.isin(radec_to_pix(8, ra[inside], dec[inside]), cells))

def test_patch_pixels_wraps_ra():
    cells = patch_pixels(8, 350.0, 10.0, -5.0, 5.0)
    pts = radec_to_pix(8, np.array([355.0, 5.0]), np.array([0.0, 0.0]))
    assert np.all(np.isin(pts, cells))
```

- [ ] **Step 2: Run, expect failure**
- [ ] **Step 3: Implement.** `cone_pixels` via `hp.query_disc(nside, hp.ang2vec(ra, dec, lonlat=True), np.radians(r), inclusive=True, nest=nest)`. `patch_pixels`: if the patch wraps, split into `[ra_min, 360)` and `[0, ra_max]` and union; each via `hp.query_polygon` on the four corners (`inclusive=True`), plus `hp.query_strip` fallback when the patch spans more than 180° in RA.
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** — `feat: _core.healpix single HEALPix helper`

---

### Task 5: `_core.geometry`

**Files:**
- Create: `oneuniverse/_core/geometry.py`
- Test: `test/core/test_geometry.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_cone_contains_haversine_small_radius():
    c = Cone(ra=10.0, dec=10.0, radius=1 / 3600)  # 1 arcsec
    assert c.contains(np.array([10.0, 10.0]), np.array([10.0, 10.0 + 0.5 / 3600])).tolist() == [True, True]
    assert not c.contains(np.array([10.0]), np.array([10.0 + 2 / 3600]))[0]

def test_skypatch_wrap():
    p = SkyPatch(350.0, 10.0, -5.0, 5.0)
    assert p.wraps and p.contains(np.array([355.0, 5.0, 180.0]), np.zeros(3)).tolist() == [True, True, False]

def test_zrange_and_cube():
    assert ZRange(0.1, 0.2).contains(np.array([0.15, 0.3])).tolist() == [True, False]
    cu = Cube((0, 0, 0), (10, 10, 10)); assert cu.volume == 1000.0
    assert cu.contains(np.array([5.0, 11.0]), np.zeros(2), np.zeros(2)).tolist() == [True, False]

def test_selections_are_frozen_and_hashable():
    assert hash(Cone(1, 2, 3)) == hash(Cone(1, 2, 3))
```

- [ ] **Step 2: Run, expect failure**
- [ ] **Step 3: Implement** the four frozen slotted dataclasses; `Cone.contains` uses haversine in float64; `__post_init__` validates ranges (`radius > 0`, `-90 <= dec <= 90`, `lo < hi`) raising `ValueError`.
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** — `feat: _core.geometry shared selectors`

---

### Task 6: Boundary and cosmology guards

**Files:**
- Create: `test/test_guards.py`
- Delete: `test/test_sim_no_pillar1_imports.py`, `test/test_measure_import_boundary.py` (superseded)

**Rules (from master plan Global Constraints):**

| Tree | Forbidden imports |
|---|---|
| `oneuniverse/_core` | `oneuniverse.data`, `.combine`, `.measure`, `.simulation`, `.twin`, `.reference` |
| `oneuniverse/simulation` | `oneuniverse.data`, `.combine`, `.measure`, `.twin`, `.reference` |
| `oneuniverse/measure` | `oneuniverse.simulation`, `.reference`, `.twin` |
| `oneuniverse/twin/engine.py` | `oneuniverse.reference` |

Cosmology guard on `oneuniverse/data`, `oneuniverse/combine`, `oneuniverse/measure`: walk the AST; fail on any `Name`/`Attribute`/function-argument identifier in `{"H0", "Om0", "omega_m", "Omega_m", "sigma8", "sigma_8", "w0", "wa"}`, any import of `astropy.cosmology`, `camb`, `classy`, and any string constant (outside docstrings) in `{"comoving_distance", "r_comoving", "dist_mpc"}`. Allow-list: a module-level dict `ALLOW = {"<relative path>": {"<identifier>", ...}}` in the test, empty at R0.

- [ ] **Step 1: Write the test** with a helper `violations(tree_root: Path, forbidden_prefixes: set[str]) -> list[str]` and `cosmology_violations(tree_root) -> list[str]`; one parametrised test per table row and one for the cosmology rule; each asserts an empty list and prints the violations on failure. Trees that do not exist yet (`reference`, `_core` sub-trees) are skipped with `pytest.skip`, not failed.
- [ ] **Step 2: Run on the old tree**

Run: `pytest test/test_guards.py -v`
Expected: boundary rows PASS (old `twin/` imports data and simulation but is not guarded; `simulation` is clean). The cosmology rule FAILS on `oneuniverse/data/surveys/test/dummy/loader.py` (H0) — this is expected and documents the old violation.

- [ ] **Step 3: Add a temporary allow-list entry** `ALLOW = {"data/surveys/test/dummy/loader.py": {"*"}}` with the comment `# removed in R1 Task 1 when the old data package is deleted`.
- [ ] **Step 4: Run, expect PASS; run the whole suite, expect unchanged counts minus the two deleted files**
- [ ] **Step 5: Commit** — `test: boundary and cosmology guards`

---

### Task 7: STATUS file and step close-out

**Files:**
- Create: `plans/refactor-v1/STATUS.md`

- [ ] **Step 1: Create `STATUS.md`** with the table `| step | date | commit | tests (passed/skipped) | default suite time | notes |` and one row for R0.
- [ ] **Step 2: Update the memory file** `project_oneuniverse_refactor_v1.md` in the claude_config memory directory (R0 done, branch name), commit and push claude_config.
- [ ] **Step 3: Commit** — `docs: R0 close-out`

**Exit criterion:** `pytest -n 6` green on `refactor-v1` with `test/test_guards.py` included; `_core` has five modules and 100 % of its functions are exercised by `test/core/`.
