# R8 — Release: tests layout, examples, docs, CLAUDE.md, README, 1.0.0

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish 1.0: a test tree that mirrors the package, diagnostic examples that write figures to an ignored directory, two scripts, a 100-line `CLAUDE.md`, a rewritten README and MkDocs site that build strictly, the old plans archived, version `1.0.0` and the merge to `main`.

**Architecture:** Nothing new in the library except removal of the R0 registry shim. Documentation is generated from docstrings by mkdocstrings; guide pages are hand-written against the §7 API of the design.

**Tech Stack:** pytest, matplotlib (examples only), MkDocs Material + mkdocstrings, GitHub Actions.

**Spec:** design §6.10, §7, §10. Master plan Global Constraints (no binary files committed; default suite < 60 s).

---

### Task 1: Test tree and markers

**Files:**
- Modify: `test/` layout; `pyproject.toml` (`[tool.pytest.ini_options]`)

- [ ] **Step 1: Ensure the layout** is exactly `test/{core,data,combine,measure,simulation,reference,twin}/test_*.py`, `test/fixtures/*.py`, `test/test_guards.py`, `test/conftest.py` (shared fixtures: `COSMO`, `two_survey_db`, `synthetic_view`). Move any stray file; no file name contains a phase or review tag (`grep -l "phase\|review\|cosmetic" test -r` returns nothing).
- [ ] **Step 2: Real-data tests.** Create `test/data/test_real_surveys.py` with two tests gated on `ONEUNIVERSE_DATA_ROOT` (`pytest.skip` when unset): convert real eBOSS DR16Q (`spectroscopic/eboss/qso/DR16Q_Superset_v3.fits`) and real DESI DR1 QSO (`spectroscopic/desi/dr1/qso/QSO_full.dat.fits`) into a temp root, assert `n_rows > 900_000` / `> 1_200_000`, `z_type == "spec"`, a cone query returns rows, and a cross-match of the two finds `> 200_000` multi-survey objects with `CrossMatchRules(sky_tol_arcsec=2.0, dz_tol=5e-3)`. Mark `slow`.
- [ ] **Step 3: Verify** `pytest` (single process) prints `< 60 s` total and `pytest -m slow -n 6` passes (skips the real-data tests on this machine). Record both times in STATUS.md.
- [ ] **Step 4: Commit** — `test: final tree layout, real-survey gated tests`

---

### Task 2: Examples (diagnostic figures) and scripts

**Files:**
- Create: `examples/README.md`, `examples/diagnostics/{sky_coverage,partition_pruning,crossmatch_separations,measurement_set_overview,resim_buffer_convergence,mock_challenge}.py`, `examples/output/.gitignore` (`*` + `!.gitignore`)
- Create: `scripts/convert_surveys.py`, `scripts/build_reference_store.py`
- Delete: every other file under `scripts/`
- Test: `test/test_examples.py`

Each diagnostics script: `python examples/diagnostics/<name>.py [--out examples/output] [--data-root ...]`, runs on the synthetic fixtures when no data root is given, writes one PNG, prints its path, uses `plt.style.use(os.environ["PLT_STYLE"])` only when the variable is set.

- [ ] **Step 1: Write `test_examples.py`**: parametrised over the six scripts, runs each with `subprocess.run([sys.executable, script, "--out", tmp_path], check=True, timeout=120)` and asserts one PNG > 30 kB was written. Mark `slow`.
- [ ] **Step 2: Write the scripts.** `convert_surveys.py --data-root --db-root [--surveys eboss_qso desi_qso] [--overwrite]` → calls `data.convert` per survey and prints `Database(db_root).summary()`. `build_reference_store.py --out [--box 512 --n-grid 128 --seed 0]` → `generate_linear_sim` + `LinearSimConverter.convert`.
- [ ] **Step 3: Run** `pytest -m slow test/test_examples.py`. Expected: PASS.
- [ ] **Step 4: Commit** — `feat(examples): diagnostic figure scripts and the two CLI scripts`

---

### Task 3: Archive history, rewrite CLAUDE.md

**Files:**
- Move: `plans/*.md` (except `2026-11-12-refactor-v1-design.md`, `2026-11-12-refactor-v1-plan.md`, `refactor-v1/`) → `docs/history/plans/`; `REVIEW.md` → `docs/history/REVIEW-2026-06.md`; `research/` stays.
- Rewrite: `CLAUDE.md`

- [ ] **Step 1: Move with `git mv`**; add `docs/history/README.md` (three lines: what this is, that it is not maintained, link to the design).
- [ ] **Step 2: Rewrite `CLAUDE.md`** (≤ 100 lines) with exactly these sections: *Mission* (3 pillars, 4 sentences), *Hard rules* (the table from design §4), *Layout* (one line per subpackage), *Formats* (OUF 3.0 and OUF-Sim 1.0 layouts, 10 lines), *Identity* (`source_row`, `object_id`, sidecar versions, 5 lines), *Test conventions* (tree mirrors package; `slow` marker; fixtures; no binaries), *Things that bite* (≤ 6 bullets: partition `cell` numbering at the manifest nside; `z_type` registry; `write_dataset` needs `ra/dec` or `hpix32`; `read_box` is periodic; `constrained_realization` needs a seed; duckdb optional), *Where the history is* (`docs/history/`, `research/`). No phase numbers, no changelog.
- [ ] **Step 3: Commit** — `docs: archive pre-1.0 plans, rewrite CLAUDE.md`

---

### Task 4: README and MkDocs site

**Files:**
- Rewrite: `README.md`, `docs_site/index.md`, `docs_site/getting-started.md`, `docs_site/architecture.md`, `docs_site/formats.md` (new), `docs_site/sql.md` (new), `docs_site/api/*.md` (one per subpackage: `core`, `data`, `combine`, `measure`, `simulation`, `reference`, `twin`), `mkdocs.yml` (nav)
- Modify: `.github/workflows/docs.yml` (unchanged command `mkdocs build --strict`; ensure `pip install -e ".[docs,sql]"`)

- [ ] **Step 1: README** (≤ 150 lines): one-paragraph pitch, the design §7 code block verbatim (it must run against the shipped API: paste it into `test/test_readme_example.py` and run it on the synthetic fixtures), "what is real vs stand-in" paragraph (two loaders validated on real data; reference engine is a toy), install (`pip install -e ".[dev,sql]"`), links to docs, citation.
- [ ] **Step 2: Site pages** against the new API; `api/*.md` use `::: oneuniverse.<pkg>` directives with `members` listed explicitly from each `__all__`.
- [ ] **Step 3: Verify** `mkdocs build --strict` passes locally and `pytest test/test_readme_example.py` passes.
- [ ] **Step 4: Commit** — `docs: README and site for 1.0`

---

### Task 5: Version, shim removal, CHANGELOG, merge

**Files:**
- Modify: `oneuniverse/__init__.py` (`__version__ = "1.0.0"`), `CITATION.cff` (version + date), `pyproject.toml` (nothing else), `CHANGELOG.md` (create)
- Delete: `oneuniverse/_registry.py` (R0 shim)

- [ ] **Step 1: CHANGELOG.md** — one `## 1.0.0 — 2026-xx-xx` section: breaking (new API, OUF 3.0, no 2.x reader), fixed (the four verified bugs, by name), removed (stub loaders, SQLite export, dead-end physics), added (SQL face, reference engine, streaming conversion).
- [ ] **Step 2: Final gates**: `pytest` (single process < 60 s), `pytest -m slow -n 6`, `ruff check oneuniverse test --select E9,F63,F7,F82`, `mkdocs build --strict`, `test/test_guards.py`, `git status` clean, no file under `test/` or `examples/` larger than 200 kB.
- [ ] **Step 3: Commit** — `chore: release 1.0.0`
- [ ] **Step 4: Merge** (owner performs or approves): `git checkout main && git merge --no-ff refactor-v1 && git tag v1.0.0 && git push origin main --tags`. Do not push without the owner's go-ahead if this is executed by an agent.
- [ ] **Step 5: STATUS.md final row; memory update** (claude_config: `project_oneuniverse_refactor_v1.md` → done, and update the laptop-era `project_oneuniverse*.md` memories under `memory/-home-ravoux-Documents-Python/` with a one-line banner "superseded by 1.0, see project_oneuniverse_refactor_v1.md"); commit + push claude_config.

**Exit criterion:** `v1.0.0` tagged on `main`; all gates in Step 2 green; the README example runs as a test.

---

## After R8 (separate plans, not part of this refactor)

1. Real data on the laptop: `scripts/convert_surveys.py`, run the gated real-survey tests, rebuild the ONEUID index.
2. Notebooks against 1.0 (data tour, cross-match, measurement set, SQL face, simulation store, resimulation), executed with real eBOSS/DESI where available.
3. Next loaders, each with real data + a research note (`feedback_save_research`).
