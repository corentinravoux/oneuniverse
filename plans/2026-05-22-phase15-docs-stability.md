# Phase 15 — Docs + Stability Hardening

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three thin slabs of polish: a Sphinx scaffold so the API is browsable, a warning-clean suite, and a oneuniverse-scoped `CLAUDE.md` so future-Claude sessions don't re-derive the package layout. No core-code refactors — this is the post-stabilisation tidy-up.

**Architecture:** Independent tasks; each can ship on its own.
- **T1** — Sphinx scaffold under `docs/`: `conf.py`, `index.rst`, autodoc of the public API surface (`oneuniverse.data`, `oneuniverse.combine`).
- **T2** — Warning audit: run `pytest -W error::FutureWarning -W error::DeprecationWarning` over the full suite, fix each one (or whitelist with reason).
- **T3** — Visual-test golden-image existence check: lightweight assertion that the PNG was produced and has a sensible size. Perceptual-hash regression is overkill at this scale.
- **T4** — `CLAUDE.md` at the `Packages/oneuniverse/` root describing the package boundaries, the OUF format, the combine ABC, the test layout. Replaces the implicit Pillar-1 chunk in the project-wide `CLAUDE.md`.

**Tech Stack:** `sphinx`, `sphinx-rtd-theme` (or furo) as optional `docs` extras. No production-code dependencies added.

**Out of scope:**
- Per-method docstring rewrite — Sphinx autodoc surfaces what is already there; doc strings are added opportunistically, not as a mass rewrite.
- Tutorial notebooks — Phase 16+.
- ReadTheDocs hosting — local build only; user-side decision whether to publish.

---

## File Structure

- Create: `docs/conf.py`, `docs/index.rst`, `docs/api.rst`, `docs/_static/.gitkeep`, `docs/Makefile`.
- Modify: `pyproject.toml` — add `[project.optional-dependencies] docs = [sphinx, sphinx-rtd-theme]`.
- Modify: `test/test_visual_selection_weights.py` — fix the one `tight_layout` UserWarning.
- Create: `CLAUDE.md` (package-scoped) at `Packages/oneuniverse/CLAUDE.md`.
- Create: `test/test_no_warnings.py` — runs a smoke test under `-W error` for each warning category we want to keep clean.

---

### Task 1: Sphinx scaffold + autodoc API page

**Files:**
- Create: `docs/conf.py`, `docs/index.rst`, `docs/api.rst`, `docs/Makefile`, `docs/_static/.gitkeep`.
- Modify: `pyproject.toml`.
- Test: a one-shot `sphinx-build -b html docs docs/_build` smoke step in CI (or just a documented manual step).

- [ ] **Step 1:** Add the `docs` extra to `pyproject.toml`:

```toml
[project.optional-dependencies]
docs = [
    "sphinx>=7.0",
    "sphinx-rtd-theme>=2.0",
]
```

- [ ] **Step 2: Write `docs/conf.py`**

```python
# docs/conf.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

project = "oneuniverse"
author = "Corentin Ravoux"
release = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # numpy / google docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]
autosummary_generate = True
autodoc_default_options = {"members": True, "undoc-members": True}
napoleon_numpy_docstring = True
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "pyarrow": ("https://arrow.apache.org/docs/", None),
}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
exclude_patterns = ["_build"]
```

- [ ] **Step 3: `docs/index.rst`**

```rst
oneuniverse
===========

A unified observational foundation for the digital twin of our Universe.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
```

- [ ] **Step 4: `docs/api.rst`** — autodoc tree for every public sub-module:

```rst
API Reference
=============

.. autosummary::
   :toctree: _autosummary
   :recursive:

   oneuniverse.data
   oneuniverse.data.dataset_view
   oneuniverse.data.database
   oneuniverse.data.oneuid
   oneuniverse.data.oneuid_rules
   oneuniverse.data.subobject
   oneuniverse.data.subobject_rules
   oneuniverse.data.temporal
   oneuniverse.data.validity
   oneuniverse.data.pdf
   oneuniverse.data.manifest
   oneuniverse.data.converter
   oneuniverse.data.schema
   oneuniverse.data.selection
   oneuniverse.combine
   oneuniverse.combine.catalog
   oneuniverse.combine.strategies
   oneuniverse.combine.measurements
   oneuniverse.combine.weights
```

- [ ] **Step 5: `docs/Makefile`** (single `html` target):

```makefile
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

clean:
	rm -rf $(BUILDDIR) _autosummary
```

- [ ] **Step 6: Smoke-build**

```bash
pip install -e '.[docs]'
cd docs && make html
```

Inspect `docs/_build/html/index.html`. Expect: API pages render, no `ImportError`/`AttributeError` from autosummary.

- [ ] **Step 7: Commit**

```bash
git add docs/ pyproject.toml
git commit -m "phase15/T1: Sphinx scaffold + autodoc API page (docs extra)"
```

---

### Task 2: Warning audit — clean suite under `-W error::FutureWarning -W error::DeprecationWarning`

**Files:**
- Modify: `test/test_visual_selection_weights.py` — fix the one `tight_layout` UserWarning observed at Phase 12 close (mollview + `add_subplot` are not `tight_layout`-compatible).
- Create: `test/test_no_warnings.py` — runs the equivalent of `pytest -W error::FutureWarning` against a representative subset (so a CI invocation can catch regressions).

- [ ] **Step 1:** `pytest -W error::FutureWarning -W error::DeprecationWarning -q` — capture the failures. Expected at Phase 14 close: one matplotlib `UserWarning` (not the targeted categories). If any FutureWarning/DeprecationWarning leaks in, fix at source. Otherwise tighten the test.

- [ ] **Step 2:** Fix the mollview `tight_layout` warning by switching the layout call:

```python
# test/test_visual_selection_weights.py
# was: fig.tight_layout()
fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.08, wspace=0.25)
```

- [ ] **Step 3:** Write `test/test_no_warnings.py`:

```python
"""Suite-wide warning-cleanliness pin.

Runs a representative slice of the suite with `-W error::FutureWarning`
to catch any new deprecation creep. NOT a full re-run — just enough
coverage to fail fast on the common offenders (pandas groupby,
healpy verbose, numpy float aliases).
"""
import warnings


def test_no_futurewarning_on_typical_workflow(tmp_path):
    import numpy as np
    import pandas as pd
    from oneuniverse.data.converter import write_ouf_dataset
    from oneuniverse.data.dataset_view import DatasetView
    from oneuniverse.data.format_spec import DataGeometry
    from oneuniverse.data.manifest import LoaderSpec

    # Tiny synthetic POINT dataset → write → re-read.
    import healpy as hp
    n = 100
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "ra": rng.uniform(0, 360, n),
        "dec": np.degrees(np.arcsin(rng.uniform(-1, 1, n))),
        "z": np.full(n, 0.5, dtype=np.float32),
        "z_type": np.array(["spec"] * n, dtype="<U4"),
        "z_err": np.full(n, 1e-3, dtype=np.float32),
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": np.array(["fake"] * n, dtype="<U16"),
        "_original_row_index": np.arange(n, dtype=np.int64),
    })
    theta = np.radians(90.0 - df["dec"].to_numpy(dtype=np.float64))
    phi = np.radians(df["ra"].to_numpy(dtype=np.float64))
    df["_healpix32"] = hp.ang2pix(32, theta, phi, nest=True).astype(np.int32)

    out_dir = tmp_path / "x" / "oneuniverse"
    out_dir.mkdir(parents=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write_ouf_dataset(
            df=df, out_dir=out_dir,
            survey_name="x", survey_type="spectroscopic",
            geometry=DataGeometry.POINT,
            loader=LoaderSpec(name="x", version="0"),
        )
        view = DatasetView.from_path(out_dir.parent)
        _ = view.read()

    bad = [
        w for w in caught
        if issubclass(w.category, (FutureWarning, DeprecationWarning))
        and "oneuniverse" in str(w.filename)
    ]
    assert bad == [], [
        f"{w.category.__name__} at {w.filename}:{w.lineno}: {w.message}"
        for w in bad
    ]
```

- [ ] **Step 4:** Run `pytest test/test_no_warnings.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add test/test_visual_selection_weights.py test/test_no_warnings.py
git commit -m "phase15/T2: warning audit — clean FutureWarning/DeprecationWarning"
```

---

### Task 3: Golden-image existence check

**Files:**
- Modify: each `test/test_visual_*.py` to assert the produced PNG is large enough to be a real plot, not an empty placeholder.

**Why:** Today the visual tests save a PNG and assert it exists; a regression that silently produces a 2KB blank canvas slips through. Bump the size floor to ~30KB (a 12×5" plot with axes and a non-trivial dataset is comfortably above that).

This change is tiny — bump the existing `stat().st_size > 10_000` assertions to `> 30_000` and add a `from PIL import Image` shape sanity check where Pillow is available.

- [ ] **Step 1:** Search:

```bash
grep -rn 'st_size > 10_000\|stat\.st_size' test/test_visual_*.py
```

For each match, change to `> 30_000` and add (optionally):

```python
try:
    from PIL import Image
    with Image.open(out_png) as im:
        assert im.width >= 800 and im.height >= 200
except ImportError:
    pass  # Pillow not installed — size check stands alone
```

- [ ] **Step 2:** Run all three visual tests → still PASS.
- [ ] **Step 3: Commit**

```bash
git add test/test_visual_*.py
git commit -m "phase15/T3: visual tests assert PNG > 30KB + image dimensions"
```

---

### Task 4: Package-scoped `CLAUDE.md`

**Files:**
- Create: `Packages/oneuniverse/CLAUDE.md`.

**Why:** The project-level `CLAUDE.md` documents `flip` in depth (200+ lines on Layer 1/2/3 + analytical models). Pillar-1 oneuniverse is referenced obliquely. Future-Claude opens a oneuniverse session and re-derives the layout from `plans/README.md`. Move the canonical map next to the code.

- [ ] **Step 1: Write `Packages/oneuniverse/CLAUDE.md`** (~150 lines, structured as):

```markdown
# CLAUDE.md — oneuniverse (Pillar 1)

This file is the package-scoped guide. The repo-level `Python/CLAUDE.md`
covers other packages (mainly `flip`).

## Mission

oneuniverse is the **data + orchestration** layer of the three-pillar
cosmology stack:

- Pillar 1 (here): standardise, cross-match, weight survey catalogs.
- Pillar 2: estimators (P(k), ξ(r), 1D Lyα power) — `flip` and friends.
- Pillar 3: forward models / mini-simulations.

No estimators ship from this package. No forward models. Just data.

## Package layout

- `oneuniverse/data/` — schema, manifest, converter, DatasetView,
  ONEUID, sub-object links, temporal validity.
- `oneuniverse/combine/` — `WeightedCatalog` + Weight ABC +
  primitives (FKP, IVar, HealpixMap, PDF, BOSS combiner).
- `oneuniverse/data/surveys/` — registered loaders. Add new ones with
  `@register class FooLoader(BaseSurveyLoader)`.

## OUF 2.1 (format on disk)

Each converted dataset lives at::

    {survey_path}/oneuniverse/
    ├── manifest.json
    ├── data/healpix32=00042/part_0000.parquet
    ├── data/healpix32=00043/part_0000.parquet
    └── ...

`manifest.json` is the typed `Manifest` dataclass — see
[`oneuniverse/data/manifest.py`](oneuniverse/data/manifest.py). Required
sub-specs: `PartitioningSpec` (NSIDE may be coarsened by Phase 12 F3),
optional `TemporalSpec`, `DatasetValidity`, `PdfSpec`.

CORE columns (every POINT dataset): `ra, dec, z, z_type, z_err,
galaxy_id, survey_id, _original_row_index, _healpix32`.

`Z_TYPE_VALUES = {"spec", "phot", "phot_pdf", "pv", "none"}`.

## Bitemporal ONEUID / sub-object

- `database.build_oneuid(datasets, rules, name)` writes
  `{root}/_oneuid/<name>.parquet`.
- `database.build_subobject_links(rules, parent_datasets, child_datasets, name)`
  writes `{root}/_subobject/<name>.parquet`.
- Both auto-archive prior versions on rebuild as `<name>__{ISO8601Z}`.
- `database.as_of(when)` returns a filtered clone; `load_oneuid(name,
  as_of=...)` resolves the right archived version.

## Weights

`WeightedCatalog.from_oneuid(index, database).fill_defaults(db,
z_type="spec")` is the canonical entry point. Per-survey custom
weights via `wc.add_weight(survey, Weight(...))`. Compose with `*`
(`ProductWeight`). Public registration:
`oneuniverse.combine.weights.register_default(survey_type, z_type,
factory)`.

## Phase status

See [`plans/README.md`](plans/README.md). Phases 1-15 complete by
2026-05-22 (stabilisation done). Real-survey loader writes (Phase 16+)
are the natural next step.

## Test conventions

- `test/fixtures/` holds factory functions for synthetic DR1 QSO,
  photo-z PDFs, HEALPix maps. Use them in new tests — do **not** ship
  binary test fixtures.
- `test/test_output/*.png` are diagnostic figures, committed for
  inspection. Phase 15 added size + dimension checks.
- `eboss_default_df` session fixture shares one ~31s eBOSS load
  across tests on machines with the DR16Q data.

## Things that bite

- `convert_survey` resolves `data_root` from kwarg → env →
  None; **no module-level state** (Phase 12 D1). Old `set_data_root`
  / `get_data_root` were removed.
- `DatasetView` reads partition NSIDE from
  `manifest.partitioning.extra["nside"]` (Phase 12 D5). Do not
  hardcode `HEALPIX_PARTITION_NSIDE` in new code.
- `_chunk_to_table(chunk, pdf_spec)` is the single path for
  DataFrame → pa.Table in the converter; route any new
  list-column work through it.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "phase15/T4: package-scoped CLAUDE.md (oneuniverse pillar-1 guide)"
```

---

### Task 5: Close Phase 15

- [ ] **Step 1:** Full suite + recorded count.
- [ ] **Step 2:** `pytest -W error::FutureWarning -W error::DeprecationWarning -q` — must be green (the test_no_warnings.py from T2 keeps this property under regression).
- [ ] **Step 3:** Update `plans/README.md` + memory file.
- [ ] **Step 4:** Final commit.

---

## Self-review checklist

**Spec coverage:** T1 docs, T2 warnings, T3 visual goldens, T4 CLAUDE.md. The four items from C+D bucket in the forward dev plan.

**Placeholder scan:** All `<sphinx-build>` and `<grep>` commands give real targets. T4 CLAUDE.md is shown in full. No "TBD" markers.

**Type consistency:** No new types introduced.

**Risk:** Sphinx config + autodoc trees sometimes silently swallow `ImportError`. Mitigation: Task 1 Step 6 inspects the rendered HTML and confirms specific pages render.
