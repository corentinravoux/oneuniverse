# Deferred refactors (#7/S10) + SQL surfacing (#2) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the two roadmap items deferred by the 2026-06-10 structural review — (#7) one shared registry utility + `converter.py` split + entry-point loaders, (S10) twin module consolidation — and (#2) surface the already-built SQL export through the README, a CLI script, and the package docstring surface.

**Architecture:** Two **independent** parts. Part A is internal-structure debt paydown — every change keeps the current public import surface intact via façade re-exports and deprecation shims, so the 817-test suite is the regression oracle. Part B is documentation/UX over code that already exists and is already tested (`data/sql.py`, `oufsim/sql.py`, `MeasurementSet.to_sql`). Parts A and B touch disjoint files and can run in either order.

**Tech Stack:** Python 3.9+, pytest, pyarrow/parquet, sqlite3 (stdlib), `importlib.metadata.entry_points`. No new runtime dependency.

**Scope note (independent subsystems):** Per the writing-plans scope check, Part A (refactor) and Part B (SQL surfacing) are independent and each produces working, testable software alone. They are kept in one document because the user asked for "points 1 and 2" together; execute Part A or Part B first, no ordering constraint.

**Invariants that must not break (both parts):**
- Cosmology-free data layer (`check_invariants` stays green). See [[feedback_no_cosmology_in_pillar1]].
- Rule-1 import guard: `oneuniverse.simulation` ⊥ `oneuniverse.data`/`combine` (`test/test_measure_import_boundary.py` and the simulation guard stay green).
- Every name currently importable from `oneuniverse.data.converter`, `oneuniverse.twin`, `oneuniverse.simulation.converter`, `oneuniverse.twin.engine`, and `oneuniverse.simulation.oufsim.native` remains importable from the same path.

**Baseline before starting:** `cd Packages/oneuniverse && pip install -e ".[dev]" && pytest -q` → expect `817 passed, 3 skipped`. Work on a branch, not `main`.

---

## PART A — Shared registry + converter split + entry-point loaders (#7) + twin consolidation (S10)

### Task A0: Branch + import-surface guard test

**Files:**
- Test: `test/test_import_surface.py` (create)

This test pins the public import surface so every later move is provably non-breaking. It is the safety net for the whole of Part A.

- [ ] **Step 1: Create the branch**

```bash
cd Packages/oneuniverse
git checkout -b refactor-registry-and-sql-surfacing
```

- [ ] **Step 2: Write the import-surface guard test**

Create `test/test_import_surface.py`:

```python
"""Pins the public import surface refactored in Part A. Every name here is
imported somewhere (notebooks, scripts, downstream). Moving its definition is
fine; changing where it can be imported from is a breaking change."""
import importlib
import pytest

SURFACE = {
    "oneuniverse.data.converter": [
        "write_ouf_dataset", "convert_survey", "convert_sightlines",
        "convert_healpix_map", "read_oneuniverse_parquet", "read_objects_table",
        "fetch_original_columns", "get_manifest", "is_converted", "get_geometry",
    ],
    "oneuniverse.data._registry": [
        "register", "get_loader", "list_surveys", "survey_status",
        "list_survey_types", "get_survey_config", "REGISTRY",
    ],
    "oneuniverse.simulation.converter": ["SimConverter", "register", "get_converter"],
    "oneuniverse.twin.engine": [
        "register_engine", "get_engine", "registered_engines",
        "ForwardEngine", "ReconstructionEngine", "Observation", "ProductBundle",
    ],
    "oneuniverse.simulation.oufsim.native": [
        "ADAPTERS", "register_adapter", "get_adapter",
    ],
    "oneuniverse.twin": [
        "cross_correlation", "power_ratio", "recover_metrics", "RecoveryMetrics",
        "wiener_reconstruct", "constrained_realization", "run_mock_challenge",
    ],
}

@pytest.mark.parametrize("module,names", SURFACE.items())
def test_public_names_importable(module, names):
    mod = importlib.import_module(module)
    missing = [n for n in names if not hasattr(mod, n)]
    assert not missing, f"{module} lost public names: {missing}"
```

- [ ] **Step 3: Run it — must pass against current code**

Run: `pytest test/test_import_surface.py -q`
Expected: PASS (all names already exist today).

- [ ] **Step 4: Commit**

```bash
git add test/test_import_surface.py
git commit -m "test: pin public import surface before registry/converter refactor"
```

---

### Task A1: Shared `Registry` utility

**Files:**
- Create: `oneuniverse/_registry.py`
- Test: `test/test_registry_util.py` (create)

One generic registry with uniform register/get/names/`__contains__`/`mapping`/entry-point semantics, replacing four hand-rolled idioms (review §2.1 S2).

- [ ] **Step 1: Write the failing test**

Create `test/test_registry_util.py`:

```python
import pytest
from oneuniverse._registry import Registry


def test_register_by_explicit_name_then_get():
    reg = Registry("widget")
    reg.register("ALPHA", name="a")
    assert reg.get("a") == "ALPHA"
    assert "a" in reg
    assert reg.names() == ["a"]


def test_register_with_key_function():
    reg = Registry("loader", key=lambda cls: cls.__name__.lower())
    class Foo: ...
    reg.register(Foo)
    assert reg.get("foo") is Foo


def test_duplicate_name_raises():
    reg = Registry("widget")
    reg.register(1, name="x")
    with pytest.raises(ValueError, match="already registered"):
        reg.register(2, name="x")


def test_unknown_get_raises_keyerror_with_known_list():
    reg = Registry("widget")
    reg.register(1, name="x")
    with pytest.raises(KeyError, match="known: \\['x'\\]"):
        reg.get("missing")


def test_mapping_is_read_only_view_of_live_dict():
    reg = Registry("widget")
    reg.register(1, name="x")
    m = reg.mapping
    assert dict(m) == {"x": 1}
    with pytest.raises(TypeError):
        m["y"] = 2  # MappingProxyType is read-only
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest test/test_registry_util.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'oneuniverse._registry'`.

- [ ] **Step 3: Write the implementation**

Create `oneuniverse/_registry.py`:

```python
"""oneuniverse._registry — one registry utility for the whole package.

Replaces four hand-rolled registries (data loaders, simulation converters,
twin engines, oufsim native adapters) that had divergent duplicate-handling and
lookup semantics. Each call site wraps an instance of ``Registry`` and keeps its
existing public functions, so this is an internal consolidation only.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Callable, Dict, Generic, List, Mapping, Optional, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Name → item registry with uniform semantics.

    Parameters
    ----------
    label : str
        Used in error messages (e.g. ``"survey loader"``).
    key : callable, optional
        Derives the registration name from the item when ``name=`` is omitted
        (e.g. ``lambda cls: cls.config.name``).
    """

    def __init__(self, label: str, *, key: Optional[Callable[[T], str]] = None):
        self._label = label
        self._key = key
        self._items: Dict[str, T] = {}

    def register(self, item: T, *, name: Optional[str] = None) -> T:
        if name is None:
            if self._key is None:
                raise ValueError(
                    f"{self._label}: cannot derive a name; pass name= or "
                    f"construct Registry with key="
                )
            name = self._key(item)
        if name in self._items:
            raise ValueError(
                f"{self._label}: '{name}' already registered "
                f"(by {self._items[name]!r})"
            )
        self._items[name] = item
        return item

    def get(self, name: str) -> T:
        if name not in self._items:
            raise KeyError(
                f"{self._label}: unknown '{name}'; known: {sorted(self._items)}"
            )
        return self._items[name]

    def names(self) -> List[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items

    @property
    def items_dict(self) -> Dict[str, T]:
        """The live internal dict (for back-compat shims that exposed it)."""
        return self._items

    @property
    def mapping(self) -> Mapping[str, T]:
        """Read-only view of the registry."""
        return MappingProxyType(self._items)

    def load_entry_points(self, group: str) -> List[str]:
        """Register every plugin advertised under *group*. Returns names added."""
        from importlib.metadata import entry_points
        added: List[str] = []
        for ep in entry_points(group=group):
            if ep.name in self._items:
                continue  # built-in of same name wins; never override silently
            self.register(ep.load(), name=ep.name)
            added.append(ep.name)
        return added
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest test/test_registry_util.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/_registry.py test/test_registry_util.py
git commit -m "feat(registry): add shared Registry utility (S2 foundation)"
```

---

### Task A2: Back the data loader registry with `Registry`

**Files:**
- Modify: `oneuniverse/data/_registry.py`

Keep every public function identical; replace the internal `_REGISTRY: Dict` and the bespoke duplicate/lookup logic with a `Registry` instance. `_REGISTRY` stays bound to the **live internal dict** so existing tests that mutate it directly keep working.

- [ ] **Step 1: Add a focused test for the unchanged behaviour**

Append to `test/test_registry_util.py`:

```python
def test_data_registry_still_exposes_live_dict_and_proxy():
    from oneuniverse.data import _registry as r
    # _REGISTRY is the live dict; REGISTRY is the read-only proxy over it.
    assert r.REGISTRY.keys() == r._REGISTRY.keys()
    assert isinstance(r.list_surveys(), dict)
```

- [ ] **Step 2: Run it — passes today, must still pass after the edit**

Run: `pytest test/test_registry_util.py -q`
Expected: PASS.

- [ ] **Step 3: Rewrite `data/_registry.py` internals**

Replace the body of `oneuniverse/data/_registry.py` with this (public names unchanged: `register`, `get_loader`, `list_surveys`, `survey_status`, `list_survey_types`, `get_survey_config`, `REGISTRY`, `_REGISTRY`):

```python
"""oneuniverse.data._registry — survey-name → loader-class registry.

Backed by the shared :class:`oneuniverse._registry.Registry`. Loaders register
at import time via the ``@register`` decorator (keyed on ``cls.config.name``).
``_REGISTRY`` remains the live internal dict for back-compat with tests that
mutate it directly; ``REGISTRY`` is the read-only proxy.
"""
from __future__ import annotations

from typing import Dict, List, Mapping

from oneuniverse._registry import Registry

_REG: "Registry[type]" = Registry("survey loader", key=lambda cls: cls.config.name)

#: Live internal dict (tests mutate this directly; production goes via register()).
_REGISTRY: Dict[str, type] = _REG.items_dict
#: Read-only public view.
REGISTRY: Mapping[str, type] = _REG.mapping


def register(cls):
    """Class decorator: register a BaseSurveyLoader subclass by ``config.name``."""
    return _REG.register(cls)


def get_loader(name: str):
    """Return an *instance* of the loader registered under *name*."""
    return _REG.get(name)()  # registry stores the class; callers want an instance


def list_surveys(survey_type=None, status=None) -> Dict[str, str]:
    out = {}
    for name in _REG.names():
        cfg = _REG.get(name).config
        if survey_type is not None and cfg.survey_type != survey_type:
            continue
        st = getattr(cfg, "status", "ready")
        if status is not None and st != status:
            continue
        out[name] = cfg.description if st == "ready" \
            else f"{cfg.description} [planned — not yet implemented]"
    return out


def survey_status(name: str) -> str:
    return getattr(_REG.get(name).config, "status", "ready")


def list_survey_types() -> List[str]:
    return sorted({_REG.get(n).config.survey_type for n in _REG.names()})


def get_survey_config(name: str):
    return _REG.get(name).config
```

Note: `get_loader`/`survey_status`/`get_survey_config` now raise `KeyError` with the message `survey loader: unknown '<name>'; known: [...]` instead of the old wording. If any test asserts the *old* message text, update that assertion to match (search: `pytest -q test/ -k "unknown_survey or get_loader" `).

- [ ] **Step 4: Run the data + registry suites**

Run: `pytest test/test_registry_util.py test/test_import_surface.py -q && pytest -q -k "loader or survey or registry or database"`
Expected: PASS. If a test asserts the old KeyError text, fix that assertion now and re-run.

- [ ] **Step 5: Full suite (the real oracle)**

Run: `pytest -q`
Expected: `817 passed, 3 skipped` (plus the new tests).

- [ ] **Step 6: Commit**

```bash
git add oneuniverse/data/_registry.py test/test_registry_util.py
git commit -m "refactor(data): back loader registry with shared Registry (S2)"
```

---

### Task A3: Back the other three registries with `Registry`

**Files:**
- Modify: `oneuniverse/simulation/converter.py` (lines ~59–85: `_REGISTRY`, `register`, `get_converter`)
- Modify: `oneuniverse/twin/engine.py` (lines ~67–88: `_ENGINES`, `register_engine`, `get_engine`, `registered_engines`)
- Modify: `oneuniverse/simulation/oufsim/native.py` (lines ~43–60: `ADAPTERS`, `register_adapter`, `get_adapter`)

Same pattern as A2: instance of `Registry`, public functions preserved, live dict preserved where it was a module global.

- [ ] **Step 1: simulation/converter.py — swap internals**

In `oneuniverse/simulation/converter.py`, replace the registry block with:

```python
from oneuniverse._registry import Registry

_REG: "Registry[type]" = Registry("sim converter", key=lambda cls: cls.format_code)
_REGISTRY = _REG.items_dict  # back-compat: live dict


def register(cls):
    """Register a SimConverter subclass by its ``format_code``."""
    return _REG.register(cls)


def get_converter(code: str):
    return _REG.get(code)
```

Verify the key attribute: confirm the duplicate-check at the old `converter.py:70` keyed on `code = cls.format_code` (or equivalent class attribute). If the attribute is named differently, use that exact name in `key=`. Check with: `grep -nE "code =|format_code|\.name" oneuniverse/simulation/converter.py`.

- [ ] **Step 2: twin/engine.py — swap internals**

In `oneuniverse/twin/engine.py`, replace the `_ENGINES` block with:

```python
from oneuniverse._registry import Registry

_REG: "Registry[type]" = Registry("twin engine", key=lambda cls: cls.name)
_ENGINES = _REG.items_dict  # back-compat: live dict


def register_engine(cls):
    return _REG.register(cls)


def get_engine(name: str):
    return _REG.get(name)


def registered_engines():
    return tuple(_REG.names())
```

Confirm engines key on `cls.name` (check `grep -nE "name =|\.name" oneuniverse/twin/engine.py` near the old `register_engine`). Use the exact attribute.

- [ ] **Step 3: oufsim/native.py — swap internals**

In `oneuniverse/simulation/oufsim/native.py`, replace the `ADAPTERS` block with:

```python
from oneuniverse._registry import Registry

_REG: "Registry[object]" = Registry("native adapter")
ADAPTERS = _REG.items_dict  # back-compat: live dict of fmt -> adapter instance


def register_adapter(cls):
    """Class decorator: instantiate and register a native reader adapter."""
    inst = cls()
    _REG.register(inst, name=inst.format_name)
    return cls


def get_adapter(native_format: str):
    return _REG.get(native_format)
```

Confirm the format key: the old code did `ADAPTERS[fmt] = cls()` — find what `fmt` was (likely `cls.format_name` or `cls().format`). Use that exact attribute in place of `inst.format_name`. Check: `grep -nE "fmt =|format|\.name" oneuniverse/simulation/oufsim/native.py`.

- [ ] **Step 4: Run targeted + full suite**

Run: `pytest -q -k "converter or engine or adapter or oufsim or twin" && pytest -q`
Expected: `817 passed, 3 skipped` + new tests. The Rule-1 guard (`test_measure_import_boundary.py`) must stay green — `_registry.py` imports nothing from `data`/`combine`, so `simulation` stays clean.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/simulation/converter.py oneuniverse/twin/engine.py oneuniverse/simulation/oufsim/native.py
git commit -m "refactor: back sim-converter, twin-engine, native-adapter registries with shared Registry (S2)"
```

---

### Task A4: Entry-point survey loaders

**Files:**
- Modify: `oneuniverse/data/__init__.py` (after built-in survey imports)
- Modify: `setup.py` or `pyproject.toml` (declare the entry-point group for the package's own loaders so the mechanism is exercised end-to-end)
- Test: `test/test_entry_point_loaders.py` (create)

Lets community survey loaders ship as separate installable packages (review §2.1 S2: "the right long-term answer to the 7 scaffold loaders").

- [ ] **Step 1: Write the failing test (monkeypatched entry points — no real install needed)**

Create `test/test_entry_point_loaders.py`:

```python
"""Entry-point loader discovery. We monkeypatch importlib.metadata.entry_points
so the test needs no external package installed."""
from oneuniverse._registry import Registry


class _FakeEP:
    def __init__(self, name, obj):
        self.name = name
        self._obj = obj
    def load(self):
        return self._obj


def test_load_entry_points_registers_plugin(monkeypatch):
    import oneuniverse._registry as rmod
    sentinel = object()
    monkeypatch.setattr(
        rmod, "entry_points",
        lambda group=None: [_FakeEP("plugin_survey", sentinel)] if group == "oneuniverse.survey_loaders" else [],
        raising=False,
    )
    reg = Registry("survey loader")
    added = reg.load_entry_points("oneuniverse.survey_loaders")
    assert added == ["plugin_survey"]
    assert reg.get("plugin_survey") is sentinel


def test_builtin_wins_over_entry_point_of_same_name(monkeypatch):
    import oneuniverse._registry as rmod
    monkeypatch.setattr(
        rmod, "entry_points",
        lambda group=None: [_FakeEP("dup", object())],
        raising=False,
    )
    reg = Registry("survey loader")
    builtin = object()
    reg.register(builtin, name="dup")
    added = reg.load_entry_points("oneuniverse.survey_loaders")
    assert added == []                 # plugin skipped
    assert reg.get("dup") is builtin   # built-in retained
```

Note: `Registry.load_entry_points` imports `entry_points` *inside* the method (`from importlib.metadata import entry_points`). For the monkeypatch above to take effect, change A1's `load_entry_points` to use a module-level import instead. Apply this edit to `oneuniverse/_registry.py`: add `from importlib.metadata import entry_points` at the top of the file, and delete the in-method `from importlib.metadata import entry_points` line.

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest test/test_entry_point_loaders.py -q`
Expected: FAIL (entry_points not yet a module-level name to patch) — confirms the patch target.

- [ ] **Step 3: Make the entry_points import module-level**

In `oneuniverse/_registry.py`: add `from importlib.metadata import entry_points` to the top imports; in `load_entry_points`, remove the local import and use the module-level name directly.

- [ ] **Step 4: Run the entry-point test**

Run: `pytest test/test_entry_point_loaders.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Call it during data import**

In `oneuniverse/data/__init__.py`, after the block that imports the built-in survey sub-packages (the `# ── Import all survey sub-packages ──` section near line 27), add:

```python
# Community loaders shipped as separate packages register here. Built-ins of the
# same name always win (see Registry.load_entry_points).
from oneuniverse.data._registry import _REG as _loader_registry
_loader_registry.load_entry_points("oneuniverse.survey_loaders")
```

(`_REG` is the instance created in Task A2. If you prefer not to export `_REG`, add a thin `def load_plugins(): return _REG.load_entry_points("oneuniverse.survey_loaders")` to `data/_registry.py` and call that.)

- [ ] **Step 6: Document the group in packaging metadata**

In `setup.py` (or `pyproject.toml`), document the entry-point group so third parties know the contract. Add a comment + (if the package ships any of its own loaders this way, none today) an empty group declaration. In `setup.py`:

```python
    # Third-party survey loaders register here:
    #   [options.entry_points]
    #   oneuniverse.survey_loaders =
    #       my_survey = my_pkg.loader:MySurveyLoader
    entry_points={
        # group reserved for community survey loaders; built-ins are imported directly
        "oneuniverse.survey_loaders": [],
    },
```

- [ ] **Step 7: Full suite**

Run: `pytest -q`
Expected: `817 passed, 3 skipped` + new tests. Discovery with no plugins installed is a no-op.

- [ ] **Step 8: Commit**

```bash
git add oneuniverse/_registry.py oneuniverse/data/__init__.py test/test_entry_point_loaders.py setup.py
git commit -m "feat(data): discover community survey loaders via entry points (S2)"
```

---

### Task A5: Split `converter.py` (S1)

**Files:**
- Create: `oneuniverse/data/_converter_core.py` (partition engine — shared writers)
- Create: `oneuniverse/data/_converter_point.py` (POINT path)
- Create: `oneuniverse/data/_converter_sightline.py` (SIGHTLINE + HEALPIX map paths)
- Create: `oneuniverse/data/_linkback.py` (original-column fetch)
- Modify: `oneuniverse/data/converter.py` → dispatch façade re-exporting all current public names

Pure mechanical cut-paste-move. `LIGHTCURVE` already lives in `_converter_lightcurve.py` — this follows that precedent (review §2.1 S1). The import-surface test (A0) + full suite are the oracle.

**Move map** (function → new home; names are from `grep -nE "^def " converter.py`):

| Function | New module |
|---|---|
| `_default_stats_builder`, `_prepare_output_dir`, `_write_partitions`, `_auto_partition_nside`, `_write_partitions_by_healpix`, `_chunk_to_table`, `_write_single_parquet`, `_count_rows`, `_log_summary`, `_load_manifest` | `_converter_core.py` |
| `write_ouf_dataset`, `convert_survey` | `_converter_point.py` |
| `convert_sightlines`, `convert_healpix_map` | `_converter_sightline.py` |
| `fetch_original_columns`, `_fetch_from_parquet`, `_fetch_from_fits`, `_fetch_from_csv` | `_linkback.py` |
| `read_oneuniverse_parquet`, `read_objects_table`, `get_manifest`, `is_converted`, `get_geometry` | stay in `converter.py` (small readers) |

- [ ] **Step 1: Create `_converter_core.py`**

Cut the ten core/writer functions listed above out of `converter.py` into a new `oneuniverse/data/_converter_core.py`. Move their imports with them (pyarrow, healpy, manifest types). Add a module docstring: `"""Shared OUF partition-writing engine: stats, partitioning, parquet writers."""`.

- [ ] **Step 2: Create `_converter_point.py`, `_converter_sightline.py`, `_linkback.py`**

Cut `write_ouf_dataset` + `convert_survey` into `_converter_point.py`; `convert_sightlines` + `convert_healpix_map` into `_converter_sightline.py`; the four fetch functions into `_linkback.py`. Each new module imports what it needs from `_converter_core` (e.g. `from oneuniverse.data._converter_core import _write_partitions_by_healpix, _default_stats_builder, ...`).

- [ ] **Step 3: Turn `converter.py` into the façade**

`converter.py` keeps the small readers (`read_oneuniverse_parquet`, `read_objects_table`, `get_manifest`, `is_converted`, `get_geometry`) and the column validator, and re-exports everything moved:

```python
# ── public façade: definitions live in sibling modules (S1 split) ──
from oneuniverse.data._converter_core import (  # noqa: F401
    _default_stats_builder, _prepare_output_dir, _write_partitions,
    _auto_partition_nside, _write_partitions_by_healpix, _chunk_to_table,
    _write_single_parquet, _count_rows, _log_summary, _load_manifest,
)
from oneuniverse.data._converter_point import (  # noqa: F401
    write_ouf_dataset, convert_survey,
)
from oneuniverse.data._converter_sightline import (  # noqa: F401
    convert_sightlines, convert_healpix_map,
)
from oneuniverse.data._linkback import (  # noqa: F401
    fetch_original_columns, _fetch_from_parquet, _fetch_from_fits, _fetch_from_csv,
)
```

Keep the underscore re-exports too: some tests import `_write_partitions_by_healpix`, `_auto_partition_nside`, `_count_rows` from `converter`. Find them first: `grep -rnE "from oneuniverse.data.converter import|converter\._" test/ | grep "_"`.

- [ ] **Step 4: Run the import-surface guard + any test importing converter internals**

Run: `pytest test/test_import_surface.py -q && pytest -q -k "converter or convert or sightline or healpix or linkback or partition"`
Expected: PASS. Fix any private-name import the guard/tests reveal by adding it to the façade re-export list.

- [ ] **Step 5: Full suite**

Run: `pytest -q`
Expected: `817 passed, 3 skipped` + new tests.

- [ ] **Step 6: Commit**

```bash
git add oneuniverse/data/_converter_core.py oneuniverse/data/_converter_point.py oneuniverse/data/_converter_sightline.py oneuniverse/data/_linkback.py oneuniverse/data/converter.py
git commit -m "refactor(data): split converter.py monolith into focused modules behind a façade (S1)"
```

---

### Task A6 (S10): Twin module consolidation — *lowest priority, fully shimmed*

**Files:**
- Create: `oneuniverse/twin/metrics.py` (consolidated field-comparison metrics)
- Modify: `oneuniverse/twin/verify.py` → deprecation shim re-exporting from `metrics.py`
- Modify: `oneuniverse/twin/validation.py` → deprecation shim re-exporting from `metrics.py`
- Modify: `oneuniverse/twin/__init__.py` → import from `metrics` (public names unchanged)
- Test: `test/test_twin_metrics_shim.py` (create)

After S9, `verify.py` (`cross_correlation`, `power_ratio`) and `validation.py` (`recover_metrics`, `RecoveryMetrics`) are both thin wrappers over `simulation.validation.binned_mode_powers`. S10 gathers the twin-side comparison surface into one module while keeping the old import paths alive via shims (review §6: "module merge breaks public imports; defer to a deprecation cycle" — this task *is* that cycle, done safely).

**This task is optional.** It is debt-only with no functional gain. If execution time is constrained, stop after A5 — Part A is already complete and valuable. Do A6 only if a clean twin surface is wanted now.

- [ ] **Step 1: Write the shim test**

Create `test/test_twin_metrics_shim.py`:

```python
"""S10: twin metrics consolidated into twin.metrics; old paths still work."""
import warnings
import numpy as np


def test_new_module_has_the_metrics():
    from oneuniverse.twin import metrics
    assert hasattr(metrics, "cross_correlation")
    assert hasattr(metrics, "power_ratio")
    assert hasattr(metrics, "recover_metrics")
    assert hasattr(metrics, "RecoveryMetrics")


def test_old_paths_still_import_and_warn():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from oneuniverse.twin.verify import cross_correlation  # noqa: F401
        from oneuniverse.twin.validation import recover_metrics  # noqa: F401
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_numerics_unchanged():
    from oneuniverse.twin import metrics
    rng = np.random.default_rng(0)
    a = rng.normal(size=(16, 16, 16))
    k, r = metrics.cross_correlation(a, a, box_size=100.0)
    assert np.allclose(r[np.isfinite(r)], 1.0, atol=1e-6)  # self-correlation = 1
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest test/test_twin_metrics_shim.py -q`
Expected: FAIL (`oneuniverse.twin.metrics` does not exist).

- [ ] **Step 3: Create `metrics.py` by moving the bodies**

Move the *implementations* of `cross_correlation`, `power_ratio` (from `verify.py`) and `recover_metrics`, `RecoveryMetrics` (from `validation.py`) into `oneuniverse/twin/metrics.py`, keeping their imports of `binned_mode_powers` / `_kf_edges` (these came from `simulation.validation` per the S9 work). Module docstring: `"""Twin-side field-comparison metrics (S10 consolidation of verify+validation)."""`.

- [ ] **Step 4: Convert `verify.py` and `validation.py` to shims**

Replace the body of `oneuniverse/twin/verify.py` with:

```python
"""Deprecated: moved to oneuniverse.twin.metrics (S10). Kept for back-compat."""
import warnings
from oneuniverse.twin.metrics import cross_correlation, power_ratio  # noqa: F401

warnings.warn(
    "oneuniverse.twin.verify is deprecated; import from oneuniverse.twin.metrics",
    DeprecationWarning, stacklevel=2,
)
```

Replace the body of `oneuniverse/twin/validation.py` with:

```python
"""Deprecated: moved to oneuniverse.twin.metrics (S10). Kept for back-compat."""
import warnings
from oneuniverse.twin.metrics import recover_metrics, RecoveryMetrics  # noqa: F401

warnings.warn(
    "oneuniverse.twin.validation is deprecated; import from oneuniverse.twin.metrics",
    DeprecationWarning, stacklevel=2,
)
```

- [ ] **Step 5: Point `twin/__init__.py` at `metrics`**

In `oneuniverse/twin/__init__.py`, change the two imports
`from oneuniverse.twin.validation import RecoveryMetrics, recover_metrics` and
`from oneuniverse.twin.verify import cross_correlation, power_ratio`
to a single:

```python
from oneuniverse.twin.metrics import (
    cross_correlation, power_ratio, recover_metrics, RecoveryMetrics,
)
```

`__all__` is unchanged.

- [ ] **Step 6: Run shim test + import-surface + full suite**

Run: `pytest test/test_twin_metrics_shim.py test/test_import_surface.py -q && pytest -q`
Expected: PASS; `817 passed, 3 skipped` + new tests. Note: notebooks import `cross_correlation` from `oneuniverse.twin.verify` and `oneuniverse.twin` — both still resolve (the package re-exports from `metrics`; the `verify` shim still exports the name).

- [ ] **Step 7: Commit**

```bash
git add oneuniverse/twin/metrics.py oneuniverse/twin/verify.py oneuniverse/twin/validation.py oneuniverse/twin/__init__.py test/test_twin_metrics_shim.py
git commit -m "refactor(twin): consolidate verify+validation into twin.metrics with deprecation shims (S10)"
```

---

## PART B — Surface the SQL export (#2)

The SQL code already exists and is tested (`oneuniverse/data/sql.py`, `oneuniverse/simulation/oufsim/sql.py`, `MeasurementSet.to_sql`, `test_data_sql.py`, `test_oufsim_sql.py`, `test_sql_attach_and_measure.py`). This part makes it *discoverable*: a README section, a runnable CLI, and a top-level re-export. Notebook 02 already demonstrates it — no notebook change needed beyond a cross-link.

### Task B1: A runnable `export_to_sql` CLI script

**Files:**
- Create: `scripts/export_to_sql.py`
- Test: `test/test_export_sql_script.py` (create)

- [ ] **Step 1: Write the failing test**

Create `test/test_export_sql_script.py`:

```python
"""The export_to_sql CLI turns an OUF (or OUF-Sim) directory into a .sqlite."""
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _tiny_ouf(tmp_path: Path) -> Path:
    from oneuniverse.data.converter import write_ouf_dataset
    from oneuniverse.data.format_spec import DataGeometry
    from oneuniverse.data.manifest import LoaderSpec
    import healpy as hp
    n = 200
    rng = np.random.default_rng(0)
    ra = rng.uniform(150, 160, n); dec = rng.uniform(0, 10, n)
    df = pd.DataFrame({
        "ra": ra, "dec": dec, "z": rng.uniform(0.5, 2.0, n),
        "z_type": np.full(n, "spec"), "z_err": np.full(n, 1e-4),
        "galaxy_id": np.arange(n), "survey_id": np.zeros(n, "i8"),
        "_original_row_index": np.arange(n, dtype="i8"),
        "_healpix32": hp.ang2pix(32, ra, dec, nest=True, lonlat=True).astype("i4"),
    })
    od = tmp_path / "toy" / "oneuniverse"
    write_ouf_dataset(df=df, out_dir=od, survey_name="toy",
                      survey_type="spectroscopic", geometry=DataGeometry.POINT,
                      loader=LoaderSpec(name="toy", version="0"))
    return od.parent


def test_cli_exports_sqlite(tmp_path):
    src = _tiny_ouf(tmp_path)
    out = tmp_path / "toy.sqlite"
    r = subprocess.run(
        [sys.executable, "scripts/export_to_sql.py", str(src), "-o", str(out)],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1],
    )
    assert r.returncode == 0, r.stderr
    assert out.exists()
    con = sqlite3.connect(out)
    n = con.execute("SELECT COUNT(*) FROM objects").fetchone()[0]
    assert n == 200
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest test/test_export_sql_script.py -q`
Expected: FAIL (`scripts/export_to_sql.py` does not exist → non-zero return code).

- [ ] **Step 3: Write the CLI**

Create `scripts/export_to_sql.py`:

```python
#!/usr/bin/env python3
"""Export an OUF or OUF-Sim directory to a SQLite database.

Examples
--------
    python scripts/export_to_sql.py /path/to/survey/oneuniverse -o survey.sqlite
    python scripts/export_to_sql.py /path/to/simstore -o sim.sqlite --sim
    python scripts/export_to_sql.py /path/to/survey/oneuniverse --attach  # DuckDB DDL only
"""
import argparse
import sys
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", type=Path, help="OUF dataset dir, or OUF-Sim store dir with --sim")
    p.add_argument("-o", "--out", type=Path, help="output .sqlite path (default: <source>.sqlite)")
    p.add_argument("--sim", action="store_true", help="treat source as an OUF-Sim store")
    p.add_argument("--attach", action="store_true",
                   help="print zero-copy DuckDB attach DDL instead of materialising")
    args = p.parse_args(argv)

    if args.sim:
        from oneuniverse.simulation.oufsim import SimStore
        from oneuniverse.simulation.oufsim.sql import export_sim_sql
        store = SimStore(args.source)
        out = args.out or args.source.with_suffix(".sqlite")
        export_sim_sql(store, out)
        print(f"wrote {out}")
        return 0

    from oneuniverse.data.dataset_view import DatasetView
    view = DatasetView.from_path(args.source.parent if args.source.name == "oneuniverse" else args.source)
    if args.attach:
        from oneuniverse.data.sql import attach_sql_ddl
        print(attach_sql_ddl([view]))
        return 0
    from oneuniverse.data.sql import export_sql
    out = args.out or Path(str(args.source).rstrip("/") + ".sqlite")
    export_sql([view], out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Verify the `DatasetView.from_path` argument convention against the notebook helper (`to_ouf_view` calls `DatasetView.from_path(od.parent)` where `od` ends in `/oneuniverse`). Adjust the `.parent if ... == "oneuniverse"` line if `from_path` expects the `oneuniverse/` dir itself.

- [ ] **Step 4: Run the CLI test**

Run: `pytest test/test_export_sql_script.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/export_to_sql.py test/test_export_sql_script.py
git commit -m "feat(scripts): export_to_sql CLI for OUF and OUF-Sim -> SQLite"
```

---

### Task B2: README SQL section + top-level re-export

**Files:**
- Modify: `README.md` (or `Packages/oneuniverse/README.md`) — add a "SQL export" section
- Modify: `oneuniverse/__init__.py` — re-export `export_sql` for discoverability
- Modify: `notebooks/README.md` — one cross-link line (notebook 02 already shows SQL)
- Test: `test/test_sql_toplevel_export.py` (create)

- [ ] **Step 1: Write the failing re-export test**

Create `test/test_sql_toplevel_export.py`:

```python
def test_export_sql_is_discoverable_from_top_level():
    import oneuniverse
    assert hasattr(oneuniverse, "export_sql")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest test/test_sql_toplevel_export.py -q`
Expected: FAIL (`oneuniverse` has no `export_sql`).

- [ ] **Step 3: Add the re-export**

In `oneuniverse/__init__.py`, add (near other top-level convenience imports; keep it lazy-safe — if `__init__` avoids heavy imports, place it behind the existing public-API block):

```python
from oneuniverse.data.sql import export_sql  # noqa: F401
```

If `oneuniverse/__init__.py` deliberately stays import-light, instead add `export_sql` to `oneuniverse.data`'s `__init__` and change the test to `import oneuniverse.data as d; assert hasattr(d, "export_sql")`. Check current style first: `sed -n '1,40p' oneuniverse/__init__.py`.

- [ ] **Step 4: Run the re-export test**

Run: `pytest test/test_sql_toplevel_export.py -q`
Expected: PASS.

- [ ] **Step 5: Add the README section**

Append to the package `README.md` a section:

````markdown
## SQL export

Both on-disk formats export to standard SQL — SQLite (a single portable file,
stdlib only) or zero-copy DuckDB views over the existing parquet.

```python
from oneuniverse import export_sql
from oneuniverse.data.dataset_view import DatasetView

view = DatasetView.from_path("/path/to/survey")
export_sql([view], "catalog.sqlite")          # materialise (≡ re-encode)
```

```bash
# or from the command line:
python scripts/export_to_sql.py /path/to/survey/oneuniverse -o catalog.sqlite
python scripts/export_to_sql.py /path/to/simstore -o sim.sqlite --sim
python scripts/export_to_sql.py /path/to/survey/oneuniverse --attach   # DuckDB DDL
```

A `MeasurementSet` also exports directly: `ms.to_sql("ms.sqlite")`.
See `notebooks/02_sql_database.ipynb` for the full tour (real eBOSS in pure SQL,
ONEUID as a JOIN, the simulation chunk index as a relational bbox query).
````

- [ ] **Step 6: Cross-link from notebooks/README**

In `notebooks/README.md`, the notebook-02 row already exists; add the script pointer at the end of the table caption or scope note: `The same export is available headless via scripts/export_to_sql.py.`

- [ ] **Step 7: Full suite + commit**

Run: `pytest -q`
Expected: `817 passed, 3 skipped` + all new Part-B tests.

```bash
git add oneuniverse/__init__.py README.md notebooks/README.md test/test_sql_toplevel_export.py
git commit -m "docs(sql): surface SQL export via README, top-level re-export, notebook cross-link"
```

---

## Final verification (both parts)

- [ ] **Step 1: Whole suite from the package dir**

Run: `cd Packages/oneuniverse && pytest -q`
Expected: `817 passed` (originals) `+ ~14` new tests, `3 skipped`, `0 failed`.

- [ ] **Step 2: Rule-1 + cosmology guards explicitly**

Run: `pytest -q -k "import_boundary or invariant or cosmolog"`
Expected: PASS — `simulation` still imports nothing from `data`/`combine`; data layer still cosmology-free.

- [ ] **Step 3: Notebooks still execute (smoke — 01 and 02 only, fast)**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/01_one_universe_of_data.ipynb notebooks/02_sql_database.ipynb`
Expected: exit 0 (proves façade re-exports `write_ouf_dataset` and the SQL surface unchanged).

- [ ] **Step 4: Update the review roadmap**

In `research/2026-06-10-structural-review-and-sql-design.md` §6, flip rows #7 and (if A6 done) S10 from `⏳ open` to `✅ done`, with a one-line pointer to this plan. Commit:

```bash
git add research/2026-06-10-structural-review-and-sql-design.md
git commit -m "docs(review): mark roadmap #7 (+S10) done — see 2026-06-15 plan"
```

---

## Self-review (done while writing — per writing-plans skill)

**Spec coverage.** Point 1 (#7) → A1 (Registry), A2/A3 (four registries migrated), A4 (entry points), A5 (converter split); S10 → A6. Point 2 (#2) → B1 (CLI), B2 (README + re-export + notebook link). All covered.

**Placeholder scan.** No TBD/"handle edge cases"/"similar to Task N"; every code step carries real code; the one genuinely mechanical step (A5 cut-paste) ships a concrete move-map table + the full façade re-export block + the import-surface test as its oracle.

**Type/name consistency.** `Registry` API (`register/get/names/__contains__/items_dict/mapping/load_entry_points`) is used identically in A2/A3/A4. `_REG` instance name consistent across migrated modules. `export_sql([view], out)` signature matches `data/sql.py` and is used identically in B1 and B2. Public names in A0's `SURFACE` map are the exact names re-exported by A5's façade and A6's `__init__` edit.

**Known verification points flagged inline** (attribute names to confirm before editing): sim-converter key attr (A3 S1), twin-engine key attr (A3 S2), native-adapter format key (A3 S3), `DatasetView.from_path` dir convention (B1 S3), `__init__` import-weight style (B2 S3). Each step says how to check.
