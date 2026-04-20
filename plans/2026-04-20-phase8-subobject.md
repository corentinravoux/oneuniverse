# Phase 8 — Sub-object Hierarchy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bitemporal **sub-object link layer** to the oneuniverse
database: a persisted sidecar that records relationships between one
astrophysical object and another (a supernova *in* its host galaxy; a
galaxy *in* a cluster), keyed on ONEUID, versioned with the same
`DatasetValidity` pattern introduced in Phase 7.

**Architecture:** Sub-object links are *named sidecars*, not per-survey
manifest entries. Links live at `{root}/_subobject/<name>.parquet` with
a companion `<name>.manifest.json`, mirroring the ONEUID-index layout.
A frozen `SubobjectRules` dataclass (symmetric in spirit to
`CrossMatchRules`, different physics) declares the parent/child survey
types, sky/redshift tolerances, and whether ambiguous many-to-one
matches are accepted. Building links requires a covering ONEUID index:
the parent and child identities *are* ONEUIDs, which makes links stable
under re-partitioning. Each link row carries a `confidence` score
derived from the matching algorithm.

**Tech Stack:** `pandas` / `pyarrow` for the sidecar parquet table,
`scipy.spatial.cKDTree` for sky pair search, `astropy.coordinates` for
on-sphere angular distance, the `DatasetValidity` frozen dataclass from
Phase 7.

---

## Dependencies

- **Phase 7 must be complete.** `DatasetValidity` lives at
  `oneuniverse/data/validity.py`; ONEUID manifests already carry a
  validity block; `OneuniverseDatabase.as_of(ts)` and
  `load_oneuid(name, as_of=T)` exist.
- **Covering ONEUID index required at build time.** The user passes
  `oneuid_name="default"` (or any named index) plus the list of parent
  and child datasets. Rows from datasets missing from the index raise
  a build-time error.

## Scope boundary (reminder)

This phase stores, indexes, and queries pairwise parent/child
relationships. It does **not**:

- Fit SN light-curves or infer peak brightness.
- Estimate host-galaxy properties (stellar mass, SFR, morphology).
- Run any probabilistic host-identification model beyond deterministic
  sky + redshift tolerance matching.
- Build forward models (this is a `flip`-side concern).

If the rules give an ambiguous match (one SN, two host candidates within
tolerance), the builder either rejects both (default) or records both
with a `confidence` < 1 (if `accept_ambiguous=True`) — it does not try
to guess which is correct.

## File structure

Files to create:

- `oneuniverse/data/subobject_rules.py` — `SubobjectRules` frozen
  dataclass + canonical hash.
- `oneuniverse/data/subobject.py` — `SubobjectLinks` container,
  sidecar path helpers, bitemporal manifest I/O, pair-builder
  `_build_subobject_pairs`, public `build_subobject_links`,
  `load_subobject_links`, `list_subobject_link_sets`.
- `test/test_subobject_rules.py` — rules hash + symmetric pair API.
- `test/test_subobject_build.py` — synthetic SN-in-host build end to
  end + confidence + ambiguity.
- `test/test_subobject_bitemporal.py` — rebuild archives previous,
  `as_of` resolution.
- `test/test_subobject_query.py` — `children_of` / `parent_of`.
- `test/test_visual_subobject.py` — diagnostic skymap with link lines.

Files to modify:

- `oneuniverse/data/database.py` — add `build_subobject_links`,
  `load_subobject_links`, `list_subobject_link_sets`.
- `oneuniverse/data/__init__.py` — export `SubobjectRules`,
  `SubobjectLinks`, `build_subobject_links`, `load_subobject_links`.
- `plans/README.md` — add Phase 8 row.

---

## Task 1: `SubobjectRules` frozen dataclass

**Files:**
- Create: `oneuniverse/data/subobject_rules.py`
- Create: `test/test_subobject_rules.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_subobject_rules.py
"""Unit tests for SubobjectRules canonicalisation and hashing."""
import pickle

import pytest

from oneuniverse.data.subobject_rules import SubobjectRules


def test_defaults_valid():
    r = SubobjectRules(
        parent_survey_type="spectroscopic",
        child_survey_type="transient",
    )
    assert r.sky_tol_arcsec == 1.0
    assert r.dz_tol == 5e-3
    assert r.relation == "contains"
    assert r.accept_ambiguous is False


def test_hash_is_deterministic_across_sessions():
    r = SubobjectRules(
        parent_survey_type="spectroscopic",
        child_survey_type="transient",
        sky_tol_arcsec=3.0,
        dz_tol=2e-2,
        relation="hosts",
        accept_ambiguous=True,
    )
    # Same fields → same hash regardless of construction order.
    r2 = pickle.loads(pickle.dumps(r))
    assert r.hash() == r2.hash()
    # Equality is hash-based.
    assert r == r2


def test_hash_sensitive_to_field_changes():
    base = SubobjectRules(
        parent_survey_type="spectroscopic", child_survey_type="transient",
    )
    assert base.hash() != SubobjectRules(
        parent_survey_type="spectroscopic",
        child_survey_type="transient",
        sky_tol_arcsec=2.0,
    ).hash()
    assert base.hash() != SubobjectRules(
        parent_survey_type="spectroscopic",
        child_survey_type="transient",
        relation="hosts",
    ).hash()


def test_empty_parent_or_child_rejected():
    with pytest.raises(ValueError, match="parent_survey_type"):
        SubobjectRules(parent_survey_type="", child_survey_type="transient")
    with pytest.raises(ValueError, match="child_survey_type"):
        SubobjectRules(parent_survey_type="spectroscopic", child_survey_type="")


def test_tolerances_must_be_positive():
    with pytest.raises(ValueError, match="sky_tol_arcsec"):
        SubobjectRules(
            parent_survey_type="spectroscopic",
            child_survey_type="transient",
            sky_tol_arcsec=-1.0,
        )
    with pytest.raises(ValueError, match="dz_tol"):
        SubobjectRules(
            parent_survey_type="spectroscopic",
            child_survey_type="transient",
            dz_tol=-0.001,
        )


def test_dz_tol_none_disables_z_cut():
    r = SubobjectRules(
        parent_survey_type="spectroscopic",
        child_survey_type="transient",
        dz_tol=None,
    )
    assert r.dz_tol is None
    # Hash still stable.
    assert r.hash() == r.hash()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/test_subobject_rules.py -v`
Expected: FAIL — `ModuleNotFoundError: oneuniverse.data.subobject_rules`.

- [ ] **Step 3: Write minimal implementation**

```python
# oneuniverse/data/subobject_rules.py
"""
oneuniverse.data.subobject_rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:class:`SubobjectRules` — policy for building sub-object links between
a *parent* survey (galaxies, clusters) and a *child* survey (SNe,
TDEs). Symmetric in spirit to :class:`CrossMatchRules`, but records a
relation of *containment*, not identity, and is directional
(parent → child).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, eq=False)
class SubobjectRules:
    """Directional parent→child link policy.

    Parameters
    ----------
    parent_survey_type
        ``survey_type`` tag of the parent loader (e.g. ``"spectroscopic"``
        for host galaxies).
    child_survey_type
        ``survey_type`` tag of the child loader (e.g. ``"transient"``
        for SNe, ``"photometric"`` for satellite galaxies).
    sky_tol_arcsec
        On-sky separation tolerance (arcsec). A child is a candidate
        sub-object of a parent iff angular separation is ≤ this value.
    dz_tol
        Tolerance on ``|z_parent − z_child|``. ``None`` disables the
        redshift check (useful when the child survey has no redshift,
        e.g. early-alert transients).
    relation
        Free-form label recorded on every link row. Default
        ``"contains"``. Examples: ``"hosts"`` (galaxy → SN),
        ``"member_of"`` (galaxy → cluster — but pass it as
        ``child_survey_type = "cluster"`` with swapped parent/child).
    accept_ambiguous
        If ``False`` (default), a child with more than one parent
        candidate within tolerance is dropped entirely. If ``True``,
        *all* candidate pairs are recorded, each with
        ``confidence < 1``.
    """

    parent_survey_type: str
    child_survey_type: str
    sky_tol_arcsec: float = 1.0
    dz_tol: Optional[float] = 5e-3
    relation: str = "contains"
    accept_ambiguous: bool = False

    # ── Validation ───────────────────────────────────────────────────

    def __post_init__(self) -> None:
        if not self.parent_survey_type:
            raise ValueError("SubobjectRules: parent_survey_type must be non-empty")
        if not self.child_survey_type:
            raise ValueError("SubobjectRules: child_survey_type must be non-empty")
        if self.sky_tol_arcsec <= 0.0:
            raise ValueError(
                f"SubobjectRules: sky_tol_arcsec must be positive, "
                f"got {self.sky_tol_arcsec!r}"
            )
        if self.dz_tol is not None and self.dz_tol < 0.0:
            raise ValueError(
                f"SubobjectRules: dz_tol must be non-negative or None, "
                f"got {self.dz_tol!r}"
            )
        if not self.relation:
            raise ValueError("SubobjectRules: relation must be non-empty")

    # ── Hashing / serialisation ──────────────────────────────────────

    def _canonical(self) -> dict:
        return {
            "parent_survey_type": self.parent_survey_type,
            "child_survey_type": self.child_survey_type,
            "sky_tol_arcsec": float(self.sky_tol_arcsec),
            "dz_tol": None if self.dz_tol is None else float(self.dz_tol),
            "relation": self.relation,
            "accept_ambiguous": bool(self.accept_ambiguous),
        }

    def hash(self) -> str:
        """Short sha256 (16 hex chars) over the canonical form."""
        payload = json.dumps(self._canonical(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def __hash__(self) -> int:  # pragma: no cover - trivial
        return hash(self.hash())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SubobjectRules):
            return NotImplemented
        return self.hash() == other.hash()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/test_subobject_rules.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/subobject_rules.py test/test_subobject_rules.py
git commit -m "feat(subobject): SubobjectRules frozen dataclass with canonical hash"
```

---

## Task 2: `SubobjectLinks` container + sidecar path helpers

**Files:**
- Create: `oneuniverse/data/subobject.py` (first half — container + paths only)

- [ ] **Step 1: Write the failing test**

```python
# test/test_subobject_build.py  (first block only — more added in Task 5)
"""Unit tests for SubobjectLinks container + sidecar path helpers."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oneuniverse.data.subobject import (
    SUBOBJECT_DIR,
    SubobjectLinks,
    _links_path,
    _links_manifest_path,
)
from oneuniverse.data.subobject_rules import SubobjectRules
from oneuniverse.data.validity import DatasetValidity


_VALIDITY = DatasetValidity(valid_from_utc="2026-04-20T00:00:00+00:00")


def _toy_links_df():
    return pd.DataFrame(
        {
            "parent_oneuid": np.array([0, 1, 2], dtype=np.int64),
            "child_oneuid": np.array([100, 101, 102], dtype=np.int64),
            "confidence": np.array([1.0, 0.7, 1.0], dtype=np.float32),
            "sky_sep_arcsec": np.array([0.3, 0.9, 0.1], dtype=np.float32),
            "dz": np.array([1e-4, 4e-3, np.nan], dtype=np.float32),
        }
    )


def test_links_path_layout(tmp_path):
    assert _links_path(tmp_path, "sne_in_hosts") == (
        tmp_path / SUBOBJECT_DIR / "sne_in_hosts.parquet"
    )
    assert _links_manifest_path(tmp_path, "sne_in_hosts") == (
        tmp_path / SUBOBJECT_DIR / "sne_in_hosts.manifest.json"
    )


def test_subobject_links_table_shape():
    rules = SubobjectRules(
        parent_survey_type="spectroscopic",
        child_survey_type="transient",
    )
    links = SubobjectLinks(
        name="sne_in_hosts",
        rules=rules,
        parent_datasets=("spec_desi",),
        child_datasets=("transient_pantheon",),
        oneuid_name="default",
        oneuid_hash="abcd" * 4,
        validity=_VALIDITY,
        table=_toy_links_df(),
    )
    assert len(links) == 3
    # Required columns.
    for c in (
        "parent_oneuid", "child_oneuid", "confidence",
        "sky_sep_arcsec", "dz",
    ):
        assert c in links.table.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/test_subobject_build.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

```python
# oneuniverse/data/subobject.py  (PART 1 — container + path helpers)
"""
oneuniverse.data.subobject
~~~~~~~~~~~~~~~~~~~~~~~~~~
**Sub-object links** — bitemporal sidecar tables recording parent→child
containment between astrophysical objects (e.g. host galaxy → SN).

Layout
------
::

    {database_root}/_subobject/<name>.parquet
    {database_root}/_subobject/<name>.manifest.json

Schema (one row per (parent, child) pair):

    parent_oneuid     int64    ONEUID of the parent (host) object
    child_oneuid      int64    ONEUID of the child (e.g. SN) object
    confidence        float32  1.0 unambiguous; <1 for accepted
                               ambiguous matches
    sky_sep_arcsec    float32  on-sky separation at match time
    dz                float32  ``z_parent − z_child``; ``NaN`` when
                               the child has no redshift

The manifest JSON mirrors the ONEUID-index manifest: it carries the
``SubobjectRules`` policy, the parent/child dataset lists, the ONEUID
index name and hash, and a :class:`DatasetValidity` block.
"""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from oneuniverse.data.subobject_rules import SubobjectRules
from oneuniverse.data.validity import DatasetValidity


logger = logging.getLogger(__name__)

SUBOBJECT_DIR = "_subobject"
SUBOBJECT_MANIFEST_FORMAT_VERSION = 1
REQUIRED_COLUMNS: Tuple[str, ...] = (
    "parent_oneuid",
    "child_oneuid",
    "confidence",
    "sky_sep_arcsec",
    "dz",
)


def _links_path(root: Path, name: str) -> Path:
    return Path(root) / SUBOBJECT_DIR / f"{name}.parquet"


def _links_manifest_path(root: Path, name: str) -> Path:
    return Path(root) / SUBOBJECT_DIR / f"{name}.manifest.json"


@dataclass(frozen=True)
class SubobjectLinks:
    """In-memory view of a named sub-object link sidecar."""

    name: str
    rules: SubobjectRules
    parent_datasets: Tuple[str, ...]
    child_datasets: Tuple[str, ...]
    oneuid_name: str
    oneuid_hash: str
    validity: DatasetValidity
    table: pd.DataFrame

    def __post_init__(self) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in self.table.columns]
        if missing:
            raise ValueError(
                f"SubobjectLinks: table missing required columns {missing!r}"
            )

    def __len__(self) -> int:
        return len(self.table)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest test/test_subobject_build.py::test_links_path_layout test/test_subobject_build.py::test_subobject_links_table_shape -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/subobject.py test/test_subobject_build.py
git commit -m "feat(subobject): SubobjectLinks container + sidecar path helpers"
```

---

## Task 3: Manifest I/O (read/write) with bitemporal block

**Files:**
- Modify: `oneuniverse/data/subobject.py` — append manifest helpers.
- Modify: `test/test_subobject_build.py` — add roundtrip test.

- [ ] **Step 1: Write the failing test**

Append to `test/test_subobject_build.py`:

```python
def test_manifest_roundtrip(tmp_path):
    rules = SubobjectRules(
        parent_survey_type="spectroscopic",
        child_survey_type="transient",
        sky_tol_arcsec=2.5,
        dz_tol=1e-2,
        relation="hosts",
        accept_ambiguous=False,
    )
    links = SubobjectLinks(
        name="sne_in_hosts",
        rules=rules,
        parent_datasets=("spec_desi",),
        child_datasets=("transient_pantheon",),
        oneuid_name="default",
        oneuid_hash="abcd" * 4,
        validity=_VALIDITY,
        table=_toy_links_df(),
    )
    from oneuniverse.data.subobject import write_subobject_links, read_subobject_links

    write_subobject_links(tmp_path, links)
    assert _links_path(tmp_path, "sne_in_hosts").exists()
    assert _links_manifest_path(tmp_path, "sne_in_hosts").exists()

    loaded = read_subobject_links(tmp_path, "sne_in_hosts")
    assert loaded.name == "sne_in_hosts"
    assert loaded.rules == rules
    assert loaded.parent_datasets == ("spec_desi",)
    assert loaded.child_datasets == ("transient_pantheon",)
    assert loaded.oneuid_hash == "abcd" * 4
    assert loaded.validity == _VALIDITY
    pd.testing.assert_frame_equal(
        loaded.table.reset_index(drop=True),
        links.table.reset_index(drop=True),
        check_dtype=True,
    )


def test_write_is_atomic(tmp_path):
    """Writer must flush to a tmp file then rename — never leave a
    half-written parquet on disk if something fails mid-write."""
    from oneuniverse.data.subobject import write_subobject_links

    rules = SubobjectRules(
        parent_survey_type="spectroscopic",
        child_survey_type="transient",
    )
    links = SubobjectLinks(
        name="partial",
        rules=rules,
        parent_datasets=("a",),
        child_datasets=("b",),
        oneuid_name="default",
        oneuid_hash="x" * 16,
        validity=_VALIDITY,
        table=_toy_links_df(),
    )
    write_subobject_links(tmp_path, links)
    # No leftover tmp files.
    leftovers = list((tmp_path / SUBOBJECT_DIR).glob("*.tmp*"))
    assert leftovers == []


def test_manifest_rejects_unknown_format_version(tmp_path):
    from oneuniverse.data.subobject import read_subobject_links

    man = _links_manifest_path(tmp_path, "bogus")
    man.parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / SUBOBJECT_DIR / "bogus.parquet").write_bytes(b"")
    man.write_text(json.dumps({"format_version": 99}))
    with pytest.raises(ValueError, match="format_version"):
        read_subobject_links(tmp_path, "bogus")
```

Add at top of file: `import json`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test/test_subobject_build.py -v`
Expected: FAIL — `ImportError: cannot import name 'write_subobject_links'`.

- [ ] **Step 3: Append manifest I/O to `subobject.py`**

```python
# ── Manifest I/O ─────────────────────────────────────────────────────


def _rules_to_dict(r: SubobjectRules) -> dict:
    return r._canonical()


def _rules_from_dict(d: dict) -> SubobjectRules:
    return SubobjectRules(
        parent_survey_type=d["parent_survey_type"],
        child_survey_type=d["child_survey_type"],
        sky_tol_arcsec=float(d["sky_tol_arcsec"]),
        dz_tol=None if d["dz_tol"] is None else float(d["dz_tol"]),
        relation=d["relation"],
        accept_ambiguous=bool(d["accept_ambiguous"]),
    )


def _manifest_to_dict(links: SubobjectLinks) -> dict:
    return {
        "format_version": SUBOBJECT_MANIFEST_FORMAT_VERSION,
        "name": links.name,
        "rules": _rules_to_dict(links.rules),
        "rules_hash": links.rules.hash(),
        "parent_datasets": list(links.parent_datasets),
        "child_datasets": list(links.child_datasets),
        "oneuid_name": links.oneuid_name,
        "oneuid_hash": links.oneuid_hash,
        "validity": links.validity.to_dict(),
        "n_links": int(len(links.table)),
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
    }


def write_subobject_links(root: Path, links: SubobjectLinks) -> None:
    """Atomic write: table + manifest under ``{root}/_subobject/``."""
    root = Path(root)
    dest_table = _links_path(root, links.name)
    dest_manifest = _links_manifest_path(root, links.name)
    dest_table.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        dir=dest_table.parent, prefix=f".{links.name}.",
    ) as tmp:
        tmp_path = Path(tmp)
        tmp_table = tmp_path / "links.parquet"
        tmp_manifest = tmp_path / "manifest.json"
        links.table.to_parquet(tmp_table, index=False, compression="zstd")
        tmp_manifest.write_text(json.dumps(
            _manifest_to_dict(links), indent=2, sort_keys=True,
        ))
        shutil.move(str(tmp_table), str(dest_table))
        shutil.move(str(tmp_manifest), str(dest_manifest))


def read_subobject_links(root: Path, name: str) -> SubobjectLinks:
    """Load a named sub-object link sidecar."""
    root = Path(root)
    man_path = _links_manifest_path(root, name)
    tbl_path = _links_path(root, name)
    if not man_path.exists():
        raise FileNotFoundError(
            f"read_subobject_links: manifest missing for {name!r} at {man_path}"
        )
    raw = json.loads(man_path.read_text())
    fmt = raw.get("format_version")
    if fmt != SUBOBJECT_MANIFEST_FORMAT_VERSION:
        raise ValueError(
            f"read_subobject_links: unsupported format_version {fmt!r} "
            f"for {name!r} (expected {SUBOBJECT_MANIFEST_FORMAT_VERSION})"
        )
    table = pd.read_parquet(tbl_path)
    return SubobjectLinks(
        name=raw["name"],
        rules=_rules_from_dict(raw["rules"]),
        parent_datasets=tuple(raw["parent_datasets"]),
        child_datasets=tuple(raw["child_datasets"]),
        oneuid_name=raw["oneuid_name"],
        oneuid_hash=raw["oneuid_hash"],
        validity=DatasetValidity.from_dict(raw["validity"]),
        table=table,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest test/test_subobject_build.py -v`
Expected: 5 passed (the 2 from Task 2 + 3 new).

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/subobject.py test/test_subobject_build.py
git commit -m "feat(subobject): atomic sidecar manifest I/O with DatasetValidity"
```

---

## Task 4: Cross-dataset pair builder (`_build_subobject_pairs`)

This is the core physics of Phase 8. Given a parent catalogue (ra, dec,
z, oneuid) and a child catalogue (ra, dec, z?, oneuid), find candidate
(parent, child) pairs passing the sky + optional redshift cuts, apply
the ambiguity policy, and return a `DataFrame` with
`REQUIRED_COLUMNS`.

We use `scipy.spatial.cKDTree` on Cartesian unit vectors (exact
spherical distance via 3-D chord length), which is the same primitive
already used by the cross-match engine in Phase 4.

**Files:**
- Modify: `oneuniverse/data/subobject.py` — append pair builder.
- Create: `test/test_subobject_build.py` — add builder tests.

- [ ] **Step 1: Write the failing test**

Append:

```python
def _radec_to_unit(ra_deg, dec_deg):
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    cd = np.cos(dec)
    return np.column_stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)])


def test_pair_builder_unambiguous():
    """One host, one SN, well within tol → one link, confidence=1.0."""
    from oneuniverse.data.subobject import _build_subobject_pairs

    parents = pd.DataFrame({
        "oneuid": [10],
        "ra": [15.0], "dec": [0.0],
        "z": [0.05],
    })
    children = pd.DataFrame({
        "oneuid": [200],
        "ra": [15.0 + 0.5 / 3600],  # 0.5" east
        "dec": [0.0],
        "z": [0.051],
    })
    rules = SubobjectRules(
        parent_survey_type="spectroscopic",
        child_survey_type="transient",
        sky_tol_arcsec=1.5, dz_tol=2e-2,
    )
    out = _build_subobject_pairs(parents, children, rules)
    assert len(out) == 1
    row = out.iloc[0]
    assert row.parent_oneuid == 10
    assert row.child_oneuid == 200
    assert row.confidence == pytest.approx(1.0)
    assert row.sky_sep_arcsec < 1.0
    assert abs(row.dz - 1e-3) < 1e-6


def test_pair_builder_rejects_out_of_tolerance():
    from oneuniverse.data.subobject import _build_subobject_pairs

    parents = pd.DataFrame({"oneuid":[10],"ra":[15.0],"dec":[0.0],"z":[0.05]})
    children = pd.DataFrame({
        "oneuid":[200],
        "ra":[15.0 + 5.0 / 3600],  # 5" east — outside 1.5" tol
        "dec":[0.0], "z":[0.051],
    })
    rules = SubobjectRules(
        parent_survey_type="spectroscopic",
        child_survey_type="transient",
        sky_tol_arcsec=1.5, dz_tol=2e-2,
    )
    out = _build_subobject_pairs(parents, children, rules)
    assert len(out) == 0


def test_pair_builder_rejects_dz_outside():
    from oneuniverse.data.subobject import _build_subobject_pairs

    parents = pd.DataFrame({"oneuid":[10],"ra":[15.0],"dec":[0.0],"z":[0.05]})
    children = pd.DataFrame({
        "oneuid":[200],
        "ra":[15.0 + 0.5 / 3600],
        "dec":[0.0], "z":[0.2],  # far in z
    })
    rules = SubobjectRules(
        parent_survey_type="spectroscopic",
        child_survey_type="transient",
        sky_tol_arcsec=1.5, dz_tol=1e-2,
    )
    out = _build_subobject_pairs(parents, children, rules)
    assert len(out) == 0


def test_pair_builder_ambiguity_rejected_by_default():
    """Two hosts within tolerance of the same SN → SN is dropped."""
    from oneuniverse.data.subobject import _build_subobject_pairs

    parents = pd.DataFrame({
        "oneuid":[10, 11],
        "ra":[15.0, 15.0 + 0.4 / 3600],
        "dec":[0.0, 0.0],
        "z":[0.050, 0.052],
    })
    children = pd.DataFrame({
        "oneuid":[200],
        "ra":[15.0 + 0.2 / 3600],  # inside tol of both
        "dec":[0.0], "z":[0.051],
    })
    rules = SubobjectRules(
        parent_survey_type="spectroscopic",
        child_survey_type="transient",
        sky_tol_arcsec=1.5, dz_tol=2e-2, accept_ambiguous=False,
    )
    out = _build_subobject_pairs(parents, children, rules)
    assert len(out) == 0


def test_pair_builder_ambiguity_accepted_with_flag():
    """Same setup; accept_ambiguous=True → both links kept with conf<1."""
    from oneuniverse.data.subobject import _build_subobject_pairs

    parents = pd.DataFrame({
        "oneuid":[10, 11],
        "ra":[15.0, 15.0 + 0.4 / 3600],
        "dec":[0.0, 0.0],
        "z":[0.050, 0.052],
    })
    children = pd.DataFrame({
        "oneuid":[200],
        "ra":[15.0 + 0.2 / 3600],
        "dec":[0.0], "z":[0.051],
    })
    rules = SubobjectRules(
        parent_survey_type="spectroscopic",
        child_survey_type="transient",
        sky_tol_arcsec=1.5, dz_tol=2e-2, accept_ambiguous=True,
    )
    out = _build_subobject_pairs(parents, children, rules)
    assert len(out) == 2
    # Confidences sum to 1 (1/n_candidates weighting).
    assert out["confidence"].sum() == pytest.approx(1.0)
    assert (out["confidence"] < 1.0).all()


def test_pair_builder_missing_child_z():
    """Child with NaN z still matches when dz_tol disables redshift cut."""
    from oneuniverse.data.subobject import _build_subobject_pairs

    parents = pd.DataFrame({"oneuid":[10],"ra":[15.0],"dec":[0.0],"z":[0.05]})
    children = pd.DataFrame({
        "oneuid":[200],
        "ra":[15.0 + 0.5 / 3600], "dec":[0.0],
        "z":[np.nan],
    })
    rules = SubobjectRules(
        parent_survey_type="spectroscopic",
        child_survey_type="transient",
        sky_tol_arcsec=1.5, dz_tol=None,
    )
    out = _build_subobject_pairs(parents, children, rules)
    assert len(out) == 1
    assert np.isnan(out.iloc[0].dz)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test/test_subobject_build.py -v`
Expected: new 6 tests FAIL — `ImportError: cannot import name '_build_subobject_pairs'`.

- [ ] **Step 3: Append pair builder to `subobject.py`**

```python
# ── Pair builder ─────────────────────────────────────────────────────


def _radec_to_unit(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    cd = np.cos(dec)
    return np.column_stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)])


def _chord_tol(sky_tol_arcsec: float) -> float:
    """Chord length (on a unit sphere) corresponding to an arc-length
    tolerance, expressed in arcsec."""
    ang = np.radians(sky_tol_arcsec / 3600.0)
    return 2.0 * np.sin(ang / 2.0)


def _chord_to_arcsec(d: np.ndarray) -> np.ndarray:
    return np.degrees(2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))) * 3600.0


def _build_subobject_pairs(
    parents: pd.DataFrame,
    children: pd.DataFrame,
    rules: SubobjectRules,
) -> pd.DataFrame:
    """Return a link table respecting the rules.

    Expected input columns on both frames: ``oneuid``, ``ra``, ``dec``,
    ``z`` (``z`` may contain ``NaN`` only on the child side when
    ``rules.dz_tol is None``).

    Ambiguous children (multiple parents within tolerance):
        * ``rules.accept_ambiguous is False`` → dropped.
        * ``rules.accept_ambiguous is True``  → all candidates kept,
          ``confidence = 1 / n_candidates``.
    """
    from scipy.spatial import cKDTree

    required = {"oneuid", "ra", "dec", "z"}
    for label, df in (("parents", parents), ("children", children)):
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"_build_subobject_pairs: {label} missing columns {missing!r}"
            )

    if len(parents) == 0 or len(children) == 0:
        return pd.DataFrame(
            {c: pd.Series(dtype=_REQUIRED_DTYPE[c]) for c in REQUIRED_COLUMNS}
        )

    # Work on numpy arrays. Drop rows with non-finite ra/dec on either
    # side (these can't be matched geometrically).
    p_ra = parents["ra"].to_numpy(float)
    p_dec = parents["dec"].to_numpy(float)
    p_z = parents["z"].to_numpy(float)
    p_uid = parents["oneuid"].to_numpy(np.int64)

    c_ra = children["ra"].to_numpy(float)
    c_dec = children["dec"].to_numpy(float)
    c_z = children["z"].to_numpy(float)
    c_uid = children["oneuid"].to_numpy(np.int64)

    p_xyz = _radec_to_unit(p_ra, p_dec)
    c_xyz = _radec_to_unit(c_ra, c_dec)

    tree = cKDTree(p_xyz)
    chord = _chord_tol(rules.sky_tol_arcsec)

    # For each child, find all parent candidates within the chord
    # tolerance.
    neighbours = tree.query_ball_point(c_xyz, r=chord)

    # Apply z cut (skipped if dz_tol is None or child z is NaN).
    parent_idx_out: List[int] = []
    child_idx_out: List[int] = []
    conf_out: List[float] = []
    sep_out: List[float] = []
    dz_out: List[float] = []

    for ci, cand in enumerate(neighbours):
        if not cand:
            continue
        cand_arr = np.asarray(cand, dtype=np.int64)
        if rules.dz_tol is not None and np.isfinite(c_z[ci]):
            dz = np.abs(p_z[cand_arr] - c_z[ci])
            keep = np.isfinite(dz) & (dz <= rules.dz_tol)
            cand_arr = cand_arr[keep]
        if cand_arr.size == 0:
            continue
        if cand_arr.size > 1 and not rules.accept_ambiguous:
            continue
        n_cand = cand_arr.size
        chords = np.linalg.norm(p_xyz[cand_arr] - c_xyz[ci], axis=1)
        seps = _chord_to_arcsec(chords)
        for k, pi in enumerate(cand_arr):
            parent_idx_out.append(int(pi))
            child_idx_out.append(ci)
            conf_out.append(1.0 / n_cand)
            sep_out.append(float(seps[k]))
            if rules.dz_tol is None or not np.isfinite(c_z[ci]):
                dz_out.append(np.nan)
            else:
                dz_out.append(float(p_z[pi] - c_z[ci]))

    return pd.DataFrame(
        {
            "parent_oneuid": np.asarray(
                p_uid[parent_idx_out] if parent_idx_out else [],
                dtype=np.int64,
            ),
            "child_oneuid": np.asarray(
                c_uid[child_idx_out] if child_idx_out else [],
                dtype=np.int64,
            ),
            "confidence": np.asarray(conf_out, dtype=np.float32),
            "sky_sep_arcsec": np.asarray(sep_out, dtype=np.float32),
            "dz": np.asarray(dz_out, dtype=np.float32),
        }
    )


_REQUIRED_DTYPE = {
    "parent_oneuid": np.int64,
    "child_oneuid": np.int64,
    "confidence": np.float32,
    "sky_sep_arcsec": np.float32,
    "dz": np.float32,
}
```

- [ ] **Step 4: Run tests**

Run: `pytest test/test_subobject_build.py -v`
Expected: all 11 pass (5 container/manifest + 6 pair builder).

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/subobject.py test/test_subobject_build.py
git commit -m "feat(subobject): cross-dataset pair builder (KDTree sky + Δz + ambiguity policy)"
```

---

## Task 5: Public `build_subobject_links` orchestrator

Wires pair-builder to the database: pulls the parent/child rows from
`DatasetView`, joins them with the covering ONEUID index, calls
`_build_subobject_pairs` once per (parent_dataset × child_dataset)
combination, concatenates, and writes the sidecar. Archives any
previous version of the same link set (mirroring the Phase 7 ONEUID
rebuild logic).

**Files:**
- Modify: `oneuniverse/data/subobject.py` — append orchestrator.
- Modify: `oneuniverse/data/database.py` — expose
  `build_subobject_links`.
- Modify: `oneuniverse/data/__init__.py` — add exports.
- Create: `test/test_subobject_build.py` — extend with end-to-end
  synthetic test.

- [ ] **Step 1: Write the failing test**

Append:

```python
def _synthetic_host_catalog(tmp_path, name, n_host=5, seed=0):
    """Write a tiny POINT dataset of host galaxies to disk and return
    its ouf path."""
    from oneuniverse.data.converter import write_ouf_dataset
    from oneuniverse.data.format_spec import DataGeometry

    rng = np.random.default_rng(seed)
    ra = rng.uniform(10.0, 20.0, n_host)
    dec = rng.uniform(-5.0, 5.0, n_host)
    z = rng.uniform(0.02, 0.1, n_host)
    df = pd.DataFrame({
        "ra": ra, "dec": dec, "z": z,
        "z_spec_err": np.full(n_host, 1e-4),
        "zwarning": np.zeros(n_host, dtype=np.int64),
        "z_type": np.array(["spec"] * n_host),
        "galaxy_id": np.arange(n_host, dtype=np.int64),
        "survey_id": np.zeros(n_host, dtype=np.int64),
        "z_spec": z.copy(),
        "_original_row_index": np.arange(n_host, dtype=np.int64),
    })
    ouf = tmp_path / name / "oneuniverse"
    write_ouf_dataset(
        ouf, df,
        survey_name=name, survey_type="spectroscopic",
        geometry=DataGeometry.POINT,
    )
    return ouf


def _synthetic_sn_catalog(tmp_path, name, parent_radec_z, seed=0):
    """Write a SN POINT dataset co-located with the given host positions
    (SN near each host, one extra isolated SN)."""
    from oneuniverse.data.converter import write_ouf_dataset
    from oneuniverse.data.format_spec import DataGeometry

    rng = np.random.default_rng(seed)
    offsets_arcsec = rng.uniform(0.1, 0.6, len(parent_radec_z))
    ra = parent_radec_z["ra"].to_numpy() + offsets_arcsec / 3600.0
    dec = parent_radec_z["dec"].to_numpy()
    z = parent_radec_z["z"].to_numpy() + rng.normal(0.0, 3e-4, len(ra))

    # Add one isolated SN far from any host.
    ra = np.append(ra, 200.0)
    dec = np.append(dec, 45.0)
    z = np.append(z, 0.15)

    n = len(ra)
    df = pd.DataFrame({
        "ra": ra, "dec": dec, "z": z,
        "z_spec_err": np.full(n, 1e-3),
        "zwarning": np.zeros(n, dtype=np.int64),
        "z_type": np.array(["spec"] * n),
        "galaxy_id": np.arange(n, dtype=np.int64),
        "survey_id": np.ones(n, dtype=np.int64),
        "z_spec": z.copy(),
        "_original_row_index": np.arange(n, dtype=np.int64),
    })
    ouf = tmp_path / name / "oneuniverse"
    write_ouf_dataset(
        ouf, df,
        survey_name=name, survey_type="transient",
        geometry=DataGeometry.POINT,
    )
    return ouf


def test_build_subobject_links_end_to_end(tmp_path):
    from oneuniverse.data.database import OneuniverseDatabase
    from oneuniverse.data.oneuid_rules import CrossMatchRules

    # Build two datasets under a shared root.
    root = tmp_path / "db"
    root.mkdir()
    _synthetic_host_catalog(root, "host_galaxies", n_host=5, seed=0)
    # Need the host ra/dec/z to generate co-located SNe.
    host_df = pd.read_parquet(
        root / "host_galaxies" / "oneuniverse",
        columns=["ra", "dec", "z"],
    )
    _synthetic_sn_catalog(root, "sne", host_df, seed=0)

    db = OneuniverseDatabase(root)
    assert set(db.list().keys()) == {"host_galaxies", "sne"}

    # Need a ONEUID index covering both datasets before linking.
    db.build_oneuid(
        datasets=["host_galaxies", "sne"],
        rules=CrossMatchRules(
            sky_tol_arcsec=0.05,       # so hosts and SNe never merge
            reject_ztype=frozenset({("spec", "spec")}),
        ),
        name="default",
    )

    rules = SubobjectRules(
        parent_survey_type="spectroscopic",
        child_survey_type="transient",
        sky_tol_arcsec=2.0, dz_tol=1e-2,
        relation="hosts", accept_ambiguous=False,
    )
    n = db.build_subobject_links(
        rules=rules,
        parent_datasets=["host_galaxies"],
        child_datasets=["sne"],
        name="sne_in_hosts",
        oneuid_name="default",
    )
    assert n == 5  # every host got one SN; the isolated SN has no match

    loaded = db.load_subobject_links("sne_in_hosts")
    assert len(loaded) == 5
    assert loaded.parent_datasets == ("host_galaxies",)
    assert loaded.child_datasets == ("sne",)
    assert loaded.oneuid_name == "default"
    assert loaded.validity.is_current()
    # Every confidence is 1 (no ambiguity).
    assert (loaded.table["confidence"] == 1.0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest test/test_subobject_build.py::test_build_subobject_links_end_to_end -v`
Expected: FAIL — `OneuniverseDatabase` has no `build_subobject_links`.

- [ ] **Step 3: Append orchestrator to `subobject.py`**

```python
# ── Public builder ───────────────────────────────────────────────────


def _load_oneuid_for(
    database, oneuid_name: str, datasets: Sequence[str],
) -> pd.DataFrame:
    """Load a ONEUID index and keep only rows for *datasets*."""
    idx = database.load_oneuid(oneuid_name).restrict_to(list(datasets))
    # Required columns from the index: oneuid, dataset, row_index,
    # ra, dec, z.
    return idx.to_dataframe()[
        ["oneuid", "dataset", "row_index", "ra", "dec", "z"]
    ].copy()


def _archive_previous(root: Path, name: str) -> None:
    """Mirror Phase-7 ONEUID archiving. If a live sidecar exists for
    *name*, close its validity at *now* and rename both the table and
    manifest with a timestamp suffix."""
    live_man = _links_manifest_path(root, name)
    if not live_man.exists():
        return
    raw = json.loads(live_man.read_text())
    now_iso = datetime.now(tz=timezone.utc).isoformat()
    validity = DatasetValidity.from_dict(raw["validity"]).closed_at(now_iso)
    raw["validity"] = validity.to_dict()

    # Build archive name: <name>__<timestamp>.{parquet,manifest.json}.
    ts = now_iso.replace(":", "").replace("+00:00", "Z")
    archive_base = f"{name}__{ts}"
    archive_table = _links_path(root, archive_base)
    archive_man = _links_manifest_path(root, archive_base)

    live_table = _links_path(root, name)
    # Rename table first (atomic rename on POSIX), then archived
    # manifest carries the closed validity.
    shutil.move(str(live_table), str(archive_table))
    archive_man.write_text(json.dumps(raw, indent=2, sort_keys=True))
    live_man.unlink()


def build_subobject_links(
    database,
    *,
    rules: SubobjectRules,
    parent_datasets: Sequence[str],
    child_datasets: Sequence[str],
    name: str = "default",
    oneuid_name: str = "default",
) -> int:
    """Build + persist a named sub-object link sidecar.

    Returns the number of link rows written.
    """
    root = Path(database._root)
    parent_datasets = tuple(parent_datasets)
    child_datasets = tuple(child_datasets)
    if not parent_datasets:
        raise ValueError("build_subobject_links: parent_datasets is empty")
    if not child_datasets:
        raise ValueError("build_subobject_links: child_datasets is empty")

    # Survey-type guard.
    for d in parent_datasets:
        st = database.entry(d).manifest.survey_type
        if st != rules.parent_survey_type:
            raise ValueError(
                f"build_subobject_links: parent dataset {d!r} has "
                f"survey_type={st!r}, rules require "
                f"{rules.parent_survey_type!r}"
            )
    for d in child_datasets:
        st = database.entry(d).manifest.survey_type
        if st != rules.child_survey_type:
            raise ValueError(
                f"build_subobject_links: child dataset {d!r} has "
                f"survey_type={st!r}, rules require "
                f"{rules.child_survey_type!r}"
            )

    # Load ONEUID, keep only the involved datasets.
    involved = list(parent_datasets) + list(child_datasets)
    parent_idx = _load_oneuid_for(database, oneuid_name, parent_datasets)
    child_idx = _load_oneuid_for(database, oneuid_name, child_datasets)

    # Run pair-matching in one shot per (parent, child) combination.
    # The KDTree scales well enough here; child datasets are small.
    frames: List[pd.DataFrame] = []
    for p_name in parent_datasets:
        parents = parent_idx[parent_idx["dataset"] == p_name].drop_duplicates(
            subset="oneuid"
        )
        for c_name in child_datasets:
            children = child_idx[child_idx["dataset"] == c_name].drop_duplicates(
                subset="oneuid"
            )
            frames.append(_build_subobject_pairs(parents, children, rules))

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        {c: pd.Series(dtype=_REQUIRED_DTYPE[c]) for c in REQUIRED_COLUMNS}
    )

    # Resolve ONEUID manifest hash for audit (if it doesn't expose a
    # hash helper, fall back to the rules hash).
    try:
        oneuid_hash = database.load_oneuid(oneuid_name).rules.hash()
    except Exception:  # pragma: no cover - defensive
        oneuid_hash = "unknown"

    links = SubobjectLinks(
        name=name,
        rules=rules,
        parent_datasets=parent_datasets,
        child_datasets=child_datasets,
        oneuid_name=oneuid_name,
        oneuid_hash=oneuid_hash,
        validity=DatasetValidity(
            valid_from_utc=datetime.now(tz=timezone.utc).isoformat(),
            version=name,
        ),
        table=combined,
    )

    _archive_previous(root, name)
    write_subobject_links(root, links)
    logger.info(
        "subobject: wrote %d links to %s (rules hash %s)",
        len(links), _links_path(root, name), rules.hash(),
    )
    return len(links)
```

Add to `oneuniverse/data/database.py` (inside `OneuniverseDatabase`):

```python
    def build_subobject_links(
        self,
        *,
        rules: "SubobjectRules",
        parent_datasets: Sequence[str],
        child_datasets: Sequence[str],
        name: str = "default",
        oneuid_name: str = "default",
    ) -> int:
        from oneuniverse.data.subobject import build_subobject_links as _build
        return _build(
            self, rules=rules,
            parent_datasets=parent_datasets,
            child_datasets=child_datasets,
            name=name, oneuid_name=oneuid_name,
        )

    def load_subobject_links(
        self,
        name: str = "default",
        *,
        as_of: "Optional[datetime]" = None,
    ) -> "SubobjectLinks":
        from oneuniverse.data.subobject import load_subobject_links as _load
        return _load(Path(self._root), name=name, as_of=as_of)

    def list_subobject_link_sets(
        self, *, include_archived: bool = False,
    ) -> List[str]:
        from oneuniverse.data.subobject import list_subobject_link_sets as _list
        return _list(Path(self._root), include_archived=include_archived)
```

Add the imports `from typing import Sequence` and
`from oneuniverse.data.subobject_rules import SubobjectRules` at the
top of `database.py` if not already present.

Edit `oneuniverse/data/__init__.py`:

```python
from oneuniverse.data.subobject_rules import SubobjectRules  # noqa: F401
from oneuniverse.data.subobject import (  # noqa: F401
    SubobjectLinks,
    build_subobject_links,
    load_subobject_links,
    list_subobject_link_sets,
)
```

- [ ] **Step 4: Run test**

Run: `pytest test/test_subobject_build.py -v`
Expected: all previous + end-to-end = 12 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/subobject.py oneuniverse/data/database.py oneuniverse/data/__init__.py test/test_subobject_build.py
git commit -m "feat(subobject): database.build_subobject_links orchestrator + archival on rebuild"
```

---

## Task 6: `load_subobject_links` and `list_subobject_link_sets` with `as_of`

Parallel to the Phase 7 `load_oneuid_index` bitemporal resolver.

**Files:**
- Modify: `oneuniverse/data/subobject.py` — append resolver + list.
- Create: `test/test_subobject_bitemporal.py`.

- [ ] **Step 1: Write the failing test**

```python
# test/test_subobject_bitemporal.py
"""Tests that rebuilding a link set archives the previous one and that
``load_subobject_links(as_of=T)`` resolves the correct version."""
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oneuniverse.data.database import OneuniverseDatabase
from oneuniverse.data.oneuid_rules import CrossMatchRules
from oneuniverse.data.subobject_rules import SubobjectRules


# reuse synthetic catalog helpers from test_subobject_build.py
from test.test_subobject_build import (
    _synthetic_host_catalog, _synthetic_sn_catalog,
)


def _setup(tmp_path):
    root = tmp_path / "db"
    root.mkdir()
    _synthetic_host_catalog(root, "host_galaxies", n_host=5)
    host_df = pd.read_parquet(
        root / "host_galaxies" / "oneuniverse",
        columns=["ra", "dec", "z"],
    )
    _synthetic_sn_catalog(root, "sne", host_df)
    db = OneuniverseDatabase(root)
    db.build_oneuid(
        datasets=["host_galaxies", "sne"],
        rules=CrossMatchRules(sky_tol_arcsec=0.05),
        name="default",
    )
    return db


def test_rebuild_archives_previous(tmp_path):
    db = _setup(tmp_path)
    r1 = SubobjectRules(
        parent_survey_type="spectroscopic",
        child_survey_type="transient",
        sky_tol_arcsec=2.0, dz_tol=1e-2,
    )
    db.build_subobject_links(
        rules=r1, parent_datasets=["host_galaxies"],
        child_datasets=["sne"], name="sne_in_hosts",
    )

    assert db.list_subobject_link_sets() == ["sne_in_hosts"]
    assert db.list_subobject_link_sets(include_archived=True) == ["sne_in_hosts"]

    # Rebuild with tighter tolerance → some links drop out; previous
    # version must be archived with closed validity.
    r2 = SubobjectRules(
        parent_survey_type="spectroscopic",
        child_survey_type="transient",
        sky_tol_arcsec=0.3, dz_tol=1e-2,
    )
    db.build_subobject_links(
        rules=r2, parent_datasets=["host_galaxies"],
        child_datasets=["sne"], name="sne_in_hosts",
    )

    live = db.list_subobject_link_sets()
    archived = db.list_subobject_link_sets(include_archived=True)
    assert live == ["sne_in_hosts"]
    # Exactly one archived version alongside the live one.
    archive_only = [a for a in archived if "__" in a]
    assert len(archive_only) == 1


def test_as_of_resolves_correct_version(tmp_path):
    db = _setup(tmp_path)
    r1 = SubobjectRules(
        parent_survey_type="spectroscopic",
        child_survey_type="transient",
        sky_tol_arcsec=2.0, dz_tol=1e-2,
    )
    db.build_subobject_links(
        rules=r1, parent_datasets=["host_galaxies"],
        child_datasets=["sne"], name="sne_in_hosts",
    )
    t_mid = dt.datetime.now(tz=dt.timezone.utc)
    r2 = SubobjectRules(
        parent_survey_type="spectroscopic",
        child_survey_type="transient",
        sky_tol_arcsec=0.3, dz_tol=1e-2,
    )
    db.build_subobject_links(
        rules=r2, parent_datasets=["host_galaxies"],
        child_datasets=["sne"], name="sne_in_hosts",
    )

    # Live version has the v2 rules.
    now = db.load_subobject_links("sne_in_hosts")
    assert now.rules == r2

    # As-of at t_mid returns the v1 rules.
    old = db.load_subobject_links("sne_in_hosts", as_of=t_mid)
    assert old.rules == r1


def test_as_of_no_version_raises(tmp_path):
    db = _setup(tmp_path)
    r1 = SubobjectRules(
        parent_survey_type="spectroscopic",
        child_survey_type="transient",
        sky_tol_arcsec=2.0, dz_tol=1e-2,
    )
    db.build_subobject_links(
        rules=r1, parent_datasets=["host_galaxies"],
        child_datasets=["sne"], name="sne_in_hosts",
    )
    # Well before first build → no match.
    ancient = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)
    with pytest.raises(FileNotFoundError, match="valid at"):
        db.load_subobject_links("sne_in_hosts", as_of=ancient)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test/test_subobject_bitemporal.py -v`
Expected: FAIL — `load_subobject_links` does not accept `as_of` /
`list_subobject_link_sets` not found.

- [ ] **Step 3: Append resolver + list to `subobject.py`**

```python
# ── Bitemporal resolver + listing ─────────────────────────────────


def load_subobject_links(
    root: Path,
    *,
    name: str = "default",
    as_of: "Optional[datetime]" = None,
) -> SubobjectLinks:
    """Return the named link set valid at *as_of* (default: current)."""
    root = Path(root)
    if as_of is None:
        return read_subobject_links(root, name)
    if as_of.tzinfo is None:
        raise ValueError(
            "load_subobject_links: as_of must be timezone-aware"
        )

    candidates = [_links_manifest_path(root, name)]
    candidates += sorted(
        (root / SUBOBJECT_DIR).glob(f"{name}__*.manifest.json")
    )
    for man in candidates:
        if not man.exists():
            continue
        raw = json.loads(man.read_text())
        v = DatasetValidity.from_dict(raw["validity"])
        if v.contains(as_of):
            return _read_from_manifest(root, man, raw)
    raise FileNotFoundError(
        f"load_subobject_links: no version of {name!r} valid at {as_of}"
    )


def _read_from_manifest(root: Path, man: Path, raw: dict) -> SubobjectLinks:
    """Materialise a SubobjectLinks from a manifest path + parsed JSON.

    The sibling parquet is inferred from the manifest stem.
    """
    stem = man.stem.removesuffix(".manifest")
    tbl_path = root / SUBOBJECT_DIR / f"{stem}.parquet"
    table = pd.read_parquet(tbl_path)
    return SubobjectLinks(
        name=raw["name"],
        rules=_rules_from_dict(raw["rules"]),
        parent_datasets=tuple(raw["parent_datasets"]),
        child_datasets=tuple(raw["child_datasets"]),
        oneuid_name=raw["oneuid_name"],
        oneuid_hash=raw["oneuid_hash"],
        validity=DatasetValidity.from_dict(raw["validity"]),
        table=table,
    )


def list_subobject_link_sets(
    root: Path, *, include_archived: bool = False,
) -> List[str]:
    root = Path(root)
    sub_dir = root / SUBOBJECT_DIR
    if not sub_dir.exists():
        return []
    live = [
        p.stem.removesuffix(".manifest")
        for p in sub_dir.glob("*.manifest.json")
        if "__" not in p.stem
    ]
    if not include_archived:
        return sorted(live)
    archived = [
        p.stem.removesuffix(".manifest")
        for p in sub_dir.glob("*__*.manifest.json")
    ]
    return sorted(live) + sorted(archived)
```

Required import at top of `subobject.py` (add if not already present):

```python
from datetime import datetime
from typing import Optional
```

- [ ] **Step 4: Run tests**

Run: `pytest test/test_subobject_bitemporal.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/subobject.py test/test_subobject_bitemporal.py
git commit -m "feat(subobject): bitemporal load/list (as_of resolver, archived listing)"
```

---

## Task 7: Query helpers — `children_of` and `parent_of`

**Files:**
- Modify: `oneuniverse/data/subobject.py` — add methods on
  `SubobjectLinks`.
- Create: `test/test_subobject_query.py`.

- [ ] **Step 1: Write the failing test**

```python
# test/test_subobject_query.py
"""Tests for SubobjectLinks.children_of / .parent_of."""
import numpy as np
import pandas as pd
import pytest

from oneuniverse.data.subobject import SubobjectLinks
from oneuniverse.data.subobject_rules import SubobjectRules
from oneuniverse.data.validity import DatasetValidity


_VALIDITY = DatasetValidity(valid_from_utc="2026-04-20T00:00:00+00:00")


def _links():
    # Host 10 has children 200, 201; host 11 has no child; host 12 has
    # one child 202. 203 is ambiguous between 10 and 12.
    df = pd.DataFrame({
        "parent_oneuid": np.array([10, 10, 12, 10, 12], dtype=np.int64),
        "child_oneuid":  np.array([200, 201, 202, 203, 203], dtype=np.int64),
        "confidence":    np.array([1.0, 1.0, 1.0, 0.5, 0.5], dtype=np.float32),
        "sky_sep_arcsec":np.array([0.3, 0.4, 0.2, 0.6, 0.7], dtype=np.float32),
        "dz":            np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    })
    rules = SubobjectRules(
        parent_survey_type="spectroscopic",
        child_survey_type="transient",
        accept_ambiguous=True,
    )
    return SubobjectLinks(
        name="q", rules=rules,
        parent_datasets=("hosts",),
        child_datasets=("sne",),
        oneuid_name="default", oneuid_hash="x" * 16,
        validity=_VALIDITY, table=df,
    )


def test_children_of_returns_children():
    links = _links()
    assert set(links.children_of(10)) == {200, 201, 203}


def test_children_of_missing_parent_returns_empty():
    assert list(_links().children_of(999)) == []


def test_parent_of_unambiguous():
    """Child 200 has exactly one parent → returned as the single int."""
    assert _links().parent_of(200) == 10


def test_parent_of_ambiguous_returns_list():
    """Child 203 has two parents → returned as a list sorted by
    descending confidence."""
    out = _links().parent_of(203)
    assert isinstance(out, list)
    assert set(out) == {10, 12}


def test_parent_of_missing_child_returns_none():
    assert _links().parent_of(9999) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test/test_subobject_query.py -v`
Expected: FAIL — methods not defined.

- [ ] **Step 3: Add methods to `SubobjectLinks`**

Extend the dataclass in `subobject.py`:

```python
    # ── Query helpers ───────────────────────────────────────────────

    def children_of(self, parent_oneuid: int) -> np.ndarray:
        """Return the child ONEUIDs under *parent_oneuid*, ordered by
        descending confidence."""
        mask = self.table["parent_oneuid"].to_numpy() == parent_oneuid
        if not mask.any():
            return np.empty(0, dtype=np.int64)
        sub = self.table.loc[mask].sort_values(
            "confidence", ascending=False,
        )
        return sub["child_oneuid"].to_numpy(dtype=np.int64)

    def parent_of(self, child_oneuid: int):
        """Return the parent ONEUID of *child_oneuid*.

        Returns ``None`` if absent, an ``int`` if unambiguous, or a
        ``list[int]`` sorted by descending confidence if the link set
        recorded ambiguous candidates.
        """
        mask = self.table["child_oneuid"].to_numpy() == child_oneuid
        if not mask.any():
            return None
        sub = self.table.loc[mask].sort_values(
            "confidence", ascending=False,
        )
        parents = sub["parent_oneuid"].to_numpy(dtype=np.int64).tolist()
        if len(parents) == 1:
            return parents[0]
        return parents
```

- [ ] **Step 4: Run tests**

Run: `pytest test/test_subobject_query.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/subobject.py test/test_subobject_query.py
git commit -m "feat(subobject): children_of / parent_of query helpers"
```

---

## Task 8: Visual diagnostic — link lines on a skymap

**Files:**
- Create: `test/test_visual_subobject.py`

- [ ] **Step 1: Write the test**

```python
# test/test_visual_subobject.py
"""Diagnostic figure for Phase 8. Writes a Mollweide-style skymap of
host galaxies + SNe with a line per sub-object link. Skipped in headless
CI without matplotlib; output under test/test_output/."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from oneuniverse.data.database import OneuniverseDatabase  # noqa: E402
from oneuniverse.data.oneuid_rules import CrossMatchRules  # noqa: E402
from oneuniverse.data.subobject_rules import SubobjectRules  # noqa: E402

from test.test_subobject_build import (  # noqa: E402
    _synthetic_host_catalog, _synthetic_sn_catalog,
)


OUTPUT_DIR = Path(__file__).parent / "test_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def test_subobject_link_skymap(tmp_path):
    root = tmp_path / "db"
    root.mkdir()
    _synthetic_host_catalog(root, "hosts", n_host=30, seed=1)
    host_df = pd.read_parquet(
        root / "hosts" / "oneuniverse",
        columns=["ra", "dec", "z"],
    )
    _synthetic_sn_catalog(root, "sne", host_df, seed=1)

    db = OneuniverseDatabase(root)
    db.build_oneuid(
        datasets=["hosts", "sne"],
        rules=CrossMatchRules(sky_tol_arcsec=0.05),
        name="default",
    )
    db.build_subobject_links(
        rules=SubobjectRules(
            parent_survey_type="spectroscopic",
            child_survey_type="transient",
            sky_tol_arcsec=2.0, dz_tol=1e-2,
        ),
        parent_datasets=["hosts"],
        child_datasets=["sne"],
        name="sne_in_hosts",
    )
    links = db.load_subobject_links("sne_in_hosts")

    host_idx = db.load_oneuid("default").to_dataframe()
    host_idx = host_idx[host_idx["dataset"] == "hosts"].set_index("oneuid")
    sn_idx = db.load_oneuid("default").to_dataframe()
    sn_idx = sn_idx[sn_idx["dataset"] == "sne"].set_index("oneuid")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(
        host_idx["ra"], host_idx["dec"],
        marker="o", s=40, facecolors="none", edgecolors="C0",
        label=f"hosts (n={len(host_idx)})",
    )
    ax.scatter(
        sn_idx["ra"], sn_idx["dec"],
        marker="*", s=30, c="C3",
        label=f"SNe (n={len(sn_idx)})",
    )
    for _, row in links.table.iterrows():
        p = host_idx.loc[row["parent_oneuid"]]
        c = sn_idx.loc[row["child_oneuid"]]
        ax.plot([p.ra, c.ra], [p.dec, c.dec], "-", lw=0.5,
                alpha=max(0.15, row["confidence"]),
                color="k")

    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_title(
        f"Sub-object links: hosts → SNe  "
        f"(n_links={len(links)}, mean sep="
        f"{links.table['sky_sep_arcsec'].mean():.2f}\")"
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    outfile = OUTPUT_DIR / "subobject_host_sn_skymap.png"
    fig.savefig(outfile, dpi=120)
    plt.close(fig)

    assert outfile.exists() and outfile.stat().st_size > 0
```

- [ ] **Step 2: Run**

Run: `pytest test/test_visual_subobject.py -v`
Expected: 1 passed, `test/test_output/subobject_host_sn_skymap.png`
created.

- [ ] **Step 3: Commit**

```bash
git add test/test_visual_subobject.py
git commit -m "test(subobject): diagnostic skymap figure for host→SN links"
```

---

## Task 9: Plans README + memory update

**Files:**
- Modify: `Packages/oneuniverse/plans/README.md`
- Modify: `~/.claude/projects/-home-ravoux-Documents-Python/memory/project_oneuniverse_stabilisation.md`

- [ ] **Step 1: Update Plans README**

Add rows to the phase-status table:

```markdown
| 7 | Temporal data model — observation-time axis + bitemporal db | **complete (YYYY-MM-DD, NNN/NNN tests green)** |
| 8 | Sub-object hierarchy (bitemporal link sidecars)             | **complete (YYYY-MM-DD, NNN/NNN tests green)** |
```

Add plan file link to the top of the README.

- [ ] **Step 2: Update memory**

Append to `project_oneuniverse_stabilisation.md`:

```markdown
- **Phase 8 complete YYYY-MM-DD** — NNN/NNN tests green. Delivered:
  `SubobjectRules` frozen dataclass (`oneuniverse/data/subobject_rules.py`)
  with canonical hash; `SubobjectLinks` container + atomic bitemporal
  sidecar I/O at `{root}/_subobject/<name>.parquet{,.manifest.json}`
  (`oneuniverse/data/subobject.py`); cross-dataset pair builder using
  `scipy.spatial.cKDTree` on ICRS unit vectors with arcsec→chord
  tolerance conversion; `database.build_subobject_links` /
  `load_subobject_links(as_of=…)` / `list_subobject_link_sets(include_archived=…)`;
  rebuild archives previous version with timestamp suffix and closes
  its validity; query helpers `children_of`/`parent_of` with
  ambiguity-aware return types; diagnostic skymap figure with link
  lines on hosts → SNe.
```

- [ ] **Step 3: Commit**

```bash
git add plans/README.md
git commit -m "docs(plans): mark Phase 8 complete"
```

(Memory file is outside the repo — no commit needed.)

---

## Success criteria

- `SubobjectRules.hash()` stable across Python sessions;
  `__eq__` delegates to the hash.
- Building against two POINT datasets and an existing ONEUID index
  produces a `_subobject/<name>.parquet` sidecar + manifest, readable
  with `database.load_subobject_links(name)`.
- `database.load_subobject_links(name, as_of=T)` resolves the correct
  archived version after a rebuild.
- `children_of(parent_oneuid)` returns an `np.ndarray[int64]` sorted by
  descending confidence; empty array if none.
- `parent_of(child_oneuid)` returns `None` / `int` / `list[int]` based
  on how many candidates were recorded.
- Ambiguous (many-to-one) matches are rejected by default; setting
  `accept_ambiguous=True` records every candidate with `confidence < 1`.
- Survey-type guard raises if parent/child datasets don't match the
  rules' `parent_survey_type` / `child_survey_type`.
- Full test suite green with ≥ 18 new tests (6 rules + 11 build + 3
  bitemporal + 5 query + 1 visual = 26, comfortably ≥ 18).
- Diagnostic figure `test/test_output/subobject_host_sn_skymap.png`
  committed as a visual smoke test (skipped on headless CI).

## Out of scope (again — the Phase 8 boundary)

- Probabilistic host matching (only deterministic sky + Δz tolerance).
- Host-galaxy property inference (stellar mass, SFR, ...).
- SN light-curve fitting or classification.
- Forward models of host-transient correlations (this is `flip`'s
  domain).
- Multi-level hierarchies (cluster ⊃ galaxy ⊃ SN). Each hierarchy
  level is recorded as a separate link set. Transitive closure is a
  query-time concern, not a build-time one.
