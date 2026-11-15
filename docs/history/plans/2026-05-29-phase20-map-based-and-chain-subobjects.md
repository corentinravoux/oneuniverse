# Phase 20 — Map-Based ONEUID + Multi-Level Sub-Object Chains Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `oneuniverse` cross-match point catalogs against per-row HEALPix probability maps (host-galaxy association for GW events, cluster contamination, depth maps) and walk multi-level sub-object hierarchies transitively (cluster → galaxy → spectrum, deblender parent → child → diaSource).

**Architecture:** Extend `SubobjectRules` with two new optional fields: `relation_type` (semantic role of the edge) and `next_level` (name of the link sidecar that continues the chain). Add a new builder `build_subobject_links_to_map(parent_dataset, event_map_dataset, *, overlap_kind, threshold, map_col, …)` that consumes a point catalog of parents + an event catalog whose rows carry a fixed-NSIDE HEALPix probability map as a `list<f4>` payload (Phase 17 plumbing), then emits the usual SubobjectLinks sidecar with `confidence = max-pixel probability` (or fractional overlap). Add `Database.chain_subobjects(start_dataset, relations=[name1, name2, ...])` that walks a sequence of link sidecars in order and returns the union of leaf-level rows reachable from each start row. No OUF format bump (sidecar manifest carries an integer `format_version`; bumped from 1 → 2 with back-compat for absent fields).

**Tech Stack:** Python 3.9+, numpy, pandas, healpy (already in use), pyarrow (Phase 17 mini-language). **No new runtime dependencies** — multi-order MOC HEALPix support (mocpy) is deferred until a concrete consumer arrives.

---

## File Structure

**New files:**
- `oneuniverse/data/subobject_map.py` — `build_subobject_links_to_map`.
- `oneuniverse/data/chain.py` — `chain_subobjects` standalone helper (called by `Database.chain_subobjects`).
- `test/test_subobject_rules_phase20.py` — `relation_type` / `next_level` field tests + (de)serialisation.
- `test/test_subobject_map.py` — map-based builder end-to-end.
- `test/test_database_chain_subobjects.py` — multi-level chain walks.
- `test/test_visual_phase20.py` — GW × galaxy-host diagnostic.

**Modified files:**
- `oneuniverse/data/subobject_rules.py` — add `relation_type` + `next_level` fields + `_canonical`.
- `oneuniverse/data/subobject.py` — `_rules_to_dict` / `_rules_from_dict` round-trip new fields; bump `SUBOBJECT_MANIFEST_FORMAT_VERSION` 1 → 2; reader accepts v1 with default fields.
- `oneuniverse/data/database.py` — add `chain_subobjects` method delegating to `chain.py`.
- `oneuniverse/CLAUDE.md` — note new sub-object chain + map-based builder.
- `plans/README.md` — Phase 20 row.
- `research/schema_generalisation_audit.md` — Phase 20 close-out cross-ref.

---

## Pre-flight

- [ ] **Step 0: Confirm baseline.**

```bash
cd /home/ravoux/Documents/Python/Packages/oneuniverse
pytest -q 2>&1 | tail -3
```

Expected: `472 passed, 1 skipped` (Phase 19 baseline).

---

## Task 1: `SubobjectRules` gains `relation_type` + `next_level`

**Files:**
- Modify: `oneuniverse/data/subobject_rules.py`
- Create: `test/test_subobject_rules_phase20.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_subobject_rules_phase20.py
"""Phase 20 T1 — SubobjectRules gains relation_type + next_level."""
import pytest

from oneuniverse.data.subobject_rules import SubobjectRules


def test_defaults():
    r = SubobjectRules(
        parent_survey_type="cluster", child_survey_type="galaxy",
    )
    assert r.relation_type == "association"
    assert r.next_level is None


def test_explicit_relation_type():
    r = SubobjectRules(
        parent_survey_type="cluster", child_survey_type="galaxy",
        relation_type="containment",
    )
    assert r.relation_type == "containment"


def test_rejects_unknown_relation_type():
    with pytest.raises(ValueError, match="relation_type"):
        SubobjectRules(
            parent_survey_type="a", child_survey_type="b",
            relation_type="bogus",
        )


def test_next_level_chain_pointer():
    r = SubobjectRules(
        parent_survey_type="cluster", child_survey_type="galaxy",
        next_level="galaxy_to_spectrum",
    )
    assert r.next_level == "galaxy_to_spectrum"


def test_hash_includes_new_fields():
    a = SubobjectRules(
        parent_survey_type="cluster", child_survey_type="galaxy",
        relation_type="containment",
    )
    b = SubobjectRules(
        parent_survey_type="cluster", child_survey_type="galaxy",
        relation_type="causality",
    )
    assert a.hash() != b.hash()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_subobject_rules_phase20.py -v
```

Expected: `TypeError: __init__() got an unexpected keyword argument 'relation_type'`.

- [ ] **Step 3: Extend `SubobjectRules`**

In `oneuniverse/data/subobject_rules.py`, replace the dataclass body
and `_canonical` with:

```python
_ALLOWED_RELATION_TYPES = frozenset({
    "containment", "causality", "association",
})


@dataclass(frozen=True, eq=False)
class SubobjectRules:
    parent_survey_type: str
    child_survey_type: str
    sky_tol_arcsec: float = 1.0
    dz_tol: Optional[float] = 5e-3
    relation: str = "contains"
    accept_ambiguous: bool = False
    # Phase 20: semantic role of the edge, and (optional) next chain
    # link to walk transitively via Database.chain_subobjects.
    relation_type: str = "association"
    next_level: Optional[str] = None

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
        if self.relation_type not in _ALLOWED_RELATION_TYPES:
            raise ValueError(
                f"SubobjectRules: unknown relation_type "
                f"{self.relation_type!r}; "
                f"allowed: {sorted(_ALLOWED_RELATION_TYPES)}"
            )

    def _canonical(self) -> dict:
        return {
            "parent_survey_type": self.parent_survey_type,
            "child_survey_type": self.child_survey_type,
            "sky_tol_arcsec": float(self.sky_tol_arcsec),
            "dz_tol": None if self.dz_tol is None else float(self.dz_tol),
            "relation": self.relation,
            "accept_ambiguous": bool(self.accept_ambiguous),
            "relation_type": self.relation_type,
            "next_level": self.next_level,
        }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/test_subobject_rules_phase20.py test/test_subobject_rules.py -v
```

Expected: green; pre-Phase-20 SubobjectRules tests stay green.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/subobject_rules.py \
        test/test_subobject_rules_phase20.py
git commit -m "phase20/T1: SubobjectRules adds relation_type + next_level (back-compat defaults)"
```

---

## Task 2: Sidecar manifest round-trips new fields with v1 back-compat

**Files:**
- Modify: `oneuniverse/data/subobject.py` — bump `SUBOBJECT_MANIFEST_FORMAT_VERSION` 1 → 2; update `_rules_to_dict` / `_rules_from_dict`.

- [ ] **Step 1: Write the failing test (append to existing file)**

Append to `test/test_subobject_rules_phase20.py`:

```python
from pathlib import Path

import numpy as np
import pandas as pd

from oneuniverse.data.subobject import (
    SubobjectLinks,
    read_subobject_links,
    write_subobject_links,
)
from oneuniverse.data.validity import DatasetValidity


def _make_links(rules: SubobjectRules) -> SubobjectLinks:
    table = pd.DataFrame({
        "parent_oneuid": np.array([1, 2], dtype="i8"),
        "child_oneuid":  np.array([10, 11], dtype="i8"),
        "confidence":    np.array([1.0, 0.8], dtype="f4"),
        "sky_sep_arcsec": np.array([0.3, 0.5], dtype="f4"),
        "dz":             np.array([0.0, 1e-4], dtype="f4"),
    })
    return SubobjectLinks(
        name="test_chain",
        rules=rules,
        parent_datasets=("parents",),
        child_datasets=("children",),
        oneuid_name="default",
        oneuid_hash="0123456789abcdef",
        validity=DatasetValidity(
            valid_from_utc="2026-05-29T00:00:00+00:00",
        ),
        table=table,
    )


def test_relation_type_and_next_level_roundtrip(tmp_path):
    rules = SubobjectRules(
        parent_survey_type="cluster",
        child_survey_type="galaxy",
        relation_type="containment",
        next_level="galaxy_to_spectrum",
    )
    links = _make_links(rules)
    write_subobject_links(tmp_path, links)
    read = read_subobject_links(tmp_path, "test_chain")
    assert read.rules.relation_type == "containment"
    assert read.rules.next_level == "galaxy_to_spectrum"


def test_v1_manifest_parses_with_default_relation(tmp_path):
    """A pre-Phase-20 (v1) sidecar must still parse."""
    import json
    rules = SubobjectRules(
        parent_survey_type="cluster",
        child_survey_type="galaxy",
    )
    links = _make_links(rules)
    write_subobject_links(tmp_path, links)

    # Re-write the manifest as v1 (drop the new fields).
    manifest_path = tmp_path / "_subobject" / "test_chain.manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["format_version"] = 1
    payload["rules"].pop("relation_type", None)
    payload["rules"].pop("next_level", None)
    manifest_path.write_text(json.dumps(payload))

    read = read_subobject_links(tmp_path, "test_chain")
    assert read.rules.relation_type == "association"
    assert read.rules.next_level is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_subobject_rules_phase20.py::test_relation_type_and_next_level_roundtrip -v
```

Expected: `KeyError` on missing key in `_rules_from_dict`.

- [ ] **Step 3: Update sidecar serialisation**

In `oneuniverse/data/subobject.py`, bump

```python
SUBOBJECT_MANIFEST_FORMAT_VERSION = 2
```

Replace `_rules_to_dict` and `_rules_from_dict` with:

```python
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
        relation_type=d.get("relation_type", "association"),
        next_level=d.get("next_level"),
    )
```

(The `format_version` key in the manifest stays a non-strict integer
— both `1` and `2` are accepted by the existing reader, and the
fall-back defaults above mean v1 sidecars produce v2 in-memory specs
with the canonical defaults.)

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest test/test_subobject_rules_phase20.py test/test_subobject_rules.py test/test_subobject_bitemporal.py test/test_subobject_query.py -q
```

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/subobject.py test/test_subobject_rules_phase20.py
git commit -m "phase20/T2: sidecar serialisation round-trips relation_type + next_level (v1 back-compat)"
```

---

## Task 3: `build_subobject_links_to_map`

**Files:**
- Create: `oneuniverse/data/subobject_map.py`
- Create: `test/test_subobject_map.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_subobject_map.py
"""Phase 20 T3 — match a point catalog to per-row HEALPix probability maps."""
import healpy as hp
import numpy as np
import pandas as pd
import pytest

from oneuniverse.data.subobject_map import build_subobject_links_to_map


def _gaussian_map(nside: int, ra: float, dec: float, sigma_deg: float):
    npix = hp.nside2npix(nside)
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    target = hp.ang2vec(theta, phi)
    pix = np.arange(npix)
    vecs = np.array(hp.pix2vec(nside, pix, nest=True))
    cos_sep = vecs.T @ target
    sep_rad = np.arccos(np.clip(cos_sep, -1.0, 1.0))
    sigma_rad = np.radians(sigma_deg)
    m = np.exp(-0.5 * (sep_rad / sigma_rad) ** 2)
    m /= m.sum()
    return m.astype("f4")


def test_match_at_map_peak_has_high_confidence():
    nside = 32
    parents = pd.DataFrame({
        "oneuid": np.array([0, 1, 2], dtype="i8"),
        "ra":  np.array([10.0, 20.0, 50.0], dtype="f8"),
        "dec": np.array([0.0, 0.0, 0.0], dtype="f8"),
    })
    map_at_p0 = _gaussian_map(nside, 10.0, 0.0, sigma_deg=2.0)
    events = pd.DataFrame({
        "oneuid": np.array([100], dtype="i8"),
        "skymap": [map_at_p0],
    })
    links = build_subobject_links_to_map(
        parents=parents, events=events,
        map_column="skymap", map_nside=nside, map_nest=True,
        threshold=0.0,
    )
    df = links.table
    # Parent at (10, 0) is in the peak; others lie far away.
    peak_rows = df[df["parent_oneuid"] == 0]
    other_rows = df[df["parent_oneuid"] != 0]
    assert len(peak_rows) == 1
    assert (peak_rows["confidence"].iloc[0]
            > other_rows["confidence"].max())


def test_threshold_drops_rows_below_cut():
    nside = 32
    parents = pd.DataFrame({
        "oneuid": np.array([0, 1], dtype="i8"),
        "ra":  np.array([0.0, 180.0], dtype="f8"),
        "dec": np.array([0.0, 0.0], dtype="f8"),
    })
    map_at_p0 = _gaussian_map(nside, 0.0, 0.0, sigma_deg=2.0)
    events = pd.DataFrame({
        "oneuid": np.array([100], dtype="i8"),
        "skymap": [map_at_p0],
    })
    above = build_subobject_links_to_map(
        parents=parents, events=events,
        map_column="skymap", map_nside=nside, map_nest=True,
        threshold=1e-3,
    )
    parent_ids = set(above.table["parent_oneuid"].tolist())
    assert 0 in parent_ids
    assert 1 not in parent_ids


def test_rejects_wrong_map_length():
    parents = pd.DataFrame({
        "oneuid": np.array([0], dtype="i8"),
        "ra": np.array([0.0], dtype="f8"),
        "dec": np.array([0.0], dtype="f8"),
    })
    events = pd.DataFrame({
        "oneuid": np.array([100], dtype="i8"),
        "skymap": [np.zeros(7, dtype="f4")],
    })
    with pytest.raises(ValueError, match="length"):
        build_subobject_links_to_map(
            parents=parents, events=events,
            map_column="skymap", map_nside=32,
            threshold=0.0,
        )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_subobject_map.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement the builder**

```python
# oneuniverse/data/subobject_map.py
"""Map-based sub-object linker for OUF 2.4.

Cross-match a point catalog of parents against a catalog of events
whose rows carry a per-row HEALPix probability map (variable-length
``list<f4>`` column, see Phase 17). Returns the canonical
:class:`SubobjectLinks` sidecar with ``confidence`` set to the
parent's pixel value in the event map.

Typical use: GW × galaxy host association (each event has a
sky-localisation map, the parents are galaxies in a redshift shell).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import healpy as hp
import numpy as np
import pandas as pd

from oneuniverse.data.subobject import REQUIRED_COLUMNS, SubobjectLinks
from oneuniverse.data.subobject_rules import SubobjectRules
from oneuniverse.data.validity import DatasetValidity


def build_subobject_links_to_map(
    parents: pd.DataFrame,
    events: pd.DataFrame,
    *,
    map_column: str,
    map_nside: int,
    map_nest: bool = True,
    threshold: float = 0.0,
    name: str = "default",
    rules: Optional[SubobjectRules] = None,
    oneuid_name: str = "default",
    oneuid_hash: str = "",
    validity: Optional[DatasetValidity] = None,
) -> SubobjectLinks:
    """Match each ``parents`` row against every ``events[map_column]``
    and emit a :class:`SubobjectLinks` whose ``confidence`` is the
    parent's pixel value in the event map.

    Parameters
    ----------
    parents
        Catalog with ``oneuid``, ``ra``, ``dec`` columns. Parents whose
        pixel value is below ``threshold`` in a given event are dropped.
    events
        Catalog with ``oneuid`` and a ``map_column`` whose rows are
        ``numpy.ndarray[f4]`` of length ``12 * map_nside²``.
    map_nside, map_nest
        Fixed HEALPix NSIDE and ordering of every event map.
    threshold
        Minimum pixel value to record a link. ``0.0`` keeps every
        non-NaN pixel.
    rules
        Optional explicit :class:`SubobjectRules`. Default builds a
        ``relation_type="association"`` rule with sentinel survey types
        (``"map_event" -> "host"``).
    """
    expected_len = 12 * map_nside * map_nside

    parent_ids = parents["oneuid"].to_numpy(dtype="i8")
    parent_ra = parents["ra"].to_numpy(dtype="f8")
    parent_dec = parents["dec"].to_numpy(dtype="f8")
    parent_pix = hp.ang2pix(
        map_nside, parent_ra, parent_dec, nest=map_nest, lonlat=True,
    )

    event_ids = events["oneuid"].to_numpy(dtype="i8")
    maps = events[map_column].to_numpy()

    parent_acc = []
    child_acc = []
    conf_acc = []
    for evt_id, m in zip(event_ids, maps):
        arr = np.asarray(m, dtype="f4")
        if arr.size != expected_len:
            raise ValueError(
                f"event {int(evt_id)}: map length {arr.size} does not "
                f"match expected length {expected_len} for NSIDE="
                f"{map_nside}"
            )
        probs = arr[parent_pix]
        keep = probs >= threshold
        if not keep.any():
            continue
        parent_acc.append(parent_ids[keep])
        child_acc.append(np.full(keep.sum(), int(evt_id), dtype="i8"))
        conf_acc.append(probs[keep].astype("f4"))

    if not parent_acc:
        table = pd.DataFrame({c: pd.Series(dtype="f4") for c in REQUIRED_COLUMNS})
        table["parent_oneuid"] = table["parent_oneuid"].astype("i8")
        table["child_oneuid"] = table["child_oneuid"].astype("i8")
    else:
        parents_flat = np.concatenate(parent_acc)
        children_flat = np.concatenate(child_acc)
        conf_flat = np.concatenate(conf_acc)
        table = pd.DataFrame({
            "parent_oneuid": parents_flat,
            "child_oneuid": children_flat,
            "confidence": conf_flat,
            "sky_sep_arcsec": np.zeros(parents_flat.size, dtype="f4"),
            "dz": np.zeros(parents_flat.size, dtype="f4"),
        })

    rules = rules or SubobjectRules(
        parent_survey_type="map_event",
        child_survey_type="host",
        sky_tol_arcsec=1.0,
        dz_tol=None,
        relation="contains",
        accept_ambiguous=True,
        relation_type="association",
    )
    validity = validity or DatasetValidity(
        valid_from_utc="2026-05-29T00:00:00+00:00",
    )
    return SubobjectLinks(
        name=name,
        rules=rules,
        parent_datasets=("events",),
        child_datasets=("parents",),
        oneuid_name=oneuid_name,
        oneuid_hash=oneuid_hash,
        validity=validity,
        table=table,
    )
```

- [ ] **Step 4: Run the test**

```bash
pytest test/test_subobject_map.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add oneuniverse/data/subobject_map.py test/test_subobject_map.py
git commit -m "phase20/T3: build_subobject_links_to_map — point catalog vs per-row HEALPix probability maps"
```

---

## Task 4: `Database.chain_subobjects`

**Files:**
- Create: `oneuniverse/data/chain.py`
- Modify: `oneuniverse/data/database.py` (add public method)
- Create: `test/test_database_chain_subobjects.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_database_chain_subobjects.py
"""Phase 20 T4 — Database.chain_subobjects walks multi-level links."""
import numpy as np
import pandas as pd
import pytest


def test_chain_two_links_returns_leaves(tmp_path, monkeypatch):
    """cluster → galaxy → spectrum: start at cluster 0, end at the
    union of all spectrum oneuids reachable through galaxy 10 and 11.
    """
    from oneuniverse.data.chain import chain_subobjects_tables

    cluster_to_galaxy = pd.DataFrame({
        "parent_oneuid": np.array([0, 0, 1], dtype="i8"),
        "child_oneuid":  np.array([10, 11, 12], dtype="i8"),
        "confidence":    np.array([1.0, 0.8, 1.0], dtype="f4"),
    })
    galaxy_to_spectrum = pd.DataFrame({
        "parent_oneuid": np.array([10, 10, 11, 12], dtype="i8"),
        "child_oneuid":  np.array([100, 101, 102, 103], dtype="i8"),
        "confidence":    np.array([1.0, 0.9, 1.0, 1.0], dtype="f4"),
    })
    leaves = chain_subobjects_tables(
        starts=[0],
        link_tables=[cluster_to_galaxy, galaxy_to_spectrum],
    )
    assert sorted(leaves) == [100, 101, 102]


def test_chain_three_links_transitive(tmp_path):
    from oneuniverse.data.chain import chain_subobjects_tables

    a_to_b = pd.DataFrame({
        "parent_oneuid": np.array([0], dtype="i8"),
        "child_oneuid":  np.array([1], dtype="i8"),
        "confidence":    np.array([1.0], dtype="f4"),
    })
    b_to_c = pd.DataFrame({
        "parent_oneuid": np.array([1], dtype="i8"),
        "child_oneuid":  np.array([2], dtype="i8"),
        "confidence":    np.array([1.0], dtype="f4"),
    })
    c_to_d = pd.DataFrame({
        "parent_oneuid": np.array([2], dtype="i8"),
        "child_oneuid":  np.array([3], dtype="i8"),
        "confidence":    np.array([1.0], dtype="f4"),
    })
    leaves = chain_subobjects_tables(
        starts=[0],
        link_tables=[a_to_b, b_to_c, c_to_d],
    )
    assert leaves == [3]


def test_chain_dead_end_returns_empty():
    from oneuniverse.data.chain import chain_subobjects_tables

    a_to_b = pd.DataFrame({
        "parent_oneuid": np.array([0], dtype="i8"),
        "child_oneuid":  np.array([1], dtype="i8"),
        "confidence":    np.array([1.0], dtype="f4"),
    })
    b_to_c_empty = pd.DataFrame({
        "parent_oneuid": np.array([], dtype="i8"),
        "child_oneuid":  np.array([], dtype="i8"),
        "confidence":    np.array([], dtype="f4"),
    })
    leaves = chain_subobjects_tables(
        starts=[0],
        link_tables=[a_to_b, b_to_c_empty],
    )
    assert leaves == []


def test_database_chain_subobjects_round_trip(tmp_path):
    """Smoke-test the Database.chain_subobjects facade against a hand-
    written pair of sidecars. Reuses the standard fixture pattern from
    the existing subobject test files.
    """
    pytest.importorskip("pyarrow")
    from oneuniverse.data.database import OneuniverseDatabase
    from oneuniverse.data.subobject import (
        SubobjectLinks, write_subobject_links,
    )
    from oneuniverse.data.subobject_rules import SubobjectRules
    from oneuniverse.data.validity import DatasetValidity

    root = tmp_path
    rules = SubobjectRules(
        parent_survey_type="cluster",
        child_survey_type="galaxy",
        next_level="galaxy_to_spectrum",
    )
    cluster_galaxy = SubobjectLinks(
        name="cluster_to_galaxy",
        rules=rules,
        parent_datasets=("clusters",),
        child_datasets=("galaxies",),
        oneuid_name="default",
        oneuid_hash="",
        validity=DatasetValidity(valid_from_utc="2026-05-29T00:00:00+00:00"),
        table=pd.DataFrame({
            "parent_oneuid": np.array([0], dtype="i8"),
            "child_oneuid":  np.array([10], dtype="i8"),
            "confidence":    np.array([1.0], dtype="f4"),
            "sky_sep_arcsec": np.array([0.1], dtype="f4"),
            "dz":             np.array([0.0], dtype="f4"),
        }),
    )
    galaxy_spectrum = SubobjectLinks(
        name="galaxy_to_spectrum",
        rules=SubobjectRules(
            parent_survey_type="galaxy",
            child_survey_type="spectroscopic",
        ),
        parent_datasets=("galaxies",),
        child_datasets=("spectra",),
        oneuid_name="default",
        oneuid_hash="",
        validity=DatasetValidity(valid_from_utc="2026-05-29T00:00:00+00:00"),
        table=pd.DataFrame({
            "parent_oneuid": np.array([10], dtype="i8"),
            "child_oneuid":  np.array([100], dtype="i8"),
            "confidence":    np.array([1.0], dtype="f4"),
            "sky_sep_arcsec": np.array([0.0], dtype="f4"),
            "dz":             np.array([0.0], dtype="f4"),
        }),
    )
    write_subobject_links(root, cluster_galaxy)
    write_subobject_links(root, galaxy_spectrum)

    db = OneuniverseDatabase(root)
    leaves = db.chain_subobjects(
        starts=[0],
        relations=["cluster_to_galaxy", "galaxy_to_spectrum"],
    )
    assert leaves == [100]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest test/test_database_chain_subobjects.py -v
```

Expected: `ImportError: No module named 'oneuniverse.data.chain'`.

- [ ] **Step 3: Implement `chain.py`**

```python
# oneuniverse/data/chain.py
"""Multi-level sub-object chain walker.

Given a list of ``SubobjectLinks.table`` frames and a starting set of
oneuids, walk the chain end-to-end and return the union of leaf-level
oneuids reachable from any starting row.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

import pandas as pd


def chain_subobjects_tables(
    starts: Iterable[int],
    link_tables: Sequence[pd.DataFrame],
) -> List[int]:
    """Walk *link_tables* in order; return the sorted union of leaf
    ``child_oneuid``s reachable from ``starts``.
    """
    current = set(int(s) for s in starts)
    for table in link_tables:
        if table.empty or not current:
            current = set()
            continue
        mask = table["parent_oneuid"].isin(current)
        current = set(table.loc[mask, "child_oneuid"].astype("int64").tolist())
    return sorted(current)
```

- [ ] **Step 4: Plug into `Database`**

In `oneuniverse/data/database.py`, append a new method to the
`OneuniverseDatabase` class (near `build_subobject_links` / `load_subobject_links`):

```python
    def chain_subobjects(
        self,
        starts: Sequence[int],
        relations: Sequence[str],
        *,
        as_of=None,
    ) -> list:
        """Walk a sequence of named sub-object link sidecars in order.

        Returns the sorted union of leaf-level oneuids reachable from
        ``starts`` after following all ``relations`` in order.
        """
        from oneuniverse.data.chain import chain_subobjects_tables
        link_tables = [
            self.load_subobject_links(name=r, as_of=as_of).table
            for r in relations
        ]
        return chain_subobjects_tables(starts, link_tables)
```

- [ ] **Step 5: Run the test**

```bash
pytest test/test_database_chain_subobjects.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add oneuniverse/data/chain.py oneuniverse/data/database.py \
        test/test_database_chain_subobjects.py
git commit -m "phase20/T4: chain_subobjects walks multi-level link sidecars (cluster→galaxy→spectrum)"
```

---

## Task 5: Visual diagnostic — GW × galaxy hosts

**Files:**
- Create: `test/test_visual_phase20.py`

- [ ] **Step 1: Write the test**

```python
# test/test_visual_phase20.py
"""Phase 20 visual diagnostic — map-based host association + chain."""
from __future__ import annotations

from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from oneuniverse.data.subobject_map import (  # noqa: E402
    build_subobject_links_to_map,
)

OUT = Path(__file__).parent / "test_output"
OUT.mkdir(exist_ok=True)


def _gaussian_map(nside: int, ra: float, dec: float, sigma_deg: float):
    npix = hp.nside2npix(nside)
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    target = hp.ang2vec(theta, phi)
    vecs = np.array(hp.pix2vec(nside, np.arange(npix), nest=True))
    cos_sep = vecs.T @ target
    sep_rad = np.arccos(np.clip(cos_sep, -1.0, 1.0))
    m = np.exp(-0.5 * (sep_rad / np.radians(sigma_deg)) ** 2)
    m /= m.sum()
    return m.astype("f4")


def test_phase20_visual(tmp_path):
    nside = 64
    rng = np.random.default_rng(0)
    n_parents = 1000
    parents = pd.DataFrame({
        "oneuid": np.arange(n_parents, dtype="i8"),
        "ra":  rng.uniform(0.0, 60.0, n_parents).astype("f8"),
        "dec": rng.uniform(-15.0, 15.0, n_parents).astype("f8"),
    })

    events = pd.DataFrame({
        "oneuid": np.array([1000, 1001], dtype="i8"),
        "skymap": [
            _gaussian_map(nside, 20.0, 0.0, sigma_deg=3.0),
            _gaussian_map(nside, 40.0, 5.0, sigma_deg=4.0),
        ],
    })

    links = build_subobject_links_to_map(
        parents=parents, events=events,
        map_column="skymap", map_nside=nside, map_nest=True,
        threshold=1e-5,
    )
    df = links.table

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].scatter(parents["ra"], parents["dec"], s=2, color="0.7",
                  label="parents (all)")
    matched = parents[parents["oneuid"].isin(df["parent_oneuid"])]
    ax[0].scatter(matched["ra"], matched["dec"], s=6, color="tab:red",
                  label="matched")
    ax[0].set_xlabel("RA [deg]")
    ax[0].set_ylabel("Dec [deg]")
    ax[0].set_title("Parents vs map matches")
    ax[0].legend()

    ax[1].hist(df["confidence"], bins=40, color="tab:blue", alpha=0.8)
    ax[1].set_xlabel("confidence (pixel probability)")
    ax[1].set_ylabel("count")
    ax[1].set_title("Match confidence distribution")

    for evt_id in events["oneuid"]:
        sel = df["child_oneuid"] == int(evt_id)
        ax[2].scatter(
            parents.loc[df.loc[sel, "parent_oneuid"], "ra"].values,
            df.loc[sel, "confidence"].values,
            s=4, alpha=0.5, label=f"event {int(evt_id)}",
        )
    ax[2].set_xlabel("parent RA [deg]")
    ax[2].set_ylabel("confidence")
    ax[2].legend()
    ax[2].set_title("Per-event RA × confidence")

    fig.tight_layout()
    out_png = OUT / "phase20_map_based_subobject.png"
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    assert out_png.exists() and out_png.stat().st_size > 30_000
    try:
        from PIL import Image
    except ImportError:
        return
    with Image.open(out_png) as im:
        assert im.width >= 800 and im.height >= 200
```

- [ ] **Step 2: Run the test**

```bash
pytest test/test_visual_phase20.py -v
```

Expected: pass; PNG ≥ 30 kB.

- [ ] **Step 3: Commit**

```bash
git add test/test_visual_phase20.py \
        test/test_output/phase20_map_based_subobject.png
git commit -m "phase20/T5: visual diagnostic — map-based host association"
```

---

## Task 6: Docs + plan-README + audit cross-ref

**Files:**
- Modify: `oneuniverse/CLAUDE.md` — add chain + map-based builder.
- Modify: `plans/README.md`.
- Modify: `research/schema_generalisation_audit.md`.

- [ ] **Step 1: CLAUDE.md**

Under "Bitemporal ONEUID / sub-object" add:

```
- `SubobjectRules` carries `relation_type` ∈
  ``{containment, causality, association}`` and an optional
  `next_level` pointing at the next link sidecar in a chain.
- `Database.chain_subobjects(starts=[…], relations=[name1, name2, …])`
  walks a sequence of link sidecars and returns the union of leaf
  oneuids (cluster → galaxy → spectrum, deblender hierarchies, …).
- `oneuniverse.data.subobject_map.build_subobject_links_to_map`
  matches a point catalog of parents against per-row HEALPix
  probability maps (fixed-NSIDE) and emits the canonical
  `SubobjectLinks` sidecar with ``confidence = pixel value``. Used
  for GW host association.
```

- [ ] **Step 2: plans/README.md**

```
| 20 | Map-based ONEUID / sub-object (point × HEALPix probability map) + multi-level chain walker + relation_type / next_level on SubobjectRules | **complete (2026-05-29, NNN/NNN tests green)** |
```

- [ ] **Step 3: research/schema_generalisation_audit.md**

Replace the existing "Phase 20" bullet with:

```
- **Phase 20 — Map-based sub-object + multi-level chains.** Landed
  2026-05-29. Adds
  `oneuniverse.data.subobject_map.build_subobject_links_to_map`
  (point × HEALPix probability map at fixed NSIDE),
  `Database.chain_subobjects(starts, relations)` walker, and
  `SubobjectRules.relation_type / next_level`. Multi-order MOC HEALPix
  (`mocpy`) deferred. No OUF format bump. See
  [`../plans/2026-05-29-phase20-map-based-and-chain-subobjects.md`](../plans/2026-05-29-phase20-map-based-and-chain-subobjects.md).
```

- [ ] **Step 4: Commit**

```bash
git add oneuniverse/CLAUDE.md plans/README.md \
        research/schema_generalisation_audit.md
git commit -m "docs(phase20): map-based subobject + chain walker + relation_type/next_level"
```

---

## Task 7: Close-out

- [ ] **Step 1: Full suite**

```bash
pytest -q 2>&1 | tail -3
```

Expected: green (Phase 19 baseline 472 + ~15–20 new).

- [ ] **Step 2: Replace `NNN/NNN` in plans/README.md.**

- [ ] **Step 3: Update memory**

Append to
`/home/ravoux/.claude/projects/-home-ravoux-Documents-Python/memory/project_oneuniverse_stabilisation.md`:

```markdown
## Phase 20 — Map-based sub-object + multi-level chains (complete 2026-05-29)

- `SubobjectRules` gains `relation_type` ∈ {containment, causality,
  association} and optional `next_level` chain pointer.
  Sidecar manifest bumped to v2 with v1 back-compat.
- New module `oneuniverse.data.subobject_map`:
  `build_subobject_links_to_map(parents, events, *, map_column,
  map_nside, map_nest=True, threshold)` matches a point catalog
  against per-row HEALPix probability maps at fixed NSIDE and
  emits the canonical `SubobjectLinks` sidecar.
- New module `oneuniverse.data.chain`:
  `chain_subobjects_tables(starts, link_tables)` walks a list of
  link tables transitively.
- `OneuniverseDatabase.chain_subobjects(starts, relations,
  as_of=None)` is the public facade.
- Multi-order MOC HEALPix (`mocpy`) deferred until a concrete
  consumer arrives.
- No OUF format bump.
- Tests: NNN/NNN green.
- Per-phase plan:
  `plans/2026-05-29-phase20-map-based-and-chain-subobjects.md`.
```

- [ ] **Step 4: Final commit**

```bash
git add plans/README.md \
        /home/ravoux/.claude/projects/-home-ravoux-Documents-Python/memory/project_oneuniverse_stabilisation.md
git commit -m "phase20: close-out — map-based subobject + chain walker, NNN tests green"
```

---

## Self-review checklist

- [ ] No cosmology metadata added anywhere.
- [ ] `SubobjectRules(relation_type="bogus")` raises.
- [ ] v1 sidecar manifests still parse with default `relation_type`
      and `next_level`.
- [ ] `build_subobject_links_to_map` raises on wrong map length.
- [ ] `chain_subobjects` returns sorted unique leaf oneuids.
- [ ] Visual PNG ≥ 30 kB.

## Spec-coverage map

| Requirement | Task |
|---|---|
| `relation_type` + `next_level` on `SubobjectRules` | T1, T2 |
| Map-based builder | T3 |
| Multi-level chain walker | T4 |
| Visual diagnostic | T5 |
| Docs | T6 |
| Close-out + memory | T7 |
