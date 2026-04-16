# Phase 4 — Unified ONEUID (rules, subsets, named indices) — Implementation Plan

> **For agentic workers:** Execute task-by-task. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Make ONEUID the sole canonical cross-survey identifier —
rule-driven, z-type-aware, subset-scoped, and storable under a name
so multiple indices can coexist (`default`, `desi_only`,
`spec_only`, …).

**Architecture:** Introduce `CrossMatchRules` as the first-class
policy object (pairwise z-type tolerances + reject list). Move the
cross-match core into `oneuniverse.data` so `weight/` no longer owns
it. Extend `OneuidIndex` with the audit columns needed to replay the
build (`z_type`, `survey_type`, rule hash). Persist indices under
`{root}/_oneuid/{name}.parquet` + sibling `.manifest.json`.

**Tech Stack:** unchanged from Phase 3 (astropy, scipy.sparse,
pyarrow).

**Context — what Phase 1–3 already delivered:**

- `z_type` is a required CORE column on every POINT dataset.
- `DatasetView.scan()` pushes `Cone` / `SkyPatch` / `z_range` filters
  down; we reuse it to feed the cross-matcher only the rows that
  matter.
- `Manifest` carries partitioned datasets with per-partition stats.

---

## File Structure

- Create: `oneuniverse/data/oneuid_rules.py`
  — `CrossMatchRules` dataclass; `sky_tol_arcsec`, per-pair
    `dz_tol`, `reject_ztype` pairs; serialisation / hash.
- Create: `oneuniverse/data/oneuid_crossmatch.py`
  — move `cross_match_surveys` here, adapt to consume
    `CrossMatchRules` and a `z_type` column.
- Modify: `oneuniverse/data/oneuid.py`
  — `OneuidIndex` gains `name`, `rules`, audit columns
    (`z_type`, `survey_type`); `build_oneuid_index(database, *,
    datasets=None, rules=..., name="default", persist=True)`; new
    path layout `{root}/_oneuid/<name>.parquet` +
    `.manifest.json`; `load_oneuid_index(name=...)`;
    `list_oneuids(database)`; `OneuidIndex.restrict_to(datasets)`.
- Modify: `oneuniverse/data/database.py`
  — `build_oneuid(datasets=None, rules=None, name="default")`,
    `load_oneuid(name="default")`, `list_oneuids()`.
- Modify: `oneuniverse/weight/crossmatch.py`
  — reduce to a compatibility shim calling the new module
    (full deletion is Phase 6).
- Create: `test/test_oneuid_rules.py`
- Create: `test/test_oneuid_named.py`

---

## Task 1: `CrossMatchRules` dataclass

**Files:**
- Create: `oneuniverse/data/oneuid_rules.py`
- Create: `test/test_oneuid_rules.py`

- [ ] **Step 1: Write failing test**

```python
from oneuniverse.data.oneuid_rules import CrossMatchRules

def test_default_rules_tol():
    r = CrossMatchRules()
    assert r.sky_tol_arcsec == 1.0
    assert r.dz_tol_for("spec", "spec") == pytest.approx(1e-3)

def test_ztype_override():
    r = CrossMatchRules(dz_tol_by_ztype={("spec", "phot"): 5e-2})
    assert r.dz_tol_for("spec", "phot") == 5e-2
    assert r.dz_tol_for("phot", "spec") == 5e-2   # symmetric
    assert r.dz_tol_for("spec", "spec") == 1e-3   # default unchanged

def test_reject_pairs():
    r = CrossMatchRules(reject_ztype={("phot", "phot")})
    assert r.accepts("phot", "phot") is False
    assert r.accepts("phot", "spec") is True

def test_rule_hash_stable():
    r1 = CrossMatchRules(dz_tol_by_ztype={("spec", "phot"): 5e-2})
    r2 = CrossMatchRules(dz_tol_by_ztype={("phot", "spec"): 5e-2})
    assert r1.hash() == r2.hash()   # order invariant
```

- [ ] **Step 2: Implement**

```python
@dataclass(frozen=True)
class CrossMatchRules:
    sky_tol_arcsec: float = 1.0
    dz_tol_default: Optional[float] = 1e-3
    dz_tol_by_ztype: Mapping[Tuple[str, str], float] = field(
        default_factory=dict,
    )
    reject_ztype: FrozenSet[Tuple[str, str]] = field(
        default_factory=frozenset,
    )

    @staticmethod
    def _key(a: str, b: str) -> Tuple[str, str]:
        return tuple(sorted((a, b)))

    def dz_tol_for(self, ztype_a: str, ztype_b: str) -> Optional[float]:
        k = self._key(ztype_a, ztype_b)
        if k in self.dz_tol_by_ztype:
            return self.dz_tol_by_ztype[k]
        # also check un-sorted caller-supplied keys
        alt = (ztype_a, ztype_b)
        if alt in self.dz_tol_by_ztype:
            return self.dz_tol_by_ztype[alt]
        return self.dz_tol_default

    def accepts(self, ztype_a: str, ztype_b: str) -> bool:
        k = self._key(ztype_a, ztype_b)
        return k not in self.reject_ztype \
            and (ztype_a, ztype_b) not in self.reject_ztype

    def hash(self) -> str:
        payload = json.dumps(self._canonical(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def _canonical(self) -> dict:
        return {
            "sky_tol_arcsec": self.sky_tol_arcsec,
            "dz_tol_default": self.dz_tol_default,
            "dz_tol_by_ztype": sorted(
                (list(self._key(*k)), v)
                for k, v in self.dz_tol_by_ztype.items()
            ),
            "reject_ztype": sorted(list(k) for k in self.reject_ztype),
        }
```

- [ ] **Step 3: Run tests**

---

## Task 2: Move + adapt the cross-matcher

**Files:**
- Create: `oneuniverse/data/oneuid_crossmatch.py`
- Modify: `oneuniverse/weight/crossmatch.py` (make it a shim)

- [ ] **Step 1: Copy `cross_match_surveys` + `CrossMatchResult`
  into `oneuid_crossmatch.py`.** Adapt signature:

```python
def cross_match_surveys(
    catalogs: Dict[str, pd.DataFrame],
    rules: CrossMatchRules,
    *,
    survey_ztype: Optional[Mapping[str, str]] = None,
) -> CrossMatchResult:
    """Cross-match using a :class:`CrossMatchRules` policy."""
```

Inside, use per-row `z_type` when present; otherwise fall back to
`survey_ztype[name]` (one ztype per whole survey); otherwise
`"none"`.

Apply the z-tolerance pair-by-pair using `rules.dz_tol_for(za, zb)`
and drop links where `rules.accepts(za, zb) is False`.

- [ ] **Step 2: `weight/crossmatch.py` shim**

```python
# Kept for API stability. Full removal in Phase 6.
from oneuniverse.data.oneuid_crossmatch import (  # noqa: F401
    CrossMatchResult,
    cross_match_surveys as _cross_match_v1,
)

def cross_match_surveys(catalogs, sky_tol_arcsec=1.0, dz_tol=1e-3,
                        ra_col="ra", dec_col="dec", z_col="z"):
    from oneuniverse.data.oneuid_rules import CrossMatchRules
    rules = CrossMatchRules(sky_tol_arcsec=sky_tol_arcsec,
                            dz_tol_default=dz_tol)
    # Column-rename shim: existing callers pass custom column names.
    renamed = {
        name: df.rename(columns={ra_col: "ra", dec_col: "dec",
                                 z_col: "z"})
        for name, df in catalogs.items()
    }
    return _cross_match_v1(renamed, rules)
```

- [ ] **Step 3: Run existing `test_weight.py` — must stay green.**

---

## Task 3: `OneuidIndex` audit columns + `restrict_to`

**Files:**
- Modify: `oneuniverse/data/oneuid.py`

- [ ] **Step 1: Extend `OneuidIndex` schema**

New columns written on build:

- `z_type` — per-row ztype (propagated from the converted data).
- `survey_type` — from manifest (e.g. `"spectroscopic"`).

Record on the dataclass:

```python
@dataclass
class OneuidIndex:
    table: pd.DataFrame
    n_unique: int
    n_multi: int
    name: str = "default"
    rules: Optional[CrossMatchRules] = None

    def restrict_to(self, datasets: Sequence[str]) -> "OneuidIndex":
        sub = self.table[self.table["dataset"].isin(datasets)].copy()
        # re-assign contiguous oneuids so downstream numpy masks stay tight
        _, inv = np.unique(sub["oneuid"].to_numpy(), return_inverse=True)
        sub["oneuid"] = inv
        counts = sub.groupby("oneuid")["dataset"].nunique()
        n_multi = int((counts > 1).sum())
        return OneuidIndex(
            table=sub.reset_index(drop=True),
            n_unique=int(inv.max()) + 1 if len(sub) else 0,
            n_multi=n_multi,
            name=f"{self.name}.restricted",
            rules=self.rules,
        )
```

- [ ] **Step 2: Tests in `test_oneuid.py`**

```python
def test_restrict_to_subsets_datasets(built_index):
    sub = built_index.restrict_to(["synth_a"])
    assert set(sub.table["dataset"].unique()) == {"synth_a"}
    assert sub.table["oneuid"].min() == 0  # re-indexed

def test_audit_columns_present(built_index):
    assert "z_type" in built_index.table.columns
    assert "survey_type" in built_index.table.columns
```

---

## Task 4: Named indices + on-disk layout

**Files:**
- Modify: `oneuniverse/data/oneuid.py`
- Modify: `oneuniverse/data/database.py`

- [ ] **Step 1: Path helpers**

```python
ONEUID_DIR = "_oneuid"

def _index_path(root: Path, name: str) -> Path:
    return root / ONEUID_DIR / f"{name}.parquet"

def _index_manifest_path(root: Path, name: str) -> Path:
    return root / ONEUID_DIR / f"{name}.manifest.json"
```

- [ ] **Step 2: `build_oneuid_index(database, *, datasets=None,
  rules=None, name="default", persist=True)`**

- If `rules is None`, use `CrossMatchRules()`.
- If `datasets is None`, use every dataset in `database`.
- Load only the rows needed for the match (ra/dec/z/z_type),
  passing through the new `DatasetView.scan(columns=[...])`.
- Persist to `_oneuid/<name>.parquet` with a sibling manifest JSON
  recording: datasets used, rules hash, sky tol, dz tol default,
  n_unique/n_multi, created_utc, code version.

- [ ] **Step 3: `load_oneuid_index(database, name="default")`**

Read both files; validate format version; construct `OneuidIndex`
with populated `rules` (reconstructed from the manifest).

- [ ] **Step 4: `list_oneuids(database) -> List[str]`**

Glob `{root}/_oneuid/*.parquet` and return the stems.

- [ ] **Step 5: Database methods**

```python
def build_oneuid(self, *, datasets=None, rules=None, name="default"):
    return build_oneuid_index(self, datasets=datasets, rules=rules, name=name)

def load_oneuid(self, name="default"):
    return load_oneuid_index(self, name=name)

def list_oneuids(self):
    return list_oneuids(self)
```

- [ ] **Step 6: Back-compat for the old single-file layout**

If `{root}/_oneuid_index.parquet` exists and the new dir does not,
`load_oneuid` should transparently load it and log a deprecation
message. Full removal lands with the Phase 6 cleanup.

- [ ] **Step 7: Tests in `test_oneuid_named.py`**

- Build two indices `default`, `spec_only` — both round-trip
  through disk.
- `list_oneuids()` reports both.
- `load_oneuid("spec_only")` rehydrates a rules object with the
  same hash.

---

## Task 5: `WeightedCatalog` consumes an `OneuidIndex`

**Files:**
- Modify: `oneuniverse/weight/catalog.py` (minimal change; full
  redesign deferred to Phase 6)

- [ ] **Step 1: Additional constructor / helper**

```python
@classmethod
def from_oneuid(cls, index: OneuidIndex, database) -> "WeightedCatalog":
    """Build a WeightedCatalog already keyed on `index.oneuid`."""
    ...
```

- [ ] **Step 2: Deprecate `WeightedCatalog.crossmatch()` self-builder**

Add a `DeprecationWarning` pointing at `from_oneuid`. Actual removal
in Phase 6.

- [ ] **Step 3: Update `test_weight.py` to exercise `from_oneuid`
  alongside the existing path.**

---

## Task 6: Integration + full suite

- [ ] **Step 1: Run the full suite**

`python3 -m pytest test/ -q` — expect 160+ green.

- [ ] **Step 2: Update `plans/README.md` and memory.**

---

## Deferred to Phase 6

- Deletion of `oneuniverse.weight.crossmatch` (now a shim).
- Deletion of `load_universal` (superseded by
  `OneuidQuery.partial_for`).
- Deletion of `WeightedCatalog.crossmatch()` self-builder.
- Deletion of the legacy `_oneuid_index.parquet` single-file path.
