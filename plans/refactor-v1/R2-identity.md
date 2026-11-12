# R2 — Identity: versioned sidecars, cross-match, ONEUID, links, Database

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One row identity (`source_row` + file position), a crash-safe versioned sidecar store shared by ONEUID and sub-object links, a pair-wise blocked cross-match with vectorised rules, hydration by `take`, and the `Database` facade over a root of OUF datasets.

**Architecture:** `VersionedSidecar` stores each version of a table + meta in its own directory under `<root>/_<kind>/<name>/`, with a `current.json` pointer swapped atomically last. `match_pair` matches two tables per coarse HEALPix block (cell + neighbour ring) with a KD-tree on unit vectors; `components` turns edges into ONEUIDs with a greedy union-find that refuses same-survey merges unless allowed. The ONEUID table records `(dataset, source_row, partition, offset)` so hydration is `DatasetView.take`.

**Tech Stack:** pyarrow, numpy, scipy (`cKDTree`, `connected_components`), healpy (`get_all_neighbours`).

**Spec:** design §6.3, §6.4, §6.6, §8 item 3, §9b (row-identity regression). Master plan Global Constraints apply.

## Interfaces consumed (R0, R1)

`_core.fs` (`atomic_dir`, `atomic_write_json`), `_core.geometry`, `_core.healpix` (`radec_to_pix`, `coarsen`), `data.manifest` (`read_manifest`, `Validity`), `data.view.DatasetView` (`.iter_partitions`, `.take`, `.table`, `.index`), `data.schema` (`z_types`).

## Interfaces produced

```
oneuniverse.data.sidecar
  SIDECAR_FORMAT_VERSION = 1
  class VersionedSidecar(root: Path, kind: str, name: str)       # directory <root>/_<kind>/<name>/
    .write(table: pa.Table, meta: dict) -> str                    # returns version id "<ISO8601>Z"; meta gets format_version, kind, name, version, valid_from, valid_to=None, n_rows
    .load(*, as_of: datetime | None = None) -> tuple[pa.Table, dict]   # NotFoundError if no version is live at as_of
    .versions() -> list[dict]                                     # metas, oldest first
    .current() -> str | None
    @staticmethod names(root, kind) -> list[str]
  layout: <root>/_<kind>/<name>/<version>/table.parquet + meta.json ; <root>/_<kind>/<name>/current.json {"version": ...}

oneuniverse.data.crossmatch
  @dataclass(frozen=True, slots=True) CrossMatchRules(sky_tol_arcsec: float = 1.0, dz_tol: float | None = 5e-3,
        dz_tol_by_pair: tuple[tuple[str, str, float], ...] = (),      # (z_type_a, z_type_b, tol); symmetric
        reject_pairs: tuple[tuple[str, str], ...] = (),               # z_type pairs never linked; symmetric
        allow_same_survey_merge: bool = False)
    .to_dict() .from_dict() .hash() -> str (16 hex) ; .matrices(z_types: Sequence[str]) -> tuple[np.ndarray[float64 (n,n)], np.ndarray[bool (n,n)]]   # tol (nan = no z cut), accept
  @dataclass Pairs(left: np.ndarray[int64], right: np.ndarray[int64], sep_arcsec: np.ndarray[float64], dz: np.ndarray[float64])
  match_pair(a: pa.Table, b: pa.Table, rules: CrossMatchRules, *, block_nside: int = 8) -> Pairs   # a, b carry ra, dec, z, z_type, hpix32 ; left indexes a, right indexes b
  components(n: int, left, right, sep_arcsec, *, survey_of: np.ndarray[int16], allow_same_survey_merge: bool) -> np.ndarray[int64]   # labels 0..k-1, deterministic

oneuniverse.data.oneuid
  ONEUID_SCHEMA = oneuid i8, dataset dictionary<int16,string>, source_row i8, partition i4, offset i8, ra f8, dec f8, z f8, z_type dictionary<int8,string>
  build_oneuid(db: Database, datasets: Sequence[str], *, rules: CrossMatchRules, name: str = "default", block_nside: int = 8) -> OneuidIndex
  load_oneuid(db, name: str = "default", *, as_of=None) -> OneuidIndex
  class OneuidIndex(table: pa.Table, meta: dict, name: str): .datasets -> list[str] ; .rules -> CrossMatchRules ; .n_objects ; .n_multi ; .multi_only() -> pa.Table ; .restrict_to(datasets) -> OneuidIndex (oneuid values kept) ; .of(oneuid) -> pa.Table
  class OneuidQuery(index: OneuidIndex, db: Database)
    .select(*, cone=None, patch=None, z=None, datasets=None) -> np.ndarray[int64]    # unique oneuids with >=1 member row matching
    .rows(oneuids) -> pa.Table                                                        # index rows, sorted by oneuid then dataset
    .hydrate(oneuids, *, columns=None, datasets=None, batch_size=100_000) -> Iterator[pa.Table]   # per dataset, per batch: columns + "oneuid" + "dataset"
    .hydrate_table(oneuids, *, columns=None, datasets=None) -> pa.Table

oneuniverse.data.links
  @dataclass(frozen=True, slots=True) SubobjectRules(parent_kind: str, child_kind: str, sky_tol_arcsec=1.0, dz_tol: float | None = 5e-3, relation: Literal["contains","associated","caused"]="contains", accept_ambiguous=False) .hash()
  LINKS_SCHEMA = parent_oneuid i8, child_oneuid i8, sep_arcsec f8, dz f8 (nan when either z is nan), confidence f4
  build_links(db, *, parents: Sequence[str], children: Sequence[str], rules, name="default", oneuid_name="default") -> SubobjectLinks   # ValidationError if a dataset's survey.kind != rules.*_kind
  load_links(db, name="default", *, as_of=None) -> SubobjectLinks
  class SubobjectLinks(table, meta, name): .rules .children_of(oneuid) -> np.ndarray[int64] (confidence desc) ; .parent_of(oneuid) -> int | list[int] | None
  chain(db, starts: Sequence[int], names: Sequence[str], *, as_of=None) -> np.ndarray[int64]

oneuniverse.data.database
  class Database(root: Path, *, as_of: datetime | None = None, strict: bool = True)
    .root ; .names() -> list[str] ; [name] -> DatasetView ; __contains__ ; __iter__ ; __len__ ; .manifest(name) ; .path(name)
    .as_of(when) -> Database ; .history(name) -> list[Manifest]   # live + _history manifests, oldest first
    .build_oneuid(datasets, *, rules, name="default") ; .load_oneuid(name="default", *, as_of=None) ; .list_oneuids() ; .oneuid_query(name="default") -> OneuidQuery
    .build_links(*, parents, children, rules, name="default", oneuid_name="default") ; .load_links(name="default", *, as_of=None) ; .list_links() ; .chain(starts, names)
    .summary() -> dict   # json-safe: datasets {name: {n_rows, geometry, kind, version}}, oneuids [names], links [names]
  discovery: every directory under root (depth <= 2) containing manifest.json; dataset name = relative path with "/" -> "__"; directories starting with "_" skipped
```

---

### Task 1: `data.sidecar`

**Files:**
- Create: `oneuniverse/data/sidecar.py`
- Test: `test/data/test_sidecar.py`

- [ ] **Step 1: Write the failing tests**

```python
def _t(n): return pa.table({"a": np.arange(n)})

def test_write_then_load_current(tmp_path):
    sc = VersionedSidecar(tmp_path, "oneuid", "qso"); v = sc.write(_t(3), {"rules": "x"})
    t, meta = sc.load(); assert t.num_rows == 3 and meta["version"] == v and meta["valid_to"] is None and meta["rules"] == "x"
    assert VersionedSidecar.names(tmp_path, "oneuid") == ["qso"] and (tmp_path / "_oneuid/qso/current.json").exists()

def test_rewrite_archives_and_as_of(tmp_path):
    sc = VersionedSidecar(tmp_path, "oneuid", "qso"); v1 = sc.write(_t(1), {}); t_between = datetime.now(timezone.utc); time.sleep(0.01); v2 = sc.write(_t(2), {})
    assert sc.current() == v2 and sc.load()[0].num_rows == 2
    old_t, old_meta = sc.load(as_of=t_between); assert old_t.num_rows == 1 and old_meta["valid_to"] == v2
    assert [m["version"] for m in sc.versions()] == [v1, v2]
    with pytest.raises(NotFoundError): sc.load(as_of=datetime(2000, 1, 1, tzinfo=timezone.utc))

def test_sidecar_crash_between_steps(tmp_path, monkeypatch):
    sc = VersionedSidecar(tmp_path, "oneuid", "qso"); v1 = sc.write(_t(1), {})
    import oneuniverse.data.sidecar as mod
    calls = {"n": 0}
    real = mod.atomic_write_json
    def flaky(path, obj):          # fail when the pointer swap is attempted
        if path.name == "current.json": raise OSError("disk full")
        return real(path, obj)
    monkeypatch.setattr(mod, "atomic_write_json", flaky)
    with pytest.raises(OSError): sc.write(_t(2), {})
    monkeypatch.undo()
    assert sc.current() == v1 and sc.load()[0].num_rows == 1 and sc.load()[1]["valid_to"] is None
    assert len(sc.versions()) == 1   # the orphan directory is not listed as a version

def test_no_version_raises(tmp_path):
    with pytest.raises(NotFoundError): VersionedSidecar(tmp_path, "links", "none").load()
```

- [ ] **Step 2: Run, expect failure**
- [ ] **Step 3: Implement.** Write order: (1) new version dir via `atomic_dir(<name>/<version>)` containing `table.parquet` (zstd) + `meta.json`; (2) swap `current.json` with `atomic_write_json`; (3) patch the previous version's `meta.json` `valid_to = <new version>`. `versions()` lists only directories reachable from the chain `current → meta.valid_from/previous` or, simpler, directories whose `meta.valid_from <= current version`; orphans (version id > current) are ignored and removed on the next successful write. If (3) fails after (2), `load(as_of)` still resolves by version ordering (valid_to is derived as "next version id" when missing), so the test above passes either way; document this in the module docstring.
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** — `feat(data): crash-safe versioned sidecar store`

---

### Task 2: `data.crossmatch`

**Files:**
- Create: `oneuniverse/data/crossmatch.py`
- Test: `test/data/test_crossmatch.py`

- [ ] **Step 1: Write the failing tests**

```python
def _cat(n, seed, *, survey, jitter_arcsec=0.0, z_jitter=0.0, base=None):  # returns pa.Table with ra, dec, z, z_type, hpix32; if base given, perturb base rows
    ...

def test_rules_hash_and_matrices():
    r = CrossMatchRules(dz_tol=1e-2, dz_tol_by_pair=(("spec", "phot", 0.1),), reject_pairs=(("phot", "phot"),))
    assert r.hash() == CrossMatchRules.from_dict(r.to_dict()).hash() and len(r.hash()) == 16
    tol, acc = r.matrices(["spec", "phot", "none"])
    assert tol[0, 1] == tol[1, 0] == 0.1 and tol[0, 0] == 1e-2 and np.isnan(tol[0, 2]) and not acc[1, 1] and acc[0, 1]

def test_match_pair_recovers_true_pairs_across_blocks():
    a = _cat(20_000, 0, survey="a"); b = _cat(20_000, 0, survey="b", jitter_arcsec=0.3, z_jitter=1e-4, base=a)   # same objects, perturbed, shuffled
    p = match_pair(a, b, CrossMatchRules(sky_tol_arcsec=1.0), block_nside=8)
    truth = b.column("source_row").to_numpy()  # fixture stores the matching a-row in source_row
    assert len(p.left) == 20_000 and np.array_equal(np.sort(p.left), np.arange(20_000)) and np.all(truth[p.right] == p.left)
    assert p.sep_arcsec.max() < 1.0

def test_match_pair_dz_and_reject():
    a = _cat(500, 1, survey="a"); b = _cat(500, 1, survey="b", base=a, z_jitter=0.05)
    assert len(match_pair(a, b, CrossMatchRules(dz_tol=1e-3)).left) == 0
    assert len(match_pair(a, b, CrossMatchRules(dz_tol=None)).left) == 500
    b2 = b.set_column(b.schema.get_field_index("z_type"), "z_type", pa.array(["phot"] * 500))
    assert len(match_pair(a, b2, CrossMatchRules(dz_tol=None, reject_pairs=(("spec", "phot"),))).left) == 0

def test_match_pair_block_boundary_and_pole():
    a = pa.table({"ra": [359.9999, 0.0, 10.0], "dec": [0.0, 0.0, 89.999], "z": [1.0, 1.0, 1.0], "z_type": ["spec"] * 3, "hpix32": radec_to_pix(32, [359.9999, 0.0, 10.0], [0.0, 0.0, 89.999])})
    b = pa.table({"ra": [0.0001, 200.0], "dec": [0.0, 89.9995], "z": [1.0, 1.0], "z_type": ["spec"] * 2, "hpix32": radec_to_pix(32, [0.0001, 200.0], [0.0, 89.9995])})
    p = match_pair(a, b, CrossMatchRules(sky_tol_arcsec=2.0)); assert sorted(zip(p.left.tolist(), p.right.tolist())) == [(0, 0), (1, 0), (2, 1)]

def test_components_refuses_same_survey_merge():
    # a1 -- b -- a2 : without allow, b joins the closer a; with allow, all three merge
    survey_of = np.array([0, 1, 0], np.int16); left = np.array([0, 2]); right = np.array([1, 1]); sep = np.array([0.5, 0.2])
    lab = components(3, left, right, sep, survey_of=survey_of, allow_same_survey_merge=False); assert lab[1] == lab[2] and lab[0] != lab[1]
    lab = components(3, left, right, sep, survey_of=survey_of, allow_same_survey_merge=True); assert len(set(lab)) == 1
```

- [ ] **Step 2: Run, expect failure**
- [ ] **Step 3: Implement.** `match_pair`: `block = coarsen(hpix32, 32, block_nside)` for both; for each block cell present in `a`: rows of `a` in the cell vs rows of `b` in the cell ∪ `hp.get_all_neighbours(block_nside, cell, nest=True)`; `cKDTree(unit_vectors_b).query_ball_point(unit_vectors_a, r=chord(sky_tol))` → candidate pairs; vectorised sep (`2 asin(chord/2)` in arcsec), `dz = |z_a - z_b|` (nan if either nan), z-type codes → `tol[code_a, code_b]`, keep if `accept & (isnan(tol) | dz <= tol | (isnan(dz) & allow_no_z))` where rows with `z_type == "none"` or nan z pass the z cut when tol is nan. De-duplicate pairs across neighbouring blocks with `np.unique` on `(left, right)`. `components`: sort edges by `sep`; union-find with per-root bitset/dict of surveys present (`allow_same_survey_merge=False` → skip an edge whose two roots share a survey); labels renumbered by first appearance. `scipy.sparse.csgraph.connected_components` is used only for the `allow=True` branch.
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** — `feat(data): pair-wise blocked cross-match with vectorised z-type rules`

---

### Task 3: `data.database` (discovery and views)

**Files:**
- Create: `oneuniverse/data/database.py`
- Test: `test/data/test_database.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_discovery_and_views(tmp_path):
    write_synthetic_dataset(tmp_path / "eboss_qso", 1_000, 0, survey="eboss_qso"); write_synthetic_dataset(tmp_path / "desi/qso", 1_000, 1, survey="desi_qso"); (tmp_path / "_oneuid").mkdir()
    db = Database(tmp_path); assert db.names() == ["desi__qso", "eboss_qso"] and db["eboss_qso"].n_rows == 1_000 and "desi__qso" in db and len(db) == 2

def test_strict_vs_lenient_on_corrupt_manifest(tmp_path):
    write_synthetic_dataset(tmp_path / "ok", 10, 0); (tmp_path / "bad").mkdir(); (tmp_path / "bad/manifest.json").write_text("{")
    with pytest.raises(FormatError, match="bad"): Database(tmp_path)
    assert Database(tmp_path, strict=False).names() == ["ok"]

def test_history_and_as_of(tmp_path):
    write_synthetic_dataset(tmp_path / "ds", 10, 0); t0 = datetime.now(timezone.utc); time.sleep(0.01); write_synthetic_dataset(tmp_path / "ds", 20, 1, overwrite=True)
    db = Database(tmp_path); assert [m.validity.version for m in db.history("ds")] == [1, 2]
    assert "ds" in db.as_of(datetime.now(timezone.utc)) and "ds" not in db.as_of(datetime(2000, 1, 1, tzinfo=timezone.utc))

def test_summary_is_json_safe(tmp_path):
    write_synthetic_dataset(tmp_path / "ds", 10, 0); json.dumps(Database(tmp_path).summary())
```

- [ ] **Step 2: Run, expect failure**
- [ ] **Step 3: Implement** discovery (`os.walk` limited to depth 2, skip `_`-prefixed dirs), `strict` behaviour, `history` (live manifest + `_history/*.manifest.json` sorted by `created_utc`), `as_of` (dataset included when `validity.valid_from <= when` and the manifest version in `history` whose window contains `when` is the live one; otherwise excluded). Sidecar methods are added in Tasks 4–5.
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** — `feat(data): Database discovery, history and as_of`

---

### Task 4: `data.oneuid` — build, load, query, hydrate

**Files:**
- Create: `oneuniverse/data/oneuid.py`
- Modify: `oneuniverse/data/database.py` (delegators)
- Test: `test/data/test_oneuid.py`

- [ ] **Step 1: Write the failing tests**

```python
@pytest.fixture
def db2(tmp_path):  # two surveys: 30k rows each, 10k objects in common (same ra/dec/z ± jitter), multi-cell, shuffled file order
    ...; return Database(tmp_path)

def test_build_and_load(db2):
    idx = db2.build_oneuid(["a", "b"], rules=CrossMatchRules(sky_tol_arcsec=1.0), name="ab")
    assert idx.n_objects == 50_000 and idx.n_multi == 10_000 and idx.datasets == ["a", "b"] and idx.table.schema.names == list(ONEUID_SCHEMA.names)
    back = db2.load_oneuid("ab"); assert back.table.equals(idx.table) and back.rules == idx.rules and db2.list_oneuids() == ["ab"]

def test_oneuid_row_identity_regression(db2):
    """Hydrated rows must be the objects the index points at (the OUF 2.x bug)."""
    db2.build_oneuid(["a", "b"], rules=CrossMatchRules(), name="ab"); q = db2.oneuid_query("ab")
    sample = q.index.table.slice(0, 2_000)
    hyd = q.hydrate_table(sample["oneuid"].to_numpy(), columns=["object_id", "source_row", "mag"])
    for ds_name in ["a", "b"]:
        part = hyd.filter(pc.field("dataset") == ds_name); src = sample.filter(pc.field("dataset") == ds_name)
        joined = part.join(src, keys=["oneuid"], right_suffix="_idx")
        assert joined.num_rows == src.num_rows and np.array_equal(joined["source_row"].to_numpy(), joined["source_row_idx"].to_numpy())
        assert np.array_equal(joined["object_id"].to_numpy(), stable_object_id(ds_name, joined["source_row"].to_numpy()))

def test_select_by_geometry(db2):
    db2.build_oneuid(["a", "b"], rules=CrossMatchRules(), name="ab"); q = db2.oneuid_query("ab"); cone = Cone(185, 15, 5)
    ids = q.select(cone=cone); rows = q.rows(ids)
    assert len(ids) > 0 and np.all(cone.contains(rows["ra"].to_numpy(), rows["dec"].to_numpy()) | np.isin(rows["oneuid"].to_numpy(), ids))  # every selected object has >=1 member in the cone
    assert len(q.select(cone=cone, datasets=["a"])) <= len(ids)

def test_hydrate_streams_and_merges(db2):
    db2.build_oneuid(["a", "b"], rules=CrossMatchRules(), name="ab"); q = db2.oneuid_query("ab")
    ids = q.index.multi_only()["oneuid"].unique().to_numpy()[:5_000]
    batches = list(q.hydrate(ids, columns=["z"], batch_size=1_000)); assert len(batches) >= 10 and all(b.num_rows <= 1_000 for b in batches)
    t = pa.concat_tables(batches); assert t.num_rows == 10_000 and set(t.column_names) == {"oneuid", "dataset", "z"}

def test_restrict_to_keeps_ids(db2):
    idx = db2.build_oneuid(["a", "b"], rules=CrossMatchRules(), name="ab"); r = idx.restrict_to(["a"])
    assert r.datasets == ["a"] and np.isin(r.table["oneuid"].to_numpy(), idx.table["oneuid"].to_numpy()).all() and r.n_objects == 30_000

def test_rebuild_archives_and_as_of(db2):
    db2.build_oneuid(["a"], rules=CrossMatchRules(), name="x"); t0 = datetime.now(timezone.utc); time.sleep(0.01); db2.build_oneuid(["a", "b"], rules=CrossMatchRules(), name="x")
    assert db2.load_oneuid("x").datasets == ["a", "b"] and db2.load_oneuid("x", as_of=t0).datasets == ["a"]

@pytest.mark.slow
def test_crossmatch_two_million_rows_under_a_minute(tmp_path):  # 2 x 1M synthetic, 20% overlap
    ...; t = time.perf_counter(); db.build_oneuid(["a", "b"], rules=CrossMatchRules(), name="big"); assert time.perf_counter() - t < 60
```

- [ ] **Step 2: Run, expect failure**
- [ ] **Step 3: Implement.** `build_oneuid`: for each dataset, `view.iter_partitions(columns=["ra","dec","z","z_type","source_row","hpix32"])` → append `partition`, `offset = arange`; concat per dataset (these five columns for 10 M rows ≈ 400 MB, acceptable; note in docstring); `match_pair` for every unordered pair of datasets; offsets into the global stack; `components`; table per `ONEUID_SCHEMA`, sorted by `oneuid`; meta: `rules.to_dict()`, `rules_hash`, `datasets`, `dataset_versions` ({name: validity.version}), `n_objects`, `n_multi`, `block_nside`; write via `VersionedSidecar(db.root, "oneuid", name)`. `OneuidQuery.select`: masks on the index arrays (`Cone.contains` etc.), `np.unique(oneuid[mask])`. `hydrate`: `rows(oneuids)` → per dataset → chunks of `batch_size` → `view.take(partition, offset, columns)` + `oneuid`/`dataset` columns. `Database` gains the delegator methods.
- [ ] **Step 4: Run, expect PASS** (run the slow test once locally: `pytest -m slow test/data/test_oneuid.py`).
- [ ] **Step 5: Commit** — `feat(data): ONEUID index with position-based hydration`

---

### Task 5: `data.links`

**Files:**
- Create: `oneuniverse/data/links.py`
- Modify: `oneuniverse/data/database.py`
- Test: `test/data/test_links.py`

- [ ] **Step 1: Write the failing tests**

```python
@pytest.fixture
def db_hosts(tmp_path):  # "hosts" (kind galaxy, 2k rows) and "sne" (kind snia, 500 rows, each within 0.5" of a host, 20 of them within 0.5" of two hosts); ONEUID "default" built over both
    ...

def test_build_links_unambiguous(db_hosts):
    links = db_hosts.build_links(parents=["hosts"], children=["sne"], rules=SubobjectRules("galaxy", "snia", sky_tol_arcsec=1.0, dz_tol=None), name="host")
    assert links.table.num_rows == 480 and links.table.schema.names == list(LINKS_SCHEMA.names) and db_hosts.list_links() == ["host"]
    child = links.table["child_oneuid"][0].as_py(); assert isinstance(links.parent_of(child), int)

def test_build_links_ambiguous_policy(db_hosts):
    l2 = db_hosts.build_links(parents=["hosts"], children=["sne"], rules=SubobjectRules("galaxy", "snia", dz_tol=None, accept_ambiguous=True), name="amb")
    assert l2.table.num_rows == 520 and np.isclose(l2.table.filter(pc.field("confidence") < 1)["confidence"].to_numpy(), 0.5).all()
    amb_child = l2.table.filter(pc.field("confidence") < 1)["child_oneuid"][0].as_py(); assert len(l2.parent_of(amb_child)) == 2

def test_kind_guard(db_hosts):
    with pytest.raises(ValidationError, match="kind"): db_hosts.build_links(parents=["sne"], children=["hosts"], rules=SubobjectRules("galaxy", "snia"), name="bad")

def test_chain_walk(db_hosts):  # clusters -> hosts -> sne, two link sets
    ...; leaves = db_hosts.chain([cluster_oneuid], ["members", "host"]); assert len(leaves) > 0 and set(leaves) <= set(sne_oneuids)
```

- [ ] **Step 2: Run, expect failure**
- [ ] **Step 3: Implement** using `match_pair` on the ONEUID index rows restricted to parents / children (tables with `ra, dec, z, z_type, hpix32` + `oneuid`), `confidence = 1 / n_candidates` for ambiguous children when accepted, else drop; `chain` loads each link set and follows `parent_oneuid → child_oneuid` with `np.isin`. Meta carries `rules`, `oneuid_name`, `oneuid_version` (the sidecar version used).
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** — `feat(data): sub-object links on the shared sidecar store`

---

### Task 6: Surface, guards and close-out

- [ ] **Step 1: Export** from `oneuniverse.data`: `Database`, `CrossMatchRules`, `OneuidIndex`, `OneuidQuery`, `SubobjectRules`, `SubobjectLinks`, `VersionedSidecar`; keep `len(__all__) <= 32` and update `test_public_api`.
- [ ] **Step 2: Run** `pytest -n 6` and `pytest` single-process (< 60 s); guards green.
- [ ] **Step 3: STATUS.md + memory**, commit + push claude_config. Commit — `docs: R2 close-out`.

**Exit criterion:** `test_oneuid_row_identity_regression` green; cross-match of two 1 M-row synthetic surveys under 60 s (slow test, run once); sidecar crash test green.
