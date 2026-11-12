import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from fixtures.two_surveys import make_two_survey_db

from oneuniverse._core.errors import NotFoundError, SchemaError, ValidationError
from oneuniverse.combine import ConstantWeight, InverseVarianceWeight, WeightedCatalog
from oneuniverse.data.crossmatch import CrossMatchRules


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    db = make_two_survey_db(tmp_path_factory.mktemp("wc"))
    idx = db.build_oneuid(["a", "b"], rules=CrossMatchRules(), name="ab")
    return db, idx


@pytest.fixture
def wc(built):
    db, idx = built
    return WeightedCatalog.from_oneuid(idx, db, columns=["z", "z_err", "mag"]), db


def test_long_table_shape(wc):
    cat, _ = wc
    assert cat.datasets == ["a", "b"] and cat.table.num_rows == 60_000
    assert set(cat.table.column_names) >= {"oneuid", "dataset", "z", "z_err", "mag", "z_type"}


def test_default_columns_are_shared(built):
    db, idx = built
    cat = WeightedCatalog.from_oneuid(idx, db)
    assert {"z", "z_err", "mag", "z_type"} <= set(cat.table.column_names)


def test_weights_require_every_dataset(wc):
    cat, db = wc
    with pytest.raises(ValidationError, match="b"):
        cat.add_weight("a", ConstantWeight()).weights()
    cat.fill_defaults(db)
    w = cat.weights()
    assert w.num_rows == 60_000 and (w["weight"].to_numpy() > 0).all()
    assert w.column_names == ["oneuid", "dataset", "weight"]


def test_add_weight_unknown_dataset_and_priority(wc):
    cat, db = wc
    with pytest.raises(NotFoundError):
        cat.add_weight("zz", ConstantWeight())
    cat.add_weight("a", ConstantWeight(5.0)).fill_defaults(db)
    w = cat.weights()
    a = w.filter(pc.field("dataset") == "a")["weight"].to_numpy()
    b = w.filter(pc.field("dataset") == "b")["weight"].to_numpy()
    assert (a == 5.0).all() and not (b == 5.0).any()
    cat.add_weight("b", ConstantWeight(2.0))
    assert (cat.weights().filter(pc.field("dataset") == "b")["weight"].to_numpy() == 2.0).all()


def test_fill_defaults_unknown_pair(built, monkeypatch):
    from oneuniverse.combine import defaults as dmod
    db, idx = built
    monkeypatch.setattr(dmod, "_DEFAULTS", {})
    cat = WeightedCatalog.from_oneuid(idx, db, columns=["z", "z_err"])
    with pytest.raises(NotFoundError, match="spectroscopic"):
        cat.fill_defaults(db)
    cat.fill_defaults(db, skip_unknown=True)
    assert (cat.weights()["weight"].to_numpy() == 1.0).all()


def test_combine_ivar_matches_formula(wc):
    cat, _ = wc
    cat.add_weight("a", InverseVarianceWeight("z_err")).add_weight("b", InverseVarianceWeight("z_err"))
    out = cat.combine("z", variance="z_err")
    multi = out.filter(pc.field("n_sources") == 2)
    assert out.num_rows == 50_000 and multi.num_rows == 10_000
    assert out.column_names == ["oneuid", "value", "variance", "n_sources", "best_dataset"]
    uid = out["oneuid"].to_numpy()
    assert (np.diff(uid) > 0).all()
    row = multi.slice(0, 1).to_pylist()[0]
    members = cat.table.filter(pc.field("oneuid") == row["oneuid"]).to_pandas()
    ivar = 1 / members["z_err"].to_numpy().astype(float) ** 2
    assert row["value"] == pytest.approx((members["z"] * ivar).sum() / ivar.sum())
    assert row["variance"] == pytest.approx(1 / ivar.sum())


def test_combine_best_and_mean(wc):
    cat, db = wc
    cat.fill_defaults(db)
    best = cat.combine("z", variance="z_err", strategy="best")
    assert set(best["best_dataset"].to_pylist()) <= {"a", "b"}
    mean = cat.combine("mag", strategy="mean")
    assert mean.num_rows == 50_000 and "variance" in mean.column_names


def test_combine_is_vectorised_fast(wc):
    cat, db = wc
    cat.fill_defaults(db)
    t = time.perf_counter()
    cat.combine("z", variance="z_err")
    assert time.perf_counter() - t < 1.0


def _toy(rows, weights=None):
    """Catalog from (oneuid, dataset, z, s) rows; weights = {dataset: const}."""
    oid, ds, z, s = zip(*rows)
    t = pa.table({"oneuid": pa.array(oid, pa.int64()), "dataset": pa.array(ds, pa.string()),
                  "z": pa.array(z, pa.float64()), "s": pa.array(s, pa.float64()),
                  "z_type": pa.array(["spec"] * len(rows))})
    cat = WeightedCatalog(t, sorted(set(ds)))
    for k, v in (weights or {}).items():
        cat.add_weight(k, ConstantWeight(v))
    return cat


def test_ivar_valid_sigma_wins_and_missing_sigma_falls_back():
    nan = float("nan")
    cat = _toy([(1, "a", 1.0, 0.1), (1, "b", 9.0, nan), (1, "c", 7.0, 0.0),
                (2, "a", 3.0, nan), (2, "b", 4.0, -1.0),
                (3, "a", 2.0, 0.2), (3, "b", 4.0, 0.2),
                (4, "a", nan, 0.1), (4, "b", nan, nan)])
    out = cat.combine("z", variance="s").to_pydict()
    assert out["oneuid"] == [1, 2, 3, 4] and out["n_sources"] == [1, 2, 2, 0]
    assert out["value"][0] == pytest.approx(1.0) and out["variance"][0] == pytest.approx(0.01)
    assert out["value"][1] == pytest.approx(3.5) and np.isnan(out["variance"][1])
    assert out["best_dataset"] == ["a", "a", "a", None]
    assert out["value"][2] == pytest.approx(3.0) and out["variance"][2] == pytest.approx(0.02)
    assert np.isnan(out["value"][3]) and np.isnan(out["variance"][3])


def test_all_nan_sigma_dataset_every_strategy():
    """The eBOSS case: dataset 'e' has no sigma at all."""
    nan = float("nan")
    cat = _toy([(1, "d", 1.0, 0.1), (1, "e", 3.0, nan), (2, "e", 5.0, nan), (3, "d", 2.0, 0.2)],
               {"d": 1.0, "e": 1.0})
    for strat in ("ivar", "best", "mean", "priority"):
        out = cat.combine("z", variance="s", strategy=strat).to_pydict()
        assert out["oneuid"] == [1, 2, 3] and not np.isnan(out["value"]).any(), strat
        assert out["value"][1] == 5.0 and out["best_dataset"][1] == "e", strat
        assert np.isnan(out["variance"][1]), strat
        assert out["n_sources"][1] == 1, strat
    mean = cat.combine("z", variance="s", strategy="mean").to_pydict()
    assert mean["value"][0] == pytest.approx(2.0) and np.isnan(mean["variance"][0])
    assert mean["n_sources"][0] == 2
    ivar = cat.combine("z", variance="s").to_pydict()
    assert ivar["value"][0] == 1.0 and ivar["n_sources"][0] == 1
    best = cat.combine("z", variance="s", strategy="best").to_pydict()
    assert best["value"][0] == 1.0 and best["variance"][0] == pytest.approx(0.01)
    pr = cat.combine("z", variance="s", strategy="priority").to_pydict()
    assert pr["value"][0] == 1.0 and pr["variance"][0] == pytest.approx(0.01)


def test_best_falls_back_to_first_finite_in_dataset_order():
    nan = float("nan")
    cat = _toy([(1, "a", nan, nan), (1, "b", 2.0, nan), (1, "c", 3.0, nan)])
    out = cat.combine("z", variance="s", strategy="best").to_pydict()
    assert out["value"] == [2.0] and out["best_dataset"] == ["b"] and np.isnan(out["variance"][0])


def test_priority_order_and_validation():
    nan = float("nan")
    cat = _toy([(1, "a", 1.0, 0.1), (1, "b", 2.0, 0.2), (2, "a", nan, 0.1), (2, "b", 4.0, nan),
                (3, "c", 6.0, 0.5), (3, "a", 5.0, 0.3)])
    d = cat.combine("z", variance="s", strategy="priority").to_pydict()  # catalog order a, b, c
    assert d["value"] == [1.0, 4.0, 5.0] and d["best_dataset"] == ["a", "b", "a"]
    assert d["variance"][0] == pytest.approx(0.01) and np.isnan(d["variance"][1])
    assert d["n_sources"] == [1, 1, 1]
    r = cat.combine("z", variance="s", strategy="priority", priority=["c", "b"]).to_pydict()
    assert r["value"] == [2.0, 4.0, 6.0] and r["best_dataset"] == ["b", "b", "c"]
    r = cat.combine("z", strategy="priority", priority=["b"]).to_pydict()  # a, c follow b
    assert r["value"] == [2.0, 4.0, 5.0] and np.isnan(r["variance"]).all()
    with pytest.raises(NotFoundError, match="zz"):
        cat.combine("z", strategy="priority", priority=["zz"])
    with pytest.raises(ValidationError):
        cat.combine("z", strategy="priority", priority="a")
    with pytest.raises(ValidationError):
        cat.combine("z", variance="s", priority=["a"])


def test_combine_refuses_angles():
    cat = _toy([(1, "a", 1.0, 0.1)])
    with pytest.raises(ValidationError, match="ONEUID"):
        cat.combine("ra", variance="s")


def test_constructor_validates_datasets():
    t = pa.table({"oneuid": pa.array([1, 1], pa.int64()), "dataset": ["a", "b"]})
    with pytest.raises(ValidationError):
        WeightedCatalog(t, ["a", "a", "b"])
    with pytest.raises(ValidationError):
        WeightedCatalog(t, ["a"])
    with pytest.raises(ValidationError):
        WeightedCatalog(t, "ab")
    assert WeightedCatalog(t, ["b", "a"]).datasets == ["b", "a"]


def test_fill_defaults_uses_stored_db_and_requires_one(built):
    db, idx = built
    cat = WeightedCatalog.from_oneuid(idx, db, columns=["z", "z_err"]).fill_defaults()
    assert cat.weights().num_rows == 60_000
    t = pa.table({"oneuid": pa.array([1], pa.int64()), "dataset": ["a"], "z_type": ["spec"]})
    with pytest.raises(ValidationError, match="Database"):
        WeightedCatalog(t, ["a"]).fill_defaults()


def _naive(rows, w, strategy):
    """Reference loop: per object (value, variance, n, best) under the documented rules."""
    out = {}
    for oid in sorted({r[0] for r in rows}):
        mem = [r for r in rows if r[0] == oid]
        fin = sorted((r for r in mem if np.isfinite(r[2])), key=lambda r: r[1])  # catalog order
        ok = [r for r in fin if np.isfinite(r[3]) and r[3] > 0]
        if strategy == "ivar":
            if ok:
                i = [1 / r[3] ** 2 for r in ok]
                out[oid] = (sum(a * r[2] for a, r in zip(i, ok)) / sum(i), 1 / sum(i), len(ok),
                            min(ok, key=lambda r: r[3])[1])
            elif fin:
                out[oid] = (np.mean([r[2] for r in fin]), np.nan, len(fin), fin[0][1])
        elif strategy == "best":
            if ok:
                b = min(ok, key=lambda r: r[3])
                out[oid] = (b[2], b[3] ** 2, 1, b[1])
            elif fin:
                out[oid] = (fin[0][2], np.nan, 1, fin[0][1])
        else:
            c = [r for r in fin if w[r[1]] > 0]
            if c:
                ws = sum(w[r[1]] for r in c)
                val = sum(w[r[1]] * r[2] for r in c) / ws
                var = (sum(w[r[1]] ** 2 * r[3] ** 2 for r in c) / ws**2
                       if all(np.isfinite(r[3]) and r[3] > 0 for r in c) else np.nan)
                out[oid] = (val, var, len(c), max(c, key=lambda r: w[r[1]])[1])
    return out


@pytest.mark.parametrize("strategy", ["ivar", "best", "mean"])
def test_fuzz_against_naive_loop(strategy):
    rng = np.random.default_rng(3)
    w = {"a": 1.0, "b": 2.5, "c": 0.0}
    rows = []
    for oid in range(200):
        for ds in rng.permutation(["a", "b", "c"])[: rng.integers(1, 4)]:
            z = float("nan") if rng.random() < 0.15 else rng.normal()
            s = rng.choice([float("nan"), 0.0, -1.0, rng.uniform(0.1, 1)], p=[.25, .05, .05, .65])
            rows.append((oid, str(ds), z, float(s)))
    rows.sort(key=lambda r: r[0])
    cat = _toy(rows, w)
    out = cat.combine("z", variance="s", strategy=strategy).to_pydict()
    ref = _naive(rows, w, strategy)
    assert out["oneuid"] == sorted({r[0] for r in rows})
    for i, oid in enumerate(out["oneuid"]):
        if oid not in ref:
            assert out["n_sources"][i] == 0 and out["best_dataset"][i] is None
            continue
        val, var, n, best = ref[oid]
        assert out["value"][i] == pytest.approx(val) and out["n_sources"][i] == n
        assert out["variance"][i] == pytest.approx(var, nan_ok=True)
        assert out["best_dataset"][i] == best


def test_best_picks_smallest_sigma():
    cat = _toy([(1, "a", 1.0, 0.3), (1, "b", 2.0, 0.1), (2, "a", 5.0, 0.2), (2, "b", 6.0, 0.4)])
    out = cat.combine("z", variance="s", strategy="best").to_pydict()
    assert out["value"] == [2.0, 5.0] and out["best_dataset"] == ["b", "a"]
    assert out["variance"] == pytest.approx([0.01, 0.04])


def test_mean_with_uncertainty_and_best_dataset():
    cat = _toy([(1, "a", 1.0, 0.1), (1, "b", 4.0, 0.2)], {"a": 1.0, "b": 3.0})
    out = cat.combine("z", variance="s", strategy="mean").to_pydict()
    assert out["value"][0] == pytest.approx((1 + 12) / 4)
    assert out["variance"][0] == pytest.approx((0.01 + 9 * 0.04) / 16)
    assert out["best_dataset"] == ["b"] and out["n_sources"] == [2]


def test_mean_without_uncertainty_weighted_sample_variance():
    cat = _toy([(1, "a", 1.0, 0.0), (1, "b", 4.0, 0.0), (2, "a", 7.0, 0.0)], {"a": 1.0, "b": 3.0})
    out = cat.combine("z", strategy="mean").to_pydict()
    mu = 13 / 4
    assert out["variance"][0] == pytest.approx((1 * (1 - mu) ** 2 + 3 * (4 - mu) ** 2) / 4)
    assert out["variance"][1] == 0.0 and out["value"][1] == 7.0


def test_combine_errors():
    cat = _toy([(1, "a", 1.0, 0.1), (1, "b", 2.0, 0.1)])
    with pytest.raises(ValidationError):
        cat.combine("z")
    with pytest.raises(ValidationError, match="strategy"):
        cat.combine("z", variance="s", strategy="nope")
    with pytest.raises(SchemaError, match="nope"):
        cat.combine("nope", variance="s")
    with pytest.raises(SchemaError, match="nope"):
        cat.combine("z", variance="nope")
    with pytest.raises(ValidationError):
        cat.combine("z", strategy="mean")  # no weights
    with pytest.raises(SchemaError, match="dataset"):
        cat.combine("dataset", variance="s")
