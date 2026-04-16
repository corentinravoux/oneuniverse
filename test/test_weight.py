"""Tests for oneuniverse.weight — base weights, cross-match, combine, catalog."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from oneuniverse.weight import (
    ColumnWeight,
    ConstantWeight,
    FKPWeight,
    InverseVarianceWeight,
    ProductWeight,
    QualityMaskWeight,
    WeightedCatalog,
    combine_weights,
    cross_match_surveys,
)


# ── Base weights ─────────────────────────────────────────────────────────


class TestBaseWeights:
    def setup_method(self):
        self.df = pd.DataFrame({
            "z": [0.1, 0.2, 0.3, 0.4],
            "z_err": [0.01, 0.02, 0.0, 0.005],
            "zwarn": [0, 0, 4, 0],
            "w_sys": [1.0, 0.9, 1.1, 1.0],
        })

    def test_constant_weight(self):
        w = ConstantWeight(2.5)(self.df)
        assert np.allclose(w, 2.5)

    def test_constant_negative_raises(self):
        with pytest.raises(ValueError):
            ConstantWeight(-1.0)

    def test_column_weight(self):
        w = ColumnWeight("w_sys")(self.df)
        assert np.allclose(w, [1.0, 0.9, 1.1, 1.0])

    def test_column_weight_missing(self):
        with pytest.raises(KeyError):
            ColumnWeight("nope")(self.df)

    def test_inverse_variance_basic(self):
        w = InverseVarianceWeight("z_err")(self.df)
        # zero error → weight 0 (no inf)
        assert np.isfinite(w).all()
        assert w[2] == 0.0
        assert np.isclose(w[0], 1.0 / 0.01**2)

    def test_inverse_variance_with_floor(self):
        w = InverseVarianceWeight("z_err", floor=0.01)(self.df)
        # variance for row 0: 0.01² + 0.01² = 2e-4
        assert np.isclose(w[0], 1.0 / (2 * 0.01**2))

    def test_inverse_variance_floor_negative(self):
        with pytest.raises(ValueError):
            InverseVarianceWeight("z_err", floor=-1.0)

    def test_quality_mask(self):
        w = QualityMaskWeight("zwarn", "==", 0)(self.df)
        assert np.array_equal(w, [1, 1, 0, 1])

    def test_quality_mask_bad_op(self):
        with pytest.raises(ValueError):
            QualityMaskWeight("zwarn", "===", 0)

    def test_fkp_constant_nbar(self):
        nbar = lambda z: np.full_like(z, 1e-4)
        w = FKPWeight(nbar=nbar, P0=1e4)(self.df)
        assert np.allclose(w, 1.0 / (1.0 + 1e-4 * 1e4))

    def test_product_composition(self):
        w1 = ConstantWeight(2.0)
        w2 = ColumnWeight("w_sys")
        prod = w1 * w2
        assert isinstance(prod, ProductWeight)
        assert np.allclose(prod(self.df), 2.0 * np.array([1.0, 0.9, 1.1, 1.0]))

    def test_shape_validation(self):
        class BadWeight(InverseVarianceWeight):
            def compute(self, df):
                return np.array([1.0, 2.0])  # wrong shape
        with pytest.raises(ValueError, match="shape"):
            BadWeight("z_err")(self.df)


# ── Cross-match ──────────────────────────────────────────────────────────


def _toy_two_surveys():
    # 3 objects in survey A; survey B has 2 of them (slightly perturbed)
    # plus 1 unique object.
    a = pd.DataFrame({
        "ra":  [10.000, 20.000, 30.000],
        "dec": [-5.000, 0.0000, 15.000],
        "z":   [0.10, 0.20, 0.30],
        "v":   [100.0, 200.0, 300.0],
        "v_err": [10.0, 20.0, 30.0],
    })
    b = pd.DataFrame({
        "ra":  [10.0001, 30.0001, 50.0],   # close to A[0], A[2], unique
        "dec": [-5.0001, 15.0000, 5.0],
        "z":   [0.1001, 0.3001, 0.4],
        "v":   [110.0, 290.0, 999.0],
        "v_err": [5.0, 50.0, 5.0],
    })
    return {"A": a, "B": b}


class TestCrossMatch:
    def test_basic_match_counts(self):
        cats = _toy_two_surveys()
        res = cross_match_surveys(cats, sky_tol_arcsec=5.0, dz_tol=1e-2)
        # 3 + 3 = 6 input rows, 4 unique groups (2 shared + A[1] alone + B[2] alone)
        assert len(res.table) == 6
        assert res.n_groups == 4
        assert res.n_multi == 2

    def test_multi_survey_subset(self):
        res = cross_match_surveys(_toy_two_surveys(), sky_tol_arcsec=5.0)
        ms = res.multi_survey()
        assert set(ms["survey"]) == {"A", "B"}
        assert len(ms) == 4  # 2 universal × 2 surveys

    def test_dz_cut_blocks_match(self):
        # Tight Δz makes the second pair fail to merge.
        res = cross_match_surveys(
            _toy_two_surveys(), sky_tol_arcsec=5.0, dz_tol=1e-5,
        )
        # Both Δz are 1e-4 → no merges; 6 rows, 6 groups
        assert res.n_groups == 6
        assert res.n_multi == 0

    def test_missing_columns_raises(self):
        with pytest.raises(KeyError):
            cross_match_surveys({"X": pd.DataFrame({"a": [1]})})

    def test_group_query(self):
        res = cross_match_surveys(_toy_two_surveys(), sky_tol_arcsec=5.0)
        ms_ids = res.multi_survey()["universal_id"].unique()
        for uid in ms_ids:
            grp = res.group(int(uid))
            assert grp["survey"].nunique() == 2


# ── combine_weights strategies ──────────────────────────────────────────


def _matched_long():
    # Two universal objects, each seen by both surveys.
    return pd.DataFrame({
        "universal_id": [0, 0, 1, 1],
        "survey":       ["A", "B", "A", "B"],
        "value":        [100.0, 110.0, 200.0, 210.0],
        "variance":     [100.0,  25.0, 400.0, 100.0],  # σ² → σ_A=10, σ_B=5, …
    })


class TestCombineWeights:
    def test_best_only_picks_min_variance(self):
        out = combine_weights(
            _matched_long(),
            value_col="value", variance_col="variance",
            strategy="best_only",
        )
        t = out.table.set_index("universal_id")
        assert np.isclose(t.loc[0, "value"], 110.0)   # B wins (σ²=25)
        assert np.isclose(t.loc[0, "variance"], 25.0)
        assert np.isclose(t.loc[1, "value"], 210.0)   # B wins (σ²=100)
        assert np.isclose(t.loc[1, "variance"], 100.0)
        assert (t["n_surveys"] == 2).all()

    def test_ivar_average_blue(self):
        out = combine_weights(
            _matched_long(),
            value_col="value", variance_col="variance",
            strategy="ivar_average",
        )
        t = out.table.set_index("universal_id")
        # BLUE: (100/100 + 110/25) / (1/100 + 1/25) = (1 + 4.4) / 0.05 = 108
        assert np.isclose(t.loc[0, "value"], 108.0)
        # 1/(1/100 + 1/25) = 20
        assert np.isclose(t.loc[0, "variance"], 20.0)

    def test_hyperparameter_priority(self):
        # Equal variances → result should land on the higher-α survey.
        df = pd.DataFrame({
            "universal_id": [0, 0],
            "survey": ["A", "B"],
            "value": [100.0, 200.0],
            "variance": [10.0, 10.0],
        })
        out = combine_weights(
            df, value_col="value", variance_col="variance",
            strategy="hyperparameter",
            survey_alpha={"A": 1.0, "B": 3.0},
        )
        # Weighted mean: (1*100 + 3*200) / 4 = 175
        v = out.table["value"].iloc[0]
        assert np.isclose(v, 175.0)

    def test_unit_mean_warns(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            combine_weights(
                _matched_long(),
                value_col="value", variance_col="variance",
                strategy="unit_mean",
            )
        assert any("unit_mean" in r.message for r in caplog.records)

    def test_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            combine_weights(
                _matched_long(),
                value_col="value", variance_col="variance",
                strategy="bogus",
            )

    def test_drops_bad_variance(self):
        df = _matched_long().copy()
        df.loc[0, "variance"] = 0.0  # invalid
        out = combine_weights(
            df, value_col="value", variance_col="variance",
            strategy="best_only",
        )
        # Object 0 still has the B measurement
        t = out.table.set_index("universal_id")
        assert np.isclose(t.loc[0, "value"], 110.0)


# ── WeightedCatalog facade ───────────────────────────────────────────────


class TestWeightedCatalog:
    def setup_method(self):
        self.cats = _toy_two_surveys()
        self.wc = WeightedCatalog(self.cats)
        self.wc.add_weight("A", InverseVarianceWeight("v_err"))
        self.wc.add_weight("B", InverseVarianceWeight("v_err"))

    def test_total_weight_shape(self):
        wA = self.wc.total_weight("A")
        assert wA.shape == (3,)
        assert np.isclose(wA[0], 1.0 / 100.0)

    def test_crossmatch_populates_long_table(self):
        res = self.wc.crossmatch(sky_tol_arcsec=5.0, dz_tol=1e-2)
        assert res.n_multi == 2
        assert self.wc._weighted_long is not None
        assert "weight" in self.wc._weighted_long.columns

    def test_concurrences_returns_all_surveys(self):
        self.wc.crossmatch(sky_tol_arcsec=5.0, dz_tol=1e-2)
        ms = self.wc._match.multi_survey()
        uid = int(ms["universal_id"].iloc[0])
        rows = self.wc.concurrences(uid)
        assert set(rows["survey"]) == {"A", "B"}
        # weight column carries the registered IVW values
        assert (rows["weight"] > 0).all()

    def test_combine_through_facade(self):
        self.wc.crossmatch(sky_tol_arcsec=5.0, dz_tol=1e-2)
        # Add a fake variance column on the long table by joining v_err².
        self.wc._weighted_long["v_var"] = self.wc._weighted_long["v_err"] ** 2
        combined = self.wc.combine(
            value_col="v", variance_col="v_var", strategy="best_only",
        )
        assert len(combined) == self.wc.n_universal()

    def test_unknown_survey_raises(self):
        with pytest.raises(KeyError):
            self.wc.add_weight("nope", ConstantWeight(1.0))

    def test_crossmatch_emits_deprecation(self):
        with pytest.warns(DeprecationWarning, match="from_oneuid"):
            self.wc.crossmatch(sky_tol_arcsec=5.0, dz_tol=1e-2)


# ── WeightedCatalog.from_oneuid (Phase 4) ───────────────────────────────


@pytest.fixture
def _db_with_oneuid(tmp_path):
    """Tiny DB with two datasets sharing 2 objects — builds ONEUID."""
    import shutil
    import uuid

    import healpy as hp

    from oneuniverse.data import OneuniverseDatabase, convert_survey
    from oneuniverse.data._base_loader import BaseSurveyLoader, SurveyConfig
    from oneuniverse.data._registry import _REGISTRY

    tag = uuid.uuid4().hex[:8]
    names = [f"wcat_a_{tag}", f"wcat_b_{tag}"]
    for name in names:
        class _Loader(BaseSurveyLoader):
            config = SurveyConfig(
                name=name, survey_type="spectroscopic",
                description=name,
                data_subpath=f"spectroscopic/{name}",
                data_filename="synth.csv", data_format="csv",
            )

            def _load_raw(self, data_path=None, **kwargs):
                from pathlib import Path
                return pd.read_csv(Path(data_path) / self.config.data_filename)
        _REGISTRY[name] = _Loader

    def _make(dir_, ras, decs, zs, vs, verrs):
        dir_.mkdir(parents=True, exist_ok=True)
        n = len(ras)
        ras, decs = np.asarray(ras), np.asarray(decs)
        df = pd.DataFrame({
            "ra": ras, "dec": decs, "z": zs,
            "z_type": ["spec"] * n,
            "z_err": np.full(n, 1e-4, dtype=np.float32),
            "galaxy_id": np.arange(n, dtype=np.int64),
            "survey_id": "synth",
            "_original_row_index": np.arange(n, dtype=np.int64),
            "_healpix32": hp.ang2pix(
                32, np.radians(90.0 - decs), np.radians(ras), nest=True,
            ).astype(np.int32),
            "v": vs,
            "v_err": verrs,
        })
        csv = dir_ / "synth.csv"
        df.to_csv(csv, index=False)
        return csv

    raw_a = tmp_path / "raw_a"
    raw_b = tmp_path / "raw_b"
    _make(raw_a, [10.0, 20.0, 30.0], [-5.0, 0.0, 15.0],
          [0.10, 0.20, 0.30], [100.0, 200.0, 300.0], [10.0, 20.0, 30.0])
    _make(raw_b, [10.0 + 1e-5, 30.0 + 1e-5, 50.0],
          [-5.0 + 1e-5, 15.0 + 1e-5, 5.0],
          [0.1 + 1e-6, 0.3 + 1e-6, 0.4],
          [110.0, 290.0, 999.0], [5.0, 50.0, 5.0])

    db_root = tmp_path / "db"
    for name, raw in zip(names, (raw_a, raw_b)):
        convert_survey(survey_name=name, raw_path=raw,
                       output_dir=db_root / f"spectroscopic/{name}",
                       overwrite=True)
    db = OneuniverseDatabase(db_root)
    ds_a = f"spectroscopic_{names[0]}"
    ds_b = f"spectroscopic_{names[1]}"
    from oneuniverse.data.oneuid_rules import CrossMatchRules
    index = db.build_oneuid(
        rules=CrossMatchRules(sky_tol_arcsec=2.0, dz_tol_default=1e-3),
    )
    yield db, index, ds_a, ds_b
    for name in names:
        _REGISTRY.pop(name, None)
    shutil.rmtree(tmp_path, ignore_errors=True)


class TestWeightedCatalogFromOneuid:
    def test_from_oneuid_loads_catalogs(self, _db_with_oneuid):
        db, index, ds_a, ds_b = _db_with_oneuid
        wc = WeightedCatalog.from_oneuid(index, db)
        assert set(wc.catalogs) == {ds_a, ds_b}
        assert wc._match is not None
        assert wc._match.n_groups == index.n_unique
        assert wc._match.n_multi == index.n_multi

    def test_from_oneuid_no_deprecation(self, _db_with_oneuid):
        db, index, _ds_a, _ds_b = _db_with_oneuid
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            WeightedCatalog.from_oneuid(index, db)

    def test_combine_via_from_oneuid(self, _db_with_oneuid):
        db, index, ds_a, ds_b = _db_with_oneuid
        wc = WeightedCatalog.from_oneuid(index, db)
        wc.add_weight(ds_a, InverseVarianceWeight("v_err"))
        wc.add_weight(ds_b, InverseVarianceWeight("v_err"))
        wc._weighted_long = None  # force rebuild via _ensure_long_table
        wc._ensure_long_table()
        long = wc._weighted_long
        assert "weight" in long.columns
        assert (long["weight"] > 0).all()
        # Combine: best_only picks the lowest-variance survey for shared uids.
        long["v_var"] = long["v_err"] ** 2
        wc._weighted_long = long
        out = wc.combine(value_col="v", variance_col="v_var",
                         strategy="best_only")
        assert len(out) == index.n_unique


