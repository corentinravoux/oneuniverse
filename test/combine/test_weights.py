import healpy as hp
import numpy as np
import pyarrow as pa
import pytest

from oneuniverse._core.errors import SchemaError, ValidationError
from oneuniverse._core.healpix import radec_to_pix
from oneuniverse.combine.weights import (
    ColumnWeight,
    ConstantWeight,
    FKPWeight,
    HealpixMapWeight,
    InverseVarianceWeight,
    ProductWeight,
    QualityMaskWeight,
    ShearWeight,
)

T = pa.table({"ra": [0.0, 10.0, 20.0], "dec": [0.0, 0.0, 0.0], "z": [0.5, 1.0, 1.5],
              "z_err": [0.01, np.nan, 0.0], "q": [1, 0, 1], "nbar_col": [1e-4, 2e-4, 3e-4]})


def test_column_and_constant_and_product():
    w = ColumnWeight("q") * ConstantWeight(2.0)
    out = w(T)
    assert out.tolist() == [2.0, 0.0, 2.0] and w.columns == {"q"} and isinstance(w, ProductWeight)


def test_product_flattens_and_unions_columns():
    w = (ColumnWeight("q") * ColumnWeight("z")) * (ConstantWeight(2.0) * ColumnWeight("nbar_col"))
    assert len(w.factors) == 4 and w.columns == {"q", "z", "nbar_col"}
    assert w.describe() == "Column(q) × Column(z) × Constant(2) × Column(nbar_col)"


def test_ivar_missing_policies():
    assert InverseVarianceWeight("z_err")(T).tolist() == [1e4, 1.0, 1.0]
    assert InverseVarianceWeight("z_err", missing="zero")(T).tolist() == [1e4, 0.0, 0.0]
    with pytest.raises(ValidationError):
        InverseVarianceWeight("z_err", missing="raise")(T)
    with pytest.raises(ValidationError, match="negative"):
        InverseVarianceWeight("e")(pa.table({"e": [-1.0]}))


def test_fkp_callable_and_column():
    assert np.allclose(FKPWeight(nbar=lambda z: 1e-4 * np.ones_like(z), P0=1e4)(T), 1 / (1 + 1.0))
    assert np.allclose(FKPWeight(nbar="nbar_col", P0=1e4)(T), 1 / (1 + np.array([1, 2, 3]) * 1.0))
    with pytest.raises(ValidationError):
        FKPWeight(nbar="nbar_col", P0=0.0)


def test_healpix_map_weight_nest_and_ring_and_nan():
    m = np.arange(12 * 4 * 4, dtype=float)
    w = HealpixMapWeight(m, nside=4)(T)
    assert np.array_equal(w, m[radec_to_pix(4, T["ra"].to_numpy(), T["dec"].to_numpy())])
    m_ring = m.copy()
    assert np.array_equal(HealpixMapWeight(m_ring, nside=4, nest=False)(T),
                          m[hp.ang2pix(4, T["ra"].to_numpy(), T["dec"].to_numpy(), lonlat=True)])
    m[radec_to_pix(4, [0.0], [0.0])[0]] = np.nan
    with pytest.raises(ValidationError):
        HealpixMapWeight(m, nside=4)(T)
    assert HealpixMapWeight(m, nside=4, nan=0.0)(T)[0] == 0.0


def test_healpix_map_wrong_length():
    with pytest.raises(ValidationError):
        HealpixMapWeight(np.ones(10), nside=4)


def test_healpix_from_fits(tmp_path):
    m = np.arange(12 * 4 * 4, dtype=float)
    path = tmp_path / "m.fits"
    hp.write_map(str(path), m, nest=True, overwrite=True)
    w = HealpixMapWeight.from_fits(path)
    assert w.nside == 4 and w.nest and np.array_equal(w.map, m)
    wr = HealpixMapWeight.from_fits(path, nest=False)
    assert np.array_equal(wr.map, hp.reorder(m, n2r=True)) and not wr.nest
    assert np.array_equal(wr(T), w(T))


def test_shear_weight_metacal_and_lensfit():
    S = pa.table({"shear_weight": [1.0, 1.0], "R11": [0.8, 0.8], "R22": [0.8, 0.8],
                  "R_S": [0.0, 0.1], "m_bias": [-0.1, 0.0], "e1_err": [0.3, 0.3], "e2_err": [0.3, 0.3]})
    assert np.allclose(ShearWeight("metacal")(S), [1 / 0.8**2, 1 / 0.9**2])
    assert np.allclose(ShearWeight("lensfit", sigma_e_columns=("e1_err", "e2_err"))(S),
                       [1 / (0.9**2 + 0.09), 1 / (1.0 + 0.09)])
    S2 = pa.table({"shear_weight": [1.0], "R11": [0.5], "R22": [0.5], "R_S": [0.0],
                   "e1_err": [0.2], "e2_err": [0.4]})
    # sigma_e^2 = (0.2^2 + 0.4^2) / 2 = 0.10
    assert np.allclose(ShearWeight("metacal", sigma_e_columns=("e1_err", "e2_err"))(S2),
                       [1 / (0.25 + 0.10)])
    with pytest.raises(SchemaError, match="R_S"):
        ShearWeight("metacal")(S.drop_columns(["R_S"]))


def test_quality_mask_and_missing_column():
    assert QualityMaskWeight("q", ">=", 1)(T).tolist() == [1.0, 0.0, 1.0]
    with pytest.raises(SchemaError, match="nope"):
        ColumnWeight("nope")(T)


def test_accepts_dataframe():
    assert ColumnWeight("q")(T.to_pandas()).tolist() == [1.0, 0.0, 1.0]


def test_result_validation():
    with pytest.raises(ValidationError, match="negative"):
        ColumnWeight("x")(pa.table({"x": [-1.0, 1.0]}))
    with pytest.raises(ValidationError, match=r"Column\(x\).*1 non-finite"):
        ColumnWeight("x")(pa.table({"x": [np.inf, 1.0]}))
    assert ConstantWeight(0.0)(T).tolist() == [0.0] * 3


def test_healpix_map_weight_stores_a_copy():
    m = np.arange(12 * 4**2, dtype=float)
    w = HealpixMapWeight(m, 4)
    m[:] = -1.0
    assert (w.map >= 0).all()


def test_describe_strings():
    assert InverseVarianceWeight("z_err").describe() == "InverseVariance(z_err)"
    assert FKPWeight("nbar_col", 1e4).describe() == "FKP(P0=10000, nbar=nbar_col)"

    def nb(z):
        return z

    assert FKPWeight(nb, 2.0).describe() == "FKP(P0=2, nbar=nb)"
    assert HealpixMapWeight(np.ones(12), 1, nest=False).describe() == "HealpixMap(nside=1, nest=False)"
    assert ShearWeight("lensfit").describe() == "Shear(lensfit)"
    assert QualityMaskWeight("q", ">=", 1).describe() == "QualityMask(q >= 1)"
    assert ColumnWeight("q").describe() == "Column(q)"
    assert ConstantWeight(2.0).describe() == "Constant(2)"


def test_quality_mask_nan_never_passes():
    t = pa.table({"x": pa.array([1.0, float("nan"), 0.0, None])})
    assert QualityMaskWeight("x", "!=", 1.0)(t).tolist() == [0.0, 0.0, 1.0, 0.0]
    assert QualityMaskWeight("x", "<=", 5.0)(t).tolist() == [1.0, 0.0, 1.0, 0.0]


def test_non_numeric_column_is_schema_error():
    t = pa.table({"s": ["a", "b"]})
    with pytest.raises(SchemaError, match="'s'.*string"):
        ColumnWeight("s")(t)
