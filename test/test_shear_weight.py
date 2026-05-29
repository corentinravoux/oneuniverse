"""Phase 19 T2 — ShearWeight (metacal + lensfit) propagates shape noise
and calibration responses.
"""
import numpy as np
import pandas as pd
import pytest

from oneuniverse.combine.weights.shear import ShearWeight


def test_metacal_default_response_one_and_zero_sigma():
    df = pd.DataFrame({
        "shear_weight": np.array([1.0, 1.0], dtype="f4"),
        "R11": np.array([1.0, 1.0], dtype="f4"),
        "R22": np.array([1.0, 1.0], dtype="f4"),
    })
    w = ShearWeight(kind="metacal").compute(df)
    np.testing.assert_allclose(w, np.array([1.0, 1.0]), rtol=1e-6)


def test_metacal_with_response_below_one_amplifies():
    df = pd.DataFrame({
        "shear_weight": np.array([1.0], dtype="f4"),
        "R11": np.array([0.7], dtype="f4"),
        "R22": np.array([0.7], dtype="f4"),
    })
    w = ShearWeight(kind="metacal").compute(df)
    np.testing.assert_allclose(w, np.array([1.0 / 0.49]), rtol=1e-6)


def test_metacal_with_selection_response_added():
    df = pd.DataFrame({
        "shear_weight": np.array([1.0], dtype="f4"),
        "R11": np.array([0.6], dtype="f4"),
        "R22": np.array([0.6], dtype="f4"),
        "R_S": np.array([0.1], dtype="f4"),
    })
    w = ShearWeight(kind="metacal").compute(df)
    np.testing.assert_allclose(w, np.array([1.0 / 0.49]), rtol=1e-6)


def test_metacal_sigma_e_in_denominator():
    df = pd.DataFrame({
        "shear_weight": np.array([1.0], dtype="f4"),
        "R11": np.array([1.0], dtype="f4"),
        "R22": np.array([1.0], dtype="f4"),
        "e1_err": np.array([0.5], dtype="f4"),
        "e2_err": np.array([0.5], dtype="f4"),
    })
    w = ShearWeight(kind="metacal", sigma_e_cols=("e1_err", "e2_err")).compute(df)
    np.testing.assert_allclose(w, np.array([1.0 / 1.5]), rtol=1e-6)


def test_lensfit_uses_one_plus_m():
    df = pd.DataFrame({
        "shear_weight": np.array([1.0], dtype="f4"),
        "m_bias": np.array([0.05], dtype="f4"),
    })
    w = ShearWeight(kind="lensfit").compute(df)
    np.testing.assert_allclose(w, np.array([1.0 / (1.05 ** 2)]), rtol=1e-6)


def test_invalid_kind_rejected():
    with pytest.raises(ValueError, match="kind"):
        ShearWeight(kind="unknown")


def test_missing_response_columns_raise():
    df = pd.DataFrame({"shear_weight": np.array([1.0], dtype="f4")})
    with pytest.raises(KeyError):
        ShearWeight(kind="metacal").compute(df)
    with pytest.raises(KeyError):
        ShearWeight(kind="lensfit").compute(df)
