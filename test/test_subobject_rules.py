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
    r2 = pickle.loads(pickle.dumps(r))
    assert r.hash() == r2.hash()
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
    assert r.hash() == r.hash()
