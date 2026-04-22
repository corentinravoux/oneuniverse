"""Tests for SubobjectLinks.children_of / .parent_of."""
import numpy as np
import pandas as pd
import pytest

from oneuniverse.data.subobject import SubobjectLinks
from oneuniverse.data.subobject_rules import SubobjectRules
from oneuniverse.data.validity import DatasetValidity


_VALIDITY = DatasetValidity(valid_from_utc="2026-04-20T00:00:00+00:00")


def _links():
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
    assert _links().parent_of(200) == 10


def test_parent_of_ambiguous_returns_list():
    out = _links().parent_of(203)
    assert isinstance(out, list)
    assert set(out) == {10, 12}


def test_parent_of_missing_child_returns_none():
    assert _links().parent_of(9999) is None
