"""Phase 20 T4 — Database.chain_subobjects walks multi-level links."""
import numpy as np
import pandas as pd
import pytest


def test_chain_two_links_returns_leaves(tmp_path):
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


def test_chain_three_links_transitive():
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
    written pair of sidecars.
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
