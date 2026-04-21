"""Unit tests for SubobjectLinks container, sidecar I/O, and pair builder."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oneuniverse.data.subobject import (
    SUBOBJECT_DIR,
    SubobjectLinks,
    _links_path,
    _links_manifest_path,
)
from oneuniverse.data.subobject_rules import SubobjectRules
from oneuniverse.data.validity import DatasetValidity


_VALIDITY = DatasetValidity(valid_from_utc="2026-04-20T00:00:00+00:00")


def _toy_links_df():
    return pd.DataFrame(
        {
            "parent_oneuid": np.array([0, 1, 2], dtype=np.int64),
            "child_oneuid": np.array([100, 101, 102], dtype=np.int64),
            "confidence": np.array([1.0, 0.7, 1.0], dtype=np.float32),
            "sky_sep_arcsec": np.array([0.3, 0.9, 0.1], dtype=np.float32),
            "dz": np.array([1e-4, 4e-3, np.nan], dtype=np.float32),
        }
    )


def test_links_path_layout(tmp_path):
    assert _links_path(tmp_path, "sne_in_hosts") == (
        tmp_path / SUBOBJECT_DIR / "sne_in_hosts.parquet"
    )
    assert _links_manifest_path(tmp_path, "sne_in_hosts") == (
        tmp_path / SUBOBJECT_DIR / "sne_in_hosts.manifest.json"
    )


def test_subobject_links_table_shape():
    rules = SubobjectRules(
        parent_survey_type="spectroscopic",
        child_survey_type="transient",
    )
    links = SubobjectLinks(
        name="sne_in_hosts",
        rules=rules,
        parent_datasets=("spec_desi",),
        child_datasets=("transient_pantheon",),
        oneuid_name="default",
        oneuid_hash="abcd" * 4,
        validity=_VALIDITY,
        table=_toy_links_df(),
    )
    assert len(links) == 3
    for c in (
        "parent_oneuid", "child_oneuid", "confidence",
        "sky_sep_arcsec", "dz",
    ):
        assert c in links.table.columns
