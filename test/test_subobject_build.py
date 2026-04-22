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


def test_manifest_roundtrip(tmp_path):
    rules = SubobjectRules(
        parent_survey_type="spectroscopic",
        child_survey_type="transient",
        sky_tol_arcsec=2.5,
        dz_tol=1e-2,
        relation="hosts",
        accept_ambiguous=False,
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
    from oneuniverse.data.subobject import (
        write_subobject_links,
        read_subobject_links,
    )

    write_subobject_links(tmp_path, links)
    assert _links_path(tmp_path, "sne_in_hosts").exists()
    assert _links_manifest_path(tmp_path, "sne_in_hosts").exists()

    loaded = read_subobject_links(tmp_path, "sne_in_hosts")
    assert loaded.name == "sne_in_hosts"
    assert loaded.rules == rules
    assert loaded.parent_datasets == ("spec_desi",)
    assert loaded.child_datasets == ("transient_pantheon",)
    assert loaded.oneuid_hash == "abcd" * 4
    assert loaded.validity == _VALIDITY
    pd.testing.assert_frame_equal(
        loaded.table.reset_index(drop=True),
        links.table.reset_index(drop=True),
        check_dtype=True,
    )


def test_write_is_atomic(tmp_path):
    from oneuniverse.data.subobject import write_subobject_links

    rules = SubobjectRules(
        parent_survey_type="spectroscopic",
        child_survey_type="transient",
    )
    links = SubobjectLinks(
        name="partial",
        rules=rules,
        parent_datasets=("a",),
        child_datasets=("b",),
        oneuid_name="default",
        oneuid_hash="x" * 16,
        validity=_VALIDITY,
        table=_toy_links_df(),
    )
    write_subobject_links(tmp_path, links)
    leftovers = list((tmp_path / SUBOBJECT_DIR).glob("*.tmp*"))
    assert leftovers == []


def test_manifest_rejects_unknown_format_version(tmp_path):
    from oneuniverse.data.subobject import read_subobject_links

    man = _links_manifest_path(tmp_path, "bogus")
    man.parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / SUBOBJECT_DIR / "bogus.parquet").write_bytes(b"")
    man.write_text(json.dumps({"format_version": 99}))
    with pytest.raises(ValueError, match="format_version"):
        read_subobject_links(tmp_path, "bogus")
