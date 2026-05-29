"""Phase 20 T1/T2 — SubobjectRules gains relation_type + next_level."""
import json

import numpy as np
import pandas as pd
import pytest

from oneuniverse.data.subobject import (
    SubobjectLinks,
    read_subobject_links,
    write_subobject_links,
)
from oneuniverse.data.subobject_rules import SubobjectRules
from oneuniverse.data.validity import DatasetValidity


def test_defaults():
    r = SubobjectRules(
        parent_survey_type="cluster", child_survey_type="galaxy",
    )
    assert r.relation_type == "association"
    assert r.next_level is None


def test_explicit_relation_type():
    r = SubobjectRules(
        parent_survey_type="cluster", child_survey_type="galaxy",
        relation_type="containment",
    )
    assert r.relation_type == "containment"


def test_rejects_unknown_relation_type():
    with pytest.raises(ValueError, match="relation_type"):
        SubobjectRules(
            parent_survey_type="a", child_survey_type="b",
            relation_type="bogus",
        )


def test_next_level_chain_pointer():
    r = SubobjectRules(
        parent_survey_type="cluster", child_survey_type="galaxy",
        next_level="galaxy_to_spectrum",
    )
    assert r.next_level == "galaxy_to_spectrum"


def test_hash_includes_new_fields():
    a = SubobjectRules(
        parent_survey_type="cluster", child_survey_type="galaxy",
        relation_type="containment",
    )
    b = SubobjectRules(
        parent_survey_type="cluster", child_survey_type="galaxy",
        relation_type="causality",
    )
    assert a.hash() != b.hash()


# ── T2 round-trip via sidecar manifest ──────────────────────────────────


def _make_links(rules: SubobjectRules) -> SubobjectLinks:
    table = pd.DataFrame({
        "parent_oneuid": np.array([1, 2], dtype="i8"),
        "child_oneuid":  np.array([10, 11], dtype="i8"),
        "confidence":    np.array([1.0, 0.8], dtype="f4"),
        "sky_sep_arcsec": np.array([0.3, 0.5], dtype="f4"),
        "dz":             np.array([0.0, 1e-4], dtype="f4"),
    })
    return SubobjectLinks(
        name="test_chain",
        rules=rules,
        parent_datasets=("parents",),
        child_datasets=("children",),
        oneuid_name="default",
        oneuid_hash="0123456789abcdef",
        validity=DatasetValidity(
            valid_from_utc="2026-05-29T00:00:00+00:00",
        ),
        table=table,
    )


def test_relation_type_and_next_level_roundtrip(tmp_path):
    rules = SubobjectRules(
        parent_survey_type="cluster",
        child_survey_type="galaxy",
        relation_type="containment",
        next_level="galaxy_to_spectrum",
    )
    links = _make_links(rules)
    write_subobject_links(tmp_path, links)
    read = read_subobject_links(tmp_path, "test_chain")
    assert read.rules.relation_type == "containment"
    assert read.rules.next_level == "galaxy_to_spectrum"


def test_v1_manifest_parses_with_default_relation(tmp_path):
    """A pre-Phase-20 (v1) sidecar must still parse."""
    rules = SubobjectRules(
        parent_survey_type="cluster",
        child_survey_type="galaxy",
    )
    links = _make_links(rules)
    write_subobject_links(tmp_path, links)

    # Re-write the manifest as v1 (drop the new fields).
    manifest_path = tmp_path / "_subobject" / "test_chain.manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["format_version"] = 1
    payload["rules"].pop("relation_type", None)
    payload["rules"].pop("next_level", None)
    manifest_path.write_text(json.dumps(payload))

    read = read_subobject_links(tmp_path, "test_chain")
    assert read.rules.relation_type == "association"
    assert read.rules.next_level is None
