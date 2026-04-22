"""
oneuniverse.data.subobject
~~~~~~~~~~~~~~~~~~~~~~~~~~
**Sub-object links** — bitemporal sidecar tables recording parent->child
containment between astrophysical objects (e.g. host galaxy -> SN).

Layout::

    {database_root}/_subobject/<name>.parquet
    {database_root}/_subobject/<name>.manifest.json

Schema (one row per (parent, child) pair):

    parent_oneuid     int64    ONEUID of the parent (host) object
    child_oneuid      int64    ONEUID of the child (e.g. SN) object
    confidence        float32  1.0 unambiguous; <1 for accepted
                               ambiguous matches
    sky_sep_arcsec    float32  on-sky separation at match time
    dz                float32  ``z_parent - z_child``; ``NaN`` when the
                               child has no redshift
"""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from oneuniverse.data.subobject_rules import SubobjectRules
from oneuniverse.data.validity import DatasetValidity


logger = logging.getLogger(__name__)

SUBOBJECT_DIR = "_subobject"
SUBOBJECT_MANIFEST_FORMAT_VERSION = 1
REQUIRED_COLUMNS: Tuple[str, ...] = (
    "parent_oneuid",
    "child_oneuid",
    "confidence",
    "sky_sep_arcsec",
    "dz",
)


def _links_path(root: Path, name: str) -> Path:
    return Path(root) / SUBOBJECT_DIR / f"{name}.parquet"


def _links_manifest_path(root: Path, name: str) -> Path:
    return Path(root) / SUBOBJECT_DIR / f"{name}.manifest.json"


@dataclass(frozen=True)
class SubobjectLinks:
    """In-memory view of a named sub-object link sidecar."""

    name: str
    rules: SubobjectRules
    parent_datasets: Tuple[str, ...]
    child_datasets: Tuple[str, ...]
    oneuid_name: str
    oneuid_hash: str
    validity: DatasetValidity
    table: pd.DataFrame

    def __post_init__(self) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in self.table.columns]
        if missing:
            raise ValueError(
                f"SubobjectLinks: table missing required columns {missing!r}"
            )

    def __len__(self) -> int:
        return len(self.table)


def _rules_to_dict(r: SubobjectRules) -> dict:
    return r._canonical()


def _rules_from_dict(d: dict) -> SubobjectRules:
    return SubobjectRules(
        parent_survey_type=d["parent_survey_type"],
        child_survey_type=d["child_survey_type"],
        sky_tol_arcsec=float(d["sky_tol_arcsec"]),
        dz_tol=None if d["dz_tol"] is None else float(d["dz_tol"]),
        relation=d["relation"],
        accept_ambiguous=bool(d["accept_ambiguous"]),
    )


def _manifest_to_dict(links: "SubobjectLinks") -> dict:
    return {
        "format_version": SUBOBJECT_MANIFEST_FORMAT_VERSION,
        "name": links.name,
        "rules": _rules_to_dict(links.rules),
        "rules_hash": links.rules.hash(),
        "parent_datasets": list(links.parent_datasets),
        "child_datasets": list(links.child_datasets),
        "oneuid_name": links.oneuid_name,
        "oneuid_hash": links.oneuid_hash,
        "validity": links.validity.to_dict(),
        "n_links": int(len(links.table)),
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
    }


def write_subobject_links(root: Path, links: "SubobjectLinks") -> None:
    root = Path(root)
    dest_table = _links_path(root, links.name)
    dest_manifest = _links_manifest_path(root, links.name)
    dest_table.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        dir=dest_table.parent, prefix=f".{links.name}.",
    ) as tmp:
        tmp_path = Path(tmp)
        tmp_table = tmp_path / "links.parquet"
        tmp_manifest = tmp_path / "manifest.json"
        links.table.to_parquet(tmp_table, index=False, compression="zstd")
        tmp_manifest.write_text(json.dumps(
            _manifest_to_dict(links), indent=2, sort_keys=True,
        ))
        shutil.move(str(tmp_table), str(dest_table))
        shutil.move(str(tmp_manifest), str(dest_manifest))


def read_subobject_links(root: Path, name: str) -> "SubobjectLinks":
    root = Path(root)
    man_path = _links_manifest_path(root, name)
    tbl_path = _links_path(root, name)
    if not man_path.exists():
        raise FileNotFoundError(
            f"read_subobject_links: manifest missing for {name!r} at {man_path}"
        )
    raw = json.loads(man_path.read_text())
    fmt = raw.get("format_version")
    if fmt != SUBOBJECT_MANIFEST_FORMAT_VERSION:
        raise ValueError(
            f"read_subobject_links: unsupported format_version {fmt!r} "
            f"for {name!r} (expected {SUBOBJECT_MANIFEST_FORMAT_VERSION})"
        )
    table = pd.read_parquet(tbl_path)
    return SubobjectLinks(
        name=raw["name"],
        rules=_rules_from_dict(raw["rules"]),
        parent_datasets=tuple(raw["parent_datasets"]),
        child_datasets=tuple(raw["child_datasets"]),
        oneuid_name=raw["oneuid_name"],
        oneuid_hash=raw["oneuid_hash"],
        validity=DatasetValidity.from_dict(raw["validity"]),
        table=table,
    )
