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
