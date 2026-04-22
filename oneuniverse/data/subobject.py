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


_REQUIRED_DTYPE = {
    "parent_oneuid": np.int64,
    "child_oneuid": np.int64,
    "confidence": np.float32,
    "sky_sep_arcsec": np.float32,
    "dz": np.float32,
}


def _radec_to_unit(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    cd = np.cos(dec)
    return np.column_stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)])


def _chord_tol(sky_tol_arcsec: float) -> float:
    ang = np.radians(sky_tol_arcsec / 3600.0)
    return 2.0 * np.sin(ang / 2.0)


def _chord_to_arcsec(d: np.ndarray) -> np.ndarray:
    return np.degrees(2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))) * 3600.0


def _build_subobject_pairs(
    parents: pd.DataFrame,
    children: pd.DataFrame,
    rules: SubobjectRules,
) -> pd.DataFrame:
    """Return a link table respecting the rules.

    Expected columns on both frames: ``oneuid``, ``ra``, ``dec``, ``z``
    (``z`` may contain ``NaN`` only on the child side when
    ``rules.dz_tol is None``).
    """
    from scipy.spatial import cKDTree

    required = {"oneuid", "ra", "dec", "z"}
    for label, df in (("parents", parents), ("children", children)):
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"_build_subobject_pairs: {label} missing columns {missing!r}"
            )

    if len(parents) == 0 or len(children) == 0:
        return pd.DataFrame(
            {c: pd.Series(dtype=_REQUIRED_DTYPE[c]) for c in REQUIRED_COLUMNS}
        )

    p_ra = parents["ra"].to_numpy(float)
    p_dec = parents["dec"].to_numpy(float)
    p_z = parents["z"].to_numpy(float)
    p_uid = parents["oneuid"].to_numpy(np.int64)

    c_ra = children["ra"].to_numpy(float)
    c_dec = children["dec"].to_numpy(float)
    c_z = children["z"].to_numpy(float)
    c_uid = children["oneuid"].to_numpy(np.int64)

    p_xyz = _radec_to_unit(p_ra, p_dec)
    c_xyz = _radec_to_unit(c_ra, c_dec)

    tree = cKDTree(p_xyz)
    chord = _chord_tol(rules.sky_tol_arcsec)

    neighbours = tree.query_ball_point(c_xyz, r=chord)

    parent_idx_out: List[int] = []
    child_idx_out: List[int] = []
    conf_out: List[float] = []
    sep_out: List[float] = []
    dz_out: List[float] = []

    for ci, cand in enumerate(neighbours):
        if not cand:
            continue
        cand_arr = np.asarray(cand, dtype=np.int64)
        if rules.dz_tol is not None and np.isfinite(c_z[ci]):
            dz = np.abs(p_z[cand_arr] - c_z[ci])
            keep = np.isfinite(dz) & (dz <= rules.dz_tol)
            cand_arr = cand_arr[keep]
        if cand_arr.size == 0:
            continue
        if cand_arr.size > 1 and not rules.accept_ambiguous:
            continue
        n_cand = cand_arr.size
        chords = np.linalg.norm(p_xyz[cand_arr] - c_xyz[ci], axis=1)
        seps = _chord_to_arcsec(chords)
        for k, pi in enumerate(cand_arr):
            parent_idx_out.append(int(pi))
            child_idx_out.append(ci)
            conf_out.append(1.0 / n_cand)
            sep_out.append(float(seps[k]))
            if rules.dz_tol is None or not np.isfinite(c_z[ci]):
                dz_out.append(np.nan)
            else:
                dz_out.append(float(p_z[pi] - c_z[ci]))

    return pd.DataFrame(
        {
            "parent_oneuid": np.asarray(
                p_uid[parent_idx_out] if parent_idx_out else [],
                dtype=np.int64,
            ),
            "child_oneuid": np.asarray(
                c_uid[child_idx_out] if child_idx_out else [],
                dtype=np.int64,
            ),
            "confidence": np.asarray(conf_out, dtype=np.float32),
            "sky_sep_arcsec": np.asarray(sep_out, dtype=np.float32),
            "dz": np.asarray(dz_out, dtype=np.float32),
        }
    )
