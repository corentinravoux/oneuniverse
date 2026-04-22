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

    def children_of(self, parent_oneuid: int) -> np.ndarray:
        """Child ONEUIDs under *parent_oneuid*, sorted by descending confidence."""
        mask = self.table["parent_oneuid"].to_numpy() == parent_oneuid
        if not mask.any():
            return np.empty(0, dtype=np.int64)
        sub = self.table.loc[mask].sort_values("confidence", ascending=False)
        return sub["child_oneuid"].to_numpy(dtype=np.int64)

    def parent_of(self, child_oneuid: int):
        """Parent ONEUID of *child_oneuid*.

        ``None`` if absent, ``int`` if unambiguous, ``list[int]`` sorted
        by descending confidence when ambiguous candidates exist.
        """
        mask = self.table["child_oneuid"].to_numpy() == child_oneuid
        if not mask.any():
            return None
        sub = self.table.loc[mask].sort_values("confidence", ascending=False)
        parents = sub["parent_oneuid"].to_numpy(dtype=np.int64).tolist()
        if len(parents) == 1:
            return parents[0]
        return parents


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


def _load_oneuid_for(
    database, oneuid_name: str, datasets: Sequence[str],
) -> pd.DataFrame:
    idx = database.load_oneuid(oneuid_name).restrict_to(list(datasets))
    return idx.table[
        ["oneuid", "dataset", "row_index", "ra", "dec", "z"]
    ].copy()


def _archive_previous(root: Path, name: str) -> None:
    """Close previous sidecar's validity and rename to archival name."""
    live_man = _links_manifest_path(root, name)
    if not live_man.exists():
        return
    raw = json.loads(live_man.read_text())
    now = datetime.now(tz=timezone.utc)
    now_iso = now.isoformat()
    validity = DatasetValidity.from_dict(raw["validity"]).closed_at(now_iso)
    raw["validity"] = validity.to_dict()

    ts = now.strftime("%Y%m%dT%H%M%SZ")
    archive_base = f"{name}__{ts}"
    archive_table = _links_path(root, archive_base)
    archive_man = _links_manifest_path(root, archive_base)

    live_table = _links_path(root, name)
    if live_table.exists():
        shutil.move(str(live_table), str(archive_table))
    archive_man.write_text(json.dumps(raw, indent=2, sort_keys=True))
    live_man.unlink()


def build_subobject_links(
    database,
    *,
    rules: SubobjectRules,
    parent_datasets: Sequence[str],
    child_datasets: Sequence[str],
    name: str = "default",
    oneuid_name: str = "default",
) -> int:
    """Build and persist a named sub-object link sidecar. Returns row count."""
    root = Path(database.root)
    parent_datasets = tuple(parent_datasets)
    child_datasets = tuple(child_datasets)
    if not parent_datasets:
        raise ValueError("build_subobject_links: parent_datasets is empty")
    if not child_datasets:
        raise ValueError("build_subobject_links: child_datasets is empty")

    for d in parent_datasets:
        st = database.entry(d).manifest.survey_type
        if st != rules.parent_survey_type:
            raise ValueError(
                f"build_subobject_links: parent dataset {d!r} has "
                f"survey_type={st!r}, rules require "
                f"{rules.parent_survey_type!r}"
            )
    for d in child_datasets:
        st = database.entry(d).manifest.survey_type
        if st != rules.child_survey_type:
            raise ValueError(
                f"build_subobject_links: child dataset {d!r} has "
                f"survey_type={st!r}, rules require "
                f"{rules.child_survey_type!r}"
            )

    parent_idx = _load_oneuid_for(database, oneuid_name, parent_datasets)
    child_idx = _load_oneuid_for(database, oneuid_name, child_datasets)

    frames: List[pd.DataFrame] = []
    for p_name in parent_datasets:
        parents = parent_idx[parent_idx["dataset"] == p_name].drop_duplicates(
            subset="oneuid"
        )
        for c_name in child_datasets:
            children = child_idx[child_idx["dataset"] == c_name].drop_duplicates(
                subset="oneuid"
            )
            frames.append(_build_subobject_pairs(parents, children, rules))

    combined = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            {c: pd.Series(dtype=_REQUIRED_DTYPE[c]) for c in REQUIRED_COLUMNS}
        )
    )

    try:
        oneuid_hash = database.load_oneuid(oneuid_name).rules.hash()
    except Exception:
        oneuid_hash = "unknown"

    links = SubobjectLinks(
        name=name,
        rules=rules,
        parent_datasets=parent_datasets,
        child_datasets=child_datasets,
        oneuid_name=oneuid_name,
        oneuid_hash=oneuid_hash,
        validity=DatasetValidity(
            valid_from_utc=datetime.now(tz=timezone.utc).isoformat(),
            version=name,
        ),
        table=combined,
    )

    _archive_previous(root, name)
    write_subobject_links(root, links)
    logger.info(
        "subobject: wrote %d links to %s (rules hash %s)",
        len(links), _links_path(root, name), rules.hash(),
    )
    return len(links)


def load_subobject_links(
    root: Path,
    name: str = "default",
    *,
    as_of: Optional[datetime] = None,
) -> "SubobjectLinks":
    """Load a named sub-object link sidecar.

    With ``as_of`` (tz-aware), return the archived version whose
    :class:`DatasetValidity` contains that timestamp.
    """
    root = Path(root)
    if as_of is None:
        return read_subobject_links(root, name)

    if as_of.tzinfo is None:
        raise ValueError("load_subobject_links: as_of must be tz-aware")

    candidates: List[Path] = []
    live = _links_manifest_path(root, name)
    if live.exists():
        candidates.append(live)
    subdir = root / SUBOBJECT_DIR
    if subdir.exists():
        for p in subdir.glob(f"{name}__*.manifest.json"):
            candidates.append(p)

    for man in candidates:
        raw = json.loads(man.read_text())
        v = DatasetValidity.from_dict(raw["validity"])
        if v.contains(as_of):
            stem = man.name[: -len(".manifest.json")]
            return read_subobject_links(root, stem)

    raise FileNotFoundError(
        f"load_subobject_links: no version of {name!r} valid at {as_of.isoformat()}"
    )


def list_subobject_link_sets(
    root: Path, *, include_archived: bool = False,
) -> List[str]:
    root = Path(root)
    subdir = root / SUBOBJECT_DIR
    if not subdir.exists():
        return []
    names = []
    for p in sorted(subdir.glob("*.manifest.json")):
        stem = p.name[: -len(".manifest.json")]
        is_archived = "__" in stem
        if is_archived and not include_archived:
            continue
        names.append(stem)
    return names
