"""Sub-object links: parent <-> child ONEUIDs (host galaxy <-> supernova, cluster <-> member).

Links are computed between ONEUID index rows of *parent* datasets and *child*
datasets (positions matched with :func:`match_pair`) and published on the
shared versioned sidecar store (kind ``links``). Each link set records the
ONEUID index name and sidecar version it was computed on; :func:`chain` refuses
to walk link sets built on different ONEUID indexes, because their ids are not
comparable.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pyarrow as pa

from oneuniverse._core.errors import ValidationError
from oneuniverse._core.healpix import radec_to_pix
from oneuniverse.data.crossmatch import CrossMatchRules, match_pair
from oneuniverse.data.oneuid import load_oneuid
from oneuniverse.data.sidecar import VersionedSidecar

__all__ = ["LINKS_SCHEMA", "SubobjectLinks", "SubobjectRules", "build_links", "chain", "load_links"]

LINKS_SCHEMA = pa.schema([
    ("parent_oneuid", pa.int64()),
    ("child_oneuid", pa.int64()),
    ("sep_arcsec", pa.float64()),
    ("dz", pa.float64()),
    ("confidence", pa.float32()),
])

_KIND = "links"
_RELATIONS = ("contains", "associated", "caused")


@dataclass(frozen=True, slots=True)
class SubobjectRules:
    """Which datasets are parents / children and how a child is matched to a parent.

    ``dz_tol=None`` disables the redshift cut. A child with several candidate
    parents is dropped unless ``accept_ambiguous``, in which case each of its
    links gets ``confidence = 1 / n_candidates``.
    """

    parent_kind: str
    child_kind: str
    sky_tol_arcsec: float = 1.0
    dz_tol: float | None = 5e-3
    relation: Literal["contains", "associated", "caused"] = "contains"
    accept_ambiguous: bool = False

    def __post_init__(self) -> None:
        if self.relation not in _RELATIONS:
            raise ValidationError(f"relation must be one of {_RELATIONS}, got {self.relation!r}")
        self.match_rules()  # validates the tolerances

    def match_rules(self) -> CrossMatchRules:
        return CrossMatchRules(sky_tol_arcsec=self.sky_tol_arcsec, dz_tol=self.dz_tol)

    def to_dict(self) -> dict:
        return {
            "parent_kind": self.parent_kind, "child_kind": self.child_kind,
            "sky_tol_arcsec": float(self.sky_tol_arcsec),
            "dz_tol": None if self.dz_tol is None else float(self.dz_tol),
            "relation": self.relation, "accept_ambiguous": bool(self.accept_ambiguous),
        }

    @classmethod
    def from_dict(cls, d: dict) -> SubobjectRules:
        return cls(d["parent_kind"], d["child_kind"], float(d["sky_tol_arcsec"]),
                   None if d["dz_tol"] is None else float(d["dz_tol"]),
                   d["relation"], bool(d["accept_ambiguous"]))

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _rows(index_table: pa.Table, ds_names: np.ndarray, datasets: Sequence[str]):
    """(match table, oneuid array) for the index rows of *datasets*."""
    sel = np.flatnonzero(np.isin(ds_names, list(datasets)))
    t = index_table.take(pa.array(sel))
    ra, dec = t["ra"].to_numpy(), t["dec"].to_numpy()
    tab = pa.table({
        "ra": ra, "dec": dec, "z": t["z"].to_numpy(),
        "z_type": t["z_type"].combine_chunks().cast(pa.string()),
        "hpix32": radec_to_pix(32, ra, dec),
    })
    return tab, t["oneuid"].to_numpy()


def build_links(db, *, parents: Sequence[str], children: Sequence[str], rules: SubobjectRules,
                name: str = "default", oneuid_name: str = "default") -> SubobjectLinks:
    """Match *children* to *parents* on ONEUID index *oneuid_name* and publish link set *name*.

    Raises ValidationError if a dataset is not in the index or its
    ``survey.kind`` differs from ``rules.parent_kind`` / ``rules.child_kind``.
    """
    parents, children = sorted(set(parents)), sorted(set(children))
    if not parents or not children:
        raise ValidationError("build_links needs at least one parent and one child dataset")
    for ds, want, role in ([(d, rules.parent_kind, "parent") for d in parents]
                           + [(d, rules.child_kind, "child") for d in children]):
        kind = db.manifest(ds).survey.kind  # NotFoundError for unknown names
        if kind != want:
            raise ValidationError(
                f"{role} dataset {ds!r} has kind {kind!r}, rules expect {role}_kind {want!r}")
    idx = load_oneuid(db, oneuid_name)
    missing = [d for d in parents + children if d not in idx.datasets]
    if missing:
        raise ValidationError(f"datasets {missing} are not members of ONEUID {oneuid_name!r}")
    d = idx.table["dataset"].combine_chunks()
    ds_names = np.asarray(d.dictionary.to_pylist(), dtype=object)[
        d.indices.to_numpy(zero_copy_only=False)]
    ptab, poid = _rows(idx.table, ds_names, parents)
    ctab, coid = _rows(idx.table, ds_names, children)
    pairs = match_pair(ptab, ctab, rules.match_rules())
    pid, cid = poid[pairs.left], coid[pairs.right]
    sep, dz = pairs.sep_arcsec, pairs.dz
    keep = pid != cid  # never link an object to itself
    pid, cid, sep, dz = pid[keep], cid[keep], sep[keep], dz[keep]
    # collapse multi-member objects: one link per (child, parent) oneuid pair
    order = np.lexsort((pid, cid))
    pid, cid, sep, dz = pid[order], cid[order], sep[order], dz[order]
    if len(pid):
        first = np.ones(len(pid), bool)
        first[1:] = (pid[1:] != pid[:-1]) | (cid[1:] != cid[:-1])
        st = np.flatnonzero(first)
        sep, dz = np.minimum.reduceat(sep, st), np.fmin.reduceat(dz, st)
        pid, cid = pid[st], cid[st]
    # ambiguity: distinct candidate parents per child
    if len(cid):
        _, inv, n = np.unique(cid, return_inverse=True, return_counts=True)
        ncand = n[inv.ravel()]
    else:
        n, ncand = np.empty(0, np.int64), np.empty(0, np.int64)
    n_amb = int((n > 1).sum())
    ok = np.ones(len(cid), bool) if rules.accept_ambiguous else ncand == 1
    table = pa.table({
        "parent_oneuid": pa.array(pid[ok], pa.int64()),
        "child_oneuid": pa.array(cid[ok], pa.int64()),
        "sep_arcsec": pa.array(sep[ok], pa.float64()),
        "dz": pa.array(dz[ok], pa.float64()),
        "confidence": pa.array((1.0 / ncand[ok]).astype(np.float32), pa.float32()),
    }, schema=LINKS_SCHEMA)
    meta = {
        "rules": rules.to_dict(), "rules_hash": rules.hash(), "oneuid_name": oneuid_name,
        "oneuid_version": idx.meta["version"], "parents": parents, "children": children,
        "n_links": table.num_rows, "n_ambiguous_children": n_amb,
    }
    VersionedSidecar(db.root, _KIND, name).write(table, meta)
    return load_links(db, name)


def load_links(db, name: str = "default", *, as_of=None) -> SubobjectLinks:
    """Load link set *name* (the version live at *as_of* if given)."""
    table, meta = VersionedSidecar(db.root, _KIND, name).load(as_of=as_of)
    return SubobjectLinks(table, meta, name)


class SubobjectLinks:
    """A ``LINKS_SCHEMA`` table sorted by (child_oneuid, parent_oneuid) + meta."""

    def __init__(self, table: pa.Table, meta: dict, name: str) -> None:
        self.table = table
        self.meta = meta
        self.name = name

    @property
    def rules(self) -> SubobjectRules:
        return SubobjectRules.from_dict(self.meta["rules"])

    def children_of(self, oneuid: int) -> np.ndarray:
        """Child oneuids of *oneuid*, by confidence descending then oneuid (int64)."""
        t = self.table
        m = t["parent_oneuid"].to_numpy() == int(oneuid)
        ch, conf = t["child_oneuid"].to_numpy()[m], t["confidence"].to_numpy()[m]
        return ch[np.lexsort((ch, -conf))].astype(np.int64)

    def parent_of(self, oneuid: int) -> int | list[int] | None:
        """The parent oneuid (int), a sorted list if several, or None."""
        t = self.table
        m = t["child_oneuid"].to_numpy() == int(oneuid)
        par = np.sort(t["parent_oneuid"].to_numpy()[m])
        if par.size == 0:
            return None
        return int(par[0]) if par.size == 1 else [int(p) for p in par]


def chain(db, starts: Sequence[int], names: Sequence[str], *, as_of=None) -> np.ndarray:
    """Follow link sets *names* in order (parents to children) from *starts*.

    Returns the sorted unique leaf oneuids (int64); no names returns *starts*
    unchanged. Raises ValidationError if consecutive link sets were built on
    different ONEUID indexes.
    """
    cur = np.asarray(starts, dtype=np.int64).ravel()
    if not names:
        return cur
    cur = np.unique(cur)
    prev: tuple[str, str] | None = None
    for n in names:
        links = load_links(db, n, as_of=as_of)
        if prev is not None and links.meta["oneuid_name"] != prev[1]:
            raise ValidationError(
                f"link set {n!r} is built on ONEUID {links.meta['oneuid_name']!r} but {prev[0]!r} "
                f"on ONEUID {prev[1]!r}; ids are not comparable")
        prev = (n, links.meta["oneuid_name"])
        t = links.table
        m = np.isin(t["parent_oneuid"].to_numpy(), cur)
        cur = np.unique(t["child_oneuid"].to_numpy()[m]).astype(np.int64)
    return cur
