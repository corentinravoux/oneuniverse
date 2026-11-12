"""Pair-wise sky/redshift cross-matching and greedy identity components.

Matching is always between two catalogues, blocked by coarse HEALPix cell plus
its 8 neighbours. One KD-tree is built per block on the candidate rows of the
second catalogue, so peak memory is bounded by the densest block pair rather
than by the full catalogues. There is never a self-join of all surveys.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

import healpy as hp
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from oneuniverse._core.errors import SchemaError, ValidationError
from oneuniverse._core.healpix import NSIDE_FINE, coarsen
from oneuniverse.data.schema import z_types

__all__ = ["CrossMatchRules", "Pairs", "match_pair", "components"]

_ARCSEC = np.pi / 180.0 / 3600.0
_REQUIRED = ("ra", "dec", "z", "z_type", "hpix32")


def _norm_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


@dataclass(frozen=True, slots=True)
class CrossMatchRules:
    """Rules deciding when two rows of different surveys are the same object.

    z cut: a candidate is kept if the z_type pair is accepted and either the
    tolerance for that pair is nan (no z cut; always the case when either
    z_type is ``"none"``) or ``dz <= tol`` with ``dz = |z_a - z_b|``. When
    either z is nan, ``dz`` is nan and the candidate passes only if the
    tolerance is nan.
    """

    sky_tol_arcsec: float = 1.0
    dz_tol: float | None = 5e-3
    dz_tol_by_pair: tuple[tuple[str, str, float], ...] = ()
    reject_pairs: tuple[tuple[str, str], ...] = ()
    allow_same_survey_merge: bool = False

    def __post_init__(self) -> None:
        if not (self.sky_tol_arcsec > 0):
            raise ValidationError("sky_tol_arcsec must be > 0")
        if self.dz_tol is not None and not (self.dz_tol >= 0):
            raise ValidationError("dz_tol must be None or >= 0")
        for t in self.dz_tol_by_pair:
            if len(t) != 3 or not (t[2] >= 0):
                raise ValidationError(f"bad dz_tol_by_pair entry {t!r}")
        for t in self.reject_pairs:
            if len(t) != 2:
                raise ValidationError(f"bad reject_pairs entry {t!r}")

    def to_dict(self) -> dict:
        return {
            "sky_tol_arcsec": float(self.sky_tol_arcsec),
            "dz_tol": None if self.dz_tol is None else float(self.dz_tol),
            "dz_tol_by_pair": [list(t) for t in self.dz_tol_by_pair],
            "reject_pairs": [list(t) for t in self.reject_pairs],
            "allow_same_survey_merge": bool(self.allow_same_survey_merge),
        }

    @classmethod
    def from_dict(cls, d: dict) -> CrossMatchRules:
        return cls(
            sky_tol_arcsec=float(d["sky_tol_arcsec"]),
            dz_tol=None if d["dz_tol"] is None else float(d["dz_tol"]),
            dz_tol_by_pair=tuple((str(a), str(b), float(t)) for a, b, t in d["dz_tol_by_pair"]),
            reject_pairs=tuple((str(a), str(b)) for a, b in d["reject_pairs"]),
            allow_same_survey_merge=bool(d["allow_same_survey_merge"]),
        )

    def hash(self) -> str:
        d = self.to_dict()
        d["dz_tol_by_pair"] = sorted([*_norm_pair(a, b), t] for a, b, t in self.dz_tol_by_pair)
        d["reject_pairs"] = sorted(list(_norm_pair(a, b)) for a, b in self.reject_pairs)
        blob = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def matrices(self, z_types: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(tol, accept)``; ``tol`` is nan where no z cut applies."""
        names = list(z_types)
        index = {n: i for i, n in enumerate(names)}
        for a, b, _ in self.dz_tol_by_pair:
            for n in (a, b):
                if n not in index:
                    raise ValidationError(f"unknown z_type {n!r} in dz_tol_by_pair")
        for a, b in self.reject_pairs:
            for n in (a, b):
                if n not in index:
                    raise ValidationError(f"unknown z_type {n!r} in reject_pairs")
        n = len(names)
        default = np.nan if self.dz_tol is None else float(self.dz_tol)
        tol = np.full((n, n), default, dtype=np.float64)
        for a, b, t in self.dz_tol_by_pair:
            tol[index[a], index[b]] = tol[index[b], index[a]] = t
        if "none" in index:
            k = index["none"]
            tol[k, :] = np.nan
            tol[:, k] = np.nan
        accept = np.ones((n, n), dtype=bool)
        for a, b in self.reject_pairs:
            accept[index[a], index[b]] = accept[index[b], index[a]] = False
        return tol, accept


@dataclass
class Pairs:
    left: np.ndarray
    right: np.ndarray
    sep_arcsec: np.ndarray
    dz: np.ndarray

    @classmethod
    def empty(cls) -> Pairs:
        return cls(np.empty(0, np.int64), np.empty(0, np.int64),
                   np.empty(0, np.float64), np.empty(0, np.float64))


def _col(t: pa.Table, name: str) -> pa.ChunkedArray:
    if name not in t.column_names:
        raise SchemaError(f"cross-match table is missing column {name!r}")
    return t.column(name)


def _prepare(t: pa.Table, names: list[str]):
    cols = [_col(t, c) for c in _REQUIRED]
    ra = np.asarray(cols[0].to_numpy(), dtype=np.float64)
    dec = np.asarray(cols[1].to_numpy(), dtype=np.float64)
    z = np.asarray(cols[2].to_numpy(), dtype=np.float64)
    zt = cols[3].combine_chunks() if cols[3].num_chunks else pa.array([], pa.string())
    if not pa.types.is_dictionary(zt.type):
        zt = pc.dictionary_encode(zt)
    dic = zt.dictionary.to_pylist()
    lookup = {n: i for i, n in enumerate(names)}
    bad = [v for v in dic if v not in lookup]
    if bad:
        raise SchemaError(f"unknown z_type values {bad!r}")
    remap = np.array([lookup[v] for v in dic], dtype=np.int64)
    idx = zt.indices.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
    code = remap[idx] if len(remap) else np.empty(0, np.int64)
    hpix = np.asarray(cols[4].to_numpy(), dtype=np.int64)
    r, d = np.radians(ra), np.radians(dec)
    xyz = np.column_stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])
    return xyz, z, code, hpix


def match_pair(a: pa.Table, b: pa.Table, rules: CrossMatchRules, *, block_nside: int = 8) -> Pairs:
    """Match rows of ``a`` to rows of ``b`` within ``rules.sky_tol_arcsec``.

    ``left`` indexes ``a``, ``right`` indexes ``b``; output is sorted by
    ``(left, right)`` and deduplicated. Candidates are blocked by coarse
    HEALPix cell (``block_nside``) and its 8 neighbours with one KD-tree per
    block over ``b``'s candidate rows, so memory is bounded by the densest
    block pair. See ``CrossMatchRules`` for the z-cut rule.
    """
    block_res = hp.nside2resol(block_nside)  # rad
    if rules.sky_tol_arcsec * _ARCSEC > 0.5 * block_res:
        raise ValidationError(
            f"sky_tol_arcsec={rules.sky_tol_arcsec} too large for block_nside={block_nside}"
        )
    names = sorted(z_types())
    tol_m, acc_m = rules.matrices(names)
    xa, za, ca, ha = _prepare(a, names)
    xb, zb, cb, hb = _prepare(b, names)
    if len(xa) == 0 or len(xb) == 0:
        return Pairs.empty()
    blk_a = coarsen(ha, NSIDE_FINE, block_nside)
    blk_b = coarsen(hb, NSIDE_FINE, block_nside)
    order_a = np.argsort(blk_a, kind="stable")
    order_b = np.argsort(blk_b, kind="stable")
    sa, sb = blk_a[order_a], blk_b[order_b]
    chord = 2.0 * np.sin(0.5 * rules.sky_tol_arcsec * _ARCSEC)
    cells, starts_a = np.unique(sa, return_index=True)
    ends_a = np.append(starts_a[1:], len(sa))
    L, R, S = [], [], []
    for cell, s0, s1 in zip(cells.tolist(), starts_a, ends_a):
        nb = hp.get_all_neighbours(block_nside, cell, nest=True)
        want = np.unique(np.append(nb[nb >= 0], cell))
        lo = np.searchsorted(sb, want, side="left")
        hi = np.searchsorted(sb, want, side="right")
        parts = [order_b[l:h] for l, h in zip(lo, hi) if h > l]
        if not parts:
            continue
        rows_b = np.concatenate(parts)
        rows_a = order_a[s0:s1]
        ta, tb = cKDTree(xa[rows_a]), cKDTree(xb[rows_b])
        m = ta.sparse_distance_matrix(tb, chord, output_type="ndarray")
        if len(m) == 0:
            continue
        ia, ib = rows_a[m["i"]], rows_b[m["j"]]
        L.append(ia)
        R.append(ib)
        S.append(2.0 * np.arcsin(np.minimum(m["v"] / 2.0, 1.0)) / _ARCSEC)
    if not L:
        return Pairs.empty()
    left, right, sep = np.concatenate(L), np.concatenate(R), np.concatenate(S)
    dz = np.abs(za[left] - zb[right])
    tol = tol_m[ca[left], cb[right]]
    keep = acc_m[ca[left], cb[right]] & (np.isnan(tol) | (dz <= tol))
    left, right, sep, dz = left[keep], right[keep], sep[keep], dz[keep]
    o = np.lexsort((right, left))
    left, right, sep, dz = left[o], right[o], sep[o], dz[o]
    if len(left):
        first = np.ones(len(left), bool)
        first[1:] = (left[1:] != left[:-1]) | (right[1:] != right[:-1])
        left, right, sep, dz = left[first], right[first], sep[first], dz[first]
    return Pairs(left.astype(np.int64), right.astype(np.int64),
                 sep.astype(np.float64), dz.astype(np.float64))


def _renumber(root: np.ndarray) -> np.ndarray:
    _, first, inv = np.unique(root, return_index=True, return_inverse=True)
    rank = np.empty(len(first), np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(len(first))
    return rank[inv.ravel()]


def components(n: int, left, right, sep_arcsec, *, survey_of: np.ndarray,
               allow_same_survey_merge: bool) -> np.ndarray:
    """Label connected rows 0..k-1 (by first appearance in row order).

    With ``allow_same_survey_merge=False`` edges are taken greedily in order of
    increasing separation (ties by left, right) and an edge is skipped when the
    two groups it would join already share a survey.
    """
    left = np.asarray(left, np.int64)
    right = np.asarray(right, np.int64)
    if allow_same_survey_merge:
        g = coo_matrix((np.ones(len(left), np.int8), (left, right)), shape=(n, n))
        _, lab = connected_components(g, directed=False)
        return _renumber(lab)
    order = np.lexsort((right, left, np.asarray(sep_arcsec, np.float64)))
    parent = np.arange(n)
    surveys = [{int(s)} for s in np.asarray(survey_of)]

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k in order.tolist():
        ra_, rb_ = find(int(left[k])), find(int(right[k]))
        if ra_ == rb_ or (surveys[ra_] & surveys[rb_]):
            continue
        if ra_ > rb_:
            ra_, rb_ = rb_, ra_
        parent[rb_] = ra_
        surveys[ra_] |= surveys[rb_]
        surveys[rb_] = set()
    roots = np.array([find(i) for i in range(n)], np.int64)
    return _renumber(roots)
