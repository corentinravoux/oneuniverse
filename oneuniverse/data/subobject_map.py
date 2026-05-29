"""Map-based sub-object linker for OUF 2.4.

Cross-match a point catalog of parents against a catalog of events
whose rows carry a per-row HEALPix probability map (variable-length
``list<f4>`` column, see Phase 17). Returns the canonical
:class:`SubobjectLinks` sidecar with ``confidence`` set to the
parent's pixel value in the event map.

Typical use: GW × galaxy host association (each event has a
sky-localisation map, the parents are galaxies in a redshift shell).
"""
from __future__ import annotations

from typing import Optional

import healpy as hp
import numpy as np
import pandas as pd

from oneuniverse.data.subobject import REQUIRED_COLUMNS, SubobjectLinks
from oneuniverse.data.subobject_rules import SubobjectRules
from oneuniverse.data.validity import DatasetValidity


def build_subobject_links_to_map(
    parents: pd.DataFrame,
    events: pd.DataFrame,
    *,
    map_column: str,
    map_nside: int,
    map_nest: bool = True,
    threshold: float = 0.0,
    name: str = "default",
    rules: Optional[SubobjectRules] = None,
    oneuid_name: str = "default",
    oneuid_hash: str = "",
    validity: Optional[DatasetValidity] = None,
) -> SubobjectLinks:
    """Match each ``parents`` row against every ``events[map_column]``
    and emit a :class:`SubobjectLinks` whose ``confidence`` is the
    parent's pixel value in the event map.

    Parameters
    ----------
    parents
        Catalog with ``oneuid``, ``ra``, ``dec`` columns. Parents whose
        pixel value is below ``threshold`` in a given event are dropped.
    events
        Catalog with ``oneuid`` and a ``map_column`` whose rows are
        ``numpy.ndarray[f4]`` of length ``12 * map_nside²``.
    map_nside, map_nest
        Fixed HEALPix NSIDE and ordering of every event map.
    threshold
        Minimum pixel value to record a link. ``0.0`` keeps every
        non-NaN pixel.
    rules
        Optional explicit :class:`SubobjectRules`. Default builds a
        ``relation_type="association"`` rule with sentinel survey types
        (``"map_event" -> "host"``).
    """
    expected_len = 12 * map_nside * map_nside

    parent_ids = parents["oneuid"].to_numpy(dtype="i8")
    parent_ra = parents["ra"].to_numpy(dtype="f8")
    parent_dec = parents["dec"].to_numpy(dtype="f8")
    parent_pix = hp.ang2pix(
        map_nside, parent_ra, parent_dec, nest=map_nest, lonlat=True,
    )

    event_ids = events["oneuid"].to_numpy(dtype="i8")
    maps = events[map_column].to_numpy()

    parent_acc = []
    child_acc = []
    conf_acc = []
    for evt_id, m in zip(event_ids, maps):
        arr = np.asarray(m, dtype="f4")
        if arr.size != expected_len:
            raise ValueError(
                f"event {int(evt_id)}: map length {arr.size} does not "
                f"match expected length {expected_len} for NSIDE="
                f"{map_nside}"
            )
        probs = arr[parent_pix]
        keep = probs >= threshold
        if not keep.any():
            continue
        parent_acc.append(parent_ids[keep])
        child_acc.append(np.full(keep.sum(), int(evt_id), dtype="i8"))
        conf_acc.append(probs[keep].astype("f4"))

    if not parent_acc:
        table = pd.DataFrame({
            "parent_oneuid": pd.Series(dtype="i8"),
            "child_oneuid": pd.Series(dtype="i8"),
            "confidence": pd.Series(dtype="f4"),
            "sky_sep_arcsec": pd.Series(dtype="f4"),
            "dz": pd.Series(dtype="f4"),
        })
    else:
        parents_flat = np.concatenate(parent_acc)
        children_flat = np.concatenate(child_acc)
        conf_flat = np.concatenate(conf_acc)
        table = pd.DataFrame({
            "parent_oneuid": parents_flat,
            "child_oneuid": children_flat,
            "confidence": conf_flat,
            "sky_sep_arcsec": np.zeros(parents_flat.size, dtype="f4"),
            "dz": np.zeros(parents_flat.size, dtype="f4"),
        })

    rules = rules or SubobjectRules(
        parent_survey_type="map_event",
        child_survey_type="host",
        sky_tol_arcsec=1.0,
        dz_tol=None,
        relation="contains",
        accept_ambiguous=True,
        relation_type="association",
    )
    validity = validity or DatasetValidity(
        valid_from_utc="2026-05-29T00:00:00+00:00",
    )
    return SubobjectLinks(
        name=name,
        rules=rules,
        parent_datasets=("events",),
        child_datasets=("parents",),
        oneuid_name=oneuid_name,
        oneuid_hash=oneuid_hash,
        validity=validity,
        table=table,
    )
