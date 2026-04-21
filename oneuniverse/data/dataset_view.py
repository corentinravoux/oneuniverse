"""
oneuniverse.data.dataset_view
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Lazy, filter-aware view over a converted OUF 2.0 dataset directory.

:class:`DatasetView` replaces the dynamic ``_make_loader_class`` path:
each entry in an :class:`OneuniverseDatabase` owns one :class:`DatasetView`.
The view is backed by :mod:`pyarrow.dataset` so column projection and row
filters push down into the Parquet readers — no full-table materialisation
until the caller asks for it.

Partition pruning
-----------------
The typed :class:`Manifest` stores per-partition min/max for ``ra``,
``dec``, and ``z``.  When the caller provides a bounding filter on any
of those columns, :meth:`DatasetView._select_partitions` skips partition
files whose stats cannot overlap the filter — *before* arrow opens them.

Usage
-----
>>> view = DatasetView.from_path(survey_dir)
>>> view.columns
['ra', 'dec', 'z', 'z_type', ...]
>>> tbl = view.scan(columns=['ra', 'dec', 'z'],
...                 z_range=(0.05, 0.10))
>>> df = view.read(columns=['ra', 'dec'])
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from oneuniverse.data.format_spec import (
    HEALPIX_PARTITION_NEST,
    HEALPIX_PARTITION_NSIDE,
    MANIFEST_FILENAME,
    OBJECTS_FILENAME,
    ONEUNIVERSE_SUBDIR,
    DataGeometry,
)
from oneuniverse.data.manifest import Manifest, PartitionSpec, read_manifest
from oneuniverse.data.selection import Cone, SkyPatch


# ── Range type ───────────────────────────────────────────────────────────

Range = Tuple[Optional[float], Optional[float]]


# ── DatasetView ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DatasetView:
    """Lazy handle to one converted OUF dataset directory.

    Holds the typed :class:`Manifest` and the ``oneuniverse/`` path; does
    not open Parquet files until :meth:`scan` or :meth:`read` is called.
    """

    ou_dir: Path
    manifest: Manifest

    # ── Construction ─────────────────────────────────────────────────────

    @classmethod
    def from_path(cls, survey_path: Union[str, Path]) -> "DatasetView":
        """Build a view from a survey directory (the parent of ``oneuniverse/``)."""
        survey_path = Path(survey_path)
        ou_dir = survey_path / ONEUNIVERSE_SUBDIR
        return cls.from_ou_dir(ou_dir)

    @classmethod
    def from_ou_dir(cls, ou_dir: Union[str, Path]) -> "DatasetView":
        """Build a view from the ``oneuniverse/`` directory itself."""
        ou_dir = Path(ou_dir)
        manifest = read_manifest(ou_dir / MANIFEST_FILENAME)
        return cls(ou_dir=ou_dir, manifest=manifest)

    # ── Introspection ────────────────────────────────────────────────────

    @property
    def geometry(self) -> DataGeometry:
        return self.manifest.geometry

    @property
    def n_rows(self) -> int:
        return self.manifest.n_rows

    @property
    def n_partitions(self) -> int:
        return self.manifest.n_partitions

    @property
    def columns(self) -> List[str]:
        return [c.name for c in self.manifest.schema]

    @property
    def survey_name(self) -> str:
        return self.manifest.survey_name

    # ── Partition pruning ────────────────────────────────────────────────

    def _select_partitions(
        self,
        *,
        ra_range: Optional[Range] = None,
        dec_range: Optional[Range] = None,
        z_range: Optional[Range] = None,
        t_range: Optional[Range] = None,
        healpix_cells: Optional[Iterable[int]] = None,
    ) -> List[PartitionSpec]:
        """Return partitions whose stats may overlap the given filters."""
        cell_filter = (
            {int(c) for c in healpix_cells}
            if healpix_cells is not None
            else None
        )
        keep: List[PartitionSpec] = []
        for part in self.manifest.partitions:
            if (
                cell_filter is not None
                and part.healpix_cell is not None
                and part.healpix_cell not in cell_filter
            ):
                continue
            if not _range_overlaps(ra_range, part.stats.ra_min, part.stats.ra_max):
                continue
            if not _range_overlaps(dec_range, part.stats.dec_min, part.stats.dec_max):
                continue
            if not _range_overlaps(z_range, part.stats.z_min, part.stats.z_max):
                continue
            if not _range_overlaps(t_range, part.stats.t_min, part.stats.t_max):
                continue
            keep.append(part)
        return keep

    def _build_dataset(
        self, partitions: Sequence[PartitionSpec],
    ) -> Optional[ds.Dataset]:
        """Build a :class:`pyarrow.dataset.Dataset` over *partitions*."""
        if not partitions:
            return None
        paths = [str(self.ou_dir / p.name) for p in partitions]
        return ds.dataset(paths, format="parquet")

    # ── Scanning ─────────────────────────────────────────────────────────

    def scan(
        self,
        columns: Optional[Sequence[str]] = None,
        filter: Optional[pc.Expression] = None,
        *,
        ra_range: Optional[Range] = None,
        dec_range: Optional[Range] = None,
        z_range: Optional[Range] = None,
        t_range: Optional[Range] = None,
        cone: Optional[Cone] = None,
        skypatch: Optional[SkyPatch] = None,
        healpix_cells: Optional[Iterable[int]] = None,
    ) -> pa.Table:
        """Return a :class:`pyarrow.Table` with projection + filter pushed down.

        ``ra_range`` / ``dec_range`` / ``z_range`` / ``t_range`` drive
        range-based partition pruning via manifest stats; ``t_range``
        pushdown also filters rows on ``manifest.temporal.time_column``.
        ``cone`` / ``skypatch`` / ``healpix_cells`` drive HEALPix-cell
        partition pruning on POINT datasets and add an exact
        in-cone/in-patch filter to the Parquet reader.
        """
        cells = self._resolve_cells(cone, skypatch, healpix_cells)
        partitions = self._select_partitions(
            ra_range=ra_range, dec_range=dec_range, z_range=z_range,
            t_range=t_range, healpix_cells=cells,
        )
        dataset = self._build_dataset(partitions)
        if dataset is None:
            return _empty_table(self.columns, columns)

        time_col = (
            self.manifest.temporal.time_column
            if self.manifest.temporal is not None else None
        )
        expr = _range_expr(filter, ra_range, dec_range, z_range,
                           t_range, time_col)
        expr = _spatial_expr(expr, cone, skypatch)
        cols = list(columns) if columns is not None else None
        return dataset.to_table(columns=cols, filter=expr)

    def read(
        self,
        columns: Optional[Sequence[str]] = None,
        filter: Optional[pc.Expression] = None,
        *,
        ra_range: Optional[Range] = None,
        dec_range: Optional[Range] = None,
        z_range: Optional[Range] = None,
        t_range: Optional[Range] = None,
        cone: Optional[Cone] = None,
        skypatch: Optional[SkyPatch] = None,
        healpix_cells: Optional[Iterable[int]] = None,
    ) -> pd.DataFrame:
        """Return a pandas :class:`DataFrame` — thin wrapper over :meth:`scan`."""
        table = self.scan(
            columns=columns, filter=filter,
            ra_range=ra_range, dec_range=dec_range, z_range=z_range,
            t_range=t_range, cone=cone, skypatch=skypatch,
            healpix_cells=healpix_cells,
        )
        return table.to_pandas()

    def objects_table(
        self, columns: Optional[Sequence[str]] = None,
    ) -> pa.Table:
        """Return the per-object metadata table.

        Defined for SIGHTLINE (one row per sightline) and LIGHTCURVE
        (one row per source) geometries. Raises on POINT/HEALPIX, which
        have no objects table on disk.
        """
        import pyarrow.parquet as pq

        if self.geometry not in {DataGeometry.SIGHTLINE, DataGeometry.LIGHTCURVE}:
            raise ValueError(
                f"objects_table() is only defined for SIGHTLINE and "
                f"LIGHTCURVE geometries, got {self.geometry.value!r}"
            )
        return pq.read_table(
            self.ou_dir / OBJECTS_FILENAME,
            columns=list(columns) if columns else None,
        )

    # ── Internals ────────────────────────────────────────────────────────

    def _resolve_cells(
        self,
        cone: Optional[Cone],
        skypatch: Optional[SkyPatch],
        healpix_cells: Optional[Iterable[int]],
    ) -> Optional[List[int]]:
        """Union caller-supplied cells with cells derived from ``cone`` /
        ``skypatch`` at the manifest's HEALPix pruning NSIDE.

        Returns ``None`` if no cell filtering applies.
        """
        if cone is None and skypatch is None and healpix_cells is None:
            return None
        nside = HEALPIX_PARTITION_NSIDE
        nest = HEALPIX_PARTITION_NEST
        acc: set = set()
        if healpix_cells is not None:
            acc.update(int(c) for c in healpix_cells)
        if cone is not None:
            acc.update(int(c) for c in cone.healpix_cells(nside, nest=nest))
        if skypatch is not None:
            acc.update(int(c) for c in skypatch.healpix_cells(nside, nest=nest))
        return sorted(acc)


# ── Helpers ──────────────────────────────────────────────────────────────


def _range_overlaps(
    query: Optional[Range], part_lo: Optional[float], part_hi: Optional[float],
) -> bool:
    """Return True if the caller's range could overlap the partition range.

    When the partition has no stats for the column (``part_lo`` /
    ``part_hi`` are ``None``), we must keep the partition.
    """
    if query is None or (part_lo is None and part_hi is None):
        return True
    qlo, qhi = query
    if qhi is not None and part_lo is not None and qhi < part_lo:
        return False
    if qlo is not None and part_hi is not None and qlo > part_hi:
        return False
    return True


def _range_expr(
    base: Optional[pc.Expression],
    ra_range: Optional[Range],
    dec_range: Optional[Range],
    z_range: Optional[Range],
    t_range: Optional[Range] = None,
    time_column: Optional[str] = None,
) -> Optional[pc.Expression]:
    """Combine a user-supplied expression with ra/dec/z/time range bounds."""
    parts: List[pc.Expression] = []
    if base is not None:
        parts.append(base)
    cols: List[Tuple[str, Optional[Range]]] = [
        ("ra", ra_range), ("dec", dec_range), ("z", z_range),
    ]
    if t_range is not None and time_column is not None:
        cols.append((time_column, t_range))
    for col, rng in cols:
        if rng is None:
            continue
        lo, hi = rng
        if lo is not None:
            parts.append(pc.field(col) >= lo)
        if hi is not None:
            parts.append(pc.field(col) <= hi)
    if not parts:
        return None
    expr = parts[0]
    for p in parts[1:]:
        expr = expr & p
    return expr


def _spatial_expr(
    base: Optional[pc.Expression],
    cone: Optional[Cone],
    skypatch: Optional[SkyPatch],
) -> Optional[pc.Expression]:
    """Add an exact ra/dec filter for a cone or skypatch to *base*.

    Cells are a cover, not an exact mask: rows inside a boundary cell
    but outside the shape must still be dropped. Delegate that to the
    Parquet reader via a pushdown expression.
    """
    parts: List[pc.Expression] = []
    if base is not None:
        parts.append(base)
    if cone is not None:
        parts.append(_cone_expr(cone))
    if skypatch is not None:
        parts.append(_skypatch_expr(skypatch))
    if not parts:
        return None
    expr = parts[0]
    for p in parts[1:]:
        expr = expr & p
    return expr


def _cone_expr(cone: Cone) -> pc.Expression:
    """Exact great-circle-distance <= radius pushdown expression."""
    ra_c = np.radians(cone.ra)
    dec_c = np.radians(cone.dec)
    cos_r = float(np.cos(np.radians(cone.radius)))
    ra = pc.field("ra") * (np.pi / 180.0)
    dec = pc.field("dec") * (np.pi / 180.0)
    # cos(dist) = sin(dec1) sin(dec2) + cos(dec1) cos(dec2) cos(ra1-ra2)
    cos_dist = (
        pc.sin(dec) * float(np.sin(dec_c))
        + pc.cos(dec) * float(np.cos(dec_c)) * pc.cos(ra - ra_c)
    )
    return cos_dist >= cos_r


def _skypatch_expr(s: SkyPatch) -> pc.Expression:
    dec_ok = (pc.field("dec") >= s.dec_min) & (pc.field("dec") <= s.dec_max)
    if s.ra_min <= s.ra_max:
        ra_ok = (pc.field("ra") >= s.ra_min) & (pc.field("ra") <= s.ra_max)
    else:
        ra_ok = (pc.field("ra") >= s.ra_min) | (pc.field("ra") <= s.ra_max)
    return dec_ok & ra_ok


def _empty_table(
    all_columns: Iterable[str], requested: Optional[Sequence[str]],
) -> pa.Table:
    """Build an empty table with the right schema."""
    cols = list(requested) if requested is not None else list(all_columns)
    return pa.table({c: pa.array([], type=pa.float64()) for c in cols})
