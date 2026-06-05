"""Ingest a HEALPix field (CMBκ / tSZ y / HI) into a measure FieldMap."""
from __future__ import annotations

from typing import Optional

import numpy as np

from oneuniverse.measure.dataproduct import FieldMap
from oneuniverse.measure.metadata import ProductMetadata, Provenance


def fieldmap_from_healpix(values, *, mask: Optional[np.ndarray] = None,
                          nside: int, nest: bool = True,
                          frame: str = "galactic",
                          dataset_id: str = "map") -> FieldMap:
    """Wrap a HEALPix field + mask as a measure FieldMap (cosmology-free)."""
    values = np.asarray(values, float)
    if mask is None:
        mask = np.ones(values.shape, dtype=bool)
    mask = np.asarray(mask, bool)
    meta = ProductMetadata(frame=frame, epoch=2000.0,
                           length_unit="dimensionless", nside_region=0)
    return FieldMap(values=values, mask=mask, nside=int(nside), nest=nest,
                    region_map=np.array([], dtype=np.int64), metadata=meta,
                    provenance=Provenance(dataset_ids=(dataset_id,)))
