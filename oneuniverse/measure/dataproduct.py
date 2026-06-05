"""DataProduct ABC + PointSet (galaxy clustering carrier). Cosmology-free."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np
import pandas as pd

from oneuniverse.measure.metadata import ProductMetadata, Provenance


@dataclass
class DataProduct(abc.ABC):
    region_map: np.ndarray
    metadata: ProductMetadata
    provenance: Provenance

    kind: ClassVar[str] = "abstract"


@dataclass(kw_only=True)
class PointSet(DataProduct):
    catalog: pd.DataFrame = None
    randoms: Optional[pd.DataFrame] = None
    nz: object = None                 # Nz | dict[int, Nz] | None
    window: object = None             # Window | None
    photoz: object = None             # ProbabilisticRedshift | None (WL)
    tomo_bin: Optional[np.ndarray] = None

    kind: ClassVar[str] = "pointset"


@dataclass(kw_only=True)
class Sightline(DataProduct):
    los: pd.DataFrame = None          # sightline_id, ra, dec, z_source, region_id
    delta: object = None              # list of per-LOS δ_F(λ)
    mask: object = None               # list of per-LOS weights
    continuum: object = None
    resolution: object = None

    kind: ClassVar[str] = "sightline"

    @property
    def n_sightlines(self) -> int:
        return len(self.los)


@dataclass(kw_only=True)
class FieldMap(DataProduct):
    values: np.ndarray = None         # HEALPix vector (or flattened voxel grid)
    mask: np.ndarray = None
    nside: int = 0
    nest: bool = True
    axes: object = None               # WCS/axis metadata for cubes (optional)

    kind: ClassVar[str] = "fieldmap"

    @property
    def npix(self) -> int:
        return int(self.values.shape[0])
