"""Universal DataProduct container — general enough for every probe.

Three geometry subtypes (PointSet / Sightline / FieldMap) carry, as optional
slots, the full atom inventory from the P1->P2 measurement-requirements
research: positions + redshift (spec / photo-z kernel / tomographic / external
dndz / z-absent), named weight families (incl. PIP), shapes + calibration,
distance indicators, light curves, mass proxies, sub-object hierarchies
(cluster members / lens systems / DLAs), windows + depth/systematics maps,
randoms, fields/cubes (+ beam / interloper / GW distance extras), and a
covariance plan. **No cosmology anywhere** — that enters at the Pillar-2 call.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import ClassVar, List, Optional

import numpy as np
import pandas as pd

from oneuniverse.measure.metadata import ProductMetadata, Provenance


@dataclass(kw_only=True)
class DataProduct(abc.ABC):
    region_map: np.ndarray
    metadata: ProductMetadata
    provenance: Provenance
    links: Optional[List[object]] = None      # SubObjectLinks (hierarchies)
    covariance: object = None                 # CovariancePlan | None

    kind: ClassVar[str] = "abstract"


@dataclass(kw_only=True)
class PointSet(DataProduct):
    catalog: pd.DataFrame = None
    randoms: Optional[pd.DataFrame] = None
    nz: object = None                 # Nz | dict[int, Nz] | None
    window: object = None             # Window | None
    photoz: object = None             # ProbabilisticRedshift | None (WL/photo-z)
    tomo_bin: Optional[np.ndarray] = None
    weights: object = None            # NamedWeights | None (kept components)
    dndz_external: object = None      # Nz | None (z-absent tracers: radio cont.)
    attributes: Optional[dict] = None  # role -> column list (shapes/distances/...)

    kind: ClassVar[str] = "pointset"


@dataclass(kw_only=True)
class Sightline(DataProduct):
    los: pd.DataFrame = None          # sightline_id, ra, dec, z_source, region_id
    delta: object = None              # list of per-LOS δ_F(λ)
    mask: object = None               # list of per-LOS weights
    continuum: object = None
    resolution: object = None
    weights: object = None            # NamedWeights | None

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
    axes: object = None               # WCS/axis metadata for cubes (LIM/HI)
    beam: object = None               # beam FWHM / window (LIM, CMB)
    spectral_response: object = None  # per-channel response (LIM/IM)
    interloper: object = None         # interloper-line model handle (LIM)
    distance_extras: object = None    # DISTMU/SIGMA/NORM per pixel (GW skymap)

    kind: ClassVar[str] = "fieldmap"

    @property
    def npix(self) -> int:
        return int(self.values.shape[0])
