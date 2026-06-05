"""Observational metadata + provenance for measure DataProducts. NO cosmology."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class ProductMetadata:
    frame: str                       # icrs / galactic / ecliptic
    epoch: float                     # e.g. 2000.0, 2016.0
    length_unit: str                 # "deg" on-sky; comoving conversion is P2
    nside_region: int                # HEALPix NSIDE of the region_map


@dataclass(frozen=True)
class Provenance:
    dataset_ids: Tuple[str, ...]
    weight_recipe: Tuple[str, ...] = ()
    randoms_source: Optional[str] = None      # "ingested" | "generated" | None
    nz_method: Optional[str] = None           # "spec_hist" | ...
    extra: dict = field(default_factory=dict)
