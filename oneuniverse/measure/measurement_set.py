"""The joint-analysis bundle handed to Pillar 2. Cosmology-free."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from oneuniverse.measure.dataproduct import DataProduct
from oneuniverse.measure.metadata import ProductMetadata
from oneuniverse.measure.spec import MeasurementSpec


@dataclass
class MeasurementSet:
    products: Dict[str, DataProduct]
    spec: MeasurementSpec
    metadata: ProductMetadata

    def check_invariants(self, *, _inject_cosmology: bool = False) -> None:
        if _inject_cosmology or hasattr(self.metadata, "cosmology"):
            raise ValueError(
                "MeasurementSet must be cosmology-free (no cosmology in "
                "metadata); cosmology enters at the Pillar-2 estimator call")
        nside = self.metadata.nside_region
        for name, p in self.products.items():
            n = len(p.region_map)
            catalog = getattr(p, "catalog", None)
            if catalog is not None and len(catalog) != n:
                raise ValueError(
                    f"product {name!r}: region_map length {n} != catalog "
                    f"length {len(catalog)}")
            if p.metadata.nside_region != nside:
                raise ValueError(
                    f"product {name!r}: region NSIDE {p.metadata.nside_region}"
                    f" != set NSIDE {nside} (shared region_map invariant)")
