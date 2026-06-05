"""Map × catalog cross-correlation MeasurementSet. Cosmology-free (C_ℓ is P2)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from oneuniverse.combine.weights import ColumnWeight
from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.measure.clustering import build_galaxy_clustering
from oneuniverse.measure.dataproduct import FieldMap
from oneuniverse.measure.measurement_set import MeasurementSet
from oneuniverse.measure.spec import MeasurementSpec


def build_map_cross(catalog_view: DatasetView, fieldmap: FieldMap, *,
                    gal_tracer: str = "gal", map_tracer: str = "kappa",
                    z_range: Tuple[float, float] = (0.0, 2.0),
                    gal_weights_columns: Tuple[str, ...] = ("weight_comp",),
                    nz_edges: Optional[np.ndarray] = None,
                    nside_region: int = 8) -> MeasurementSet:
    """Galaxy PointSet × FieldMap → cross-correlation MeasurementSet."""
    if nz_edges is None:
        nz_edges = np.linspace(0.0, 2.0, 21)
    gal_ms = build_galaxy_clustering(
        catalog_view, tracer=gal_tracer, z_range=z_range,
        weights=[ColumnWeight(c) for c in gal_weights_columns],
        nz_edges=nz_edges, randoms="none", nside_region=nside_region)
    gal_ps = gal_ms.products[gal_tracer]
    spec = MeasurementSpec(tracers=(gal_tracer, map_tracer),
                           pairs=((gal_tracer, map_tracer),),
                           statistic="cl", estimator_family="cross")
    return MeasurementSet(products={gal_tracer: gal_ps, map_tracer: fieldmap},
                          spec=spec, metadata=gal_ms.metadata)
