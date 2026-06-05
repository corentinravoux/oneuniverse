"""Lyα forest connection. Cosmology-free (r_∥/r_⊥ + P(k) are P2)."""
from __future__ import annotations

from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.measure.measurement_set import MeasurementSet
from oneuniverse.measure.sightline import sightline_from_view
from oneuniverse.measure.spec import MeasurementSpec


def build_lya(view: DatasetView, *, tracer: str = "lya",
              statistic: str = "p1d", nside_region: int = 16) -> MeasurementSet:
    """SIGHTLINE view -> Lyα MeasurementSet (Sightline product)."""
    sl = sightline_from_view(view, nside_region=nside_region)
    spec = MeasurementSpec(tracers=(tracer,), pairs=((tracer, tracer),),
                           statistic=statistic, estimator_family="lya")
    return MeasurementSet(products={tracer: sl}, spec=spec,
                          metadata=sl.metadata)
