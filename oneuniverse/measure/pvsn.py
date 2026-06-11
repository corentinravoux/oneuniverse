"""PV + SN connections. Cosmology-free (μ->distance, Hubble fit are P2)."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.measure.covariance import CovarianceHandle
from oneuniverse.measure._pipeline import prepare_pointset
from oneuniverse.measure.dataproduct import PointSet
from oneuniverse.measure.distances import attach_distances
from oneuniverse.measure.measurement_set import MeasurementSet
from oneuniverse.measure.metadata import ProductMetadata, Provenance
from oneuniverse.measure.spec import MeasurementSpec


def _base(view, z_range, distance_columns, nside_window, nside_region):
    # S5: the shared spine; PV/SN adds only the distance-atom validation.
    cat, win, region, meta, _ = prepare_pointset(
        view, z_range=z_range, weights=None, nside_window=nside_window,
        nside_region=nside_region)
    cat, dcols = attach_distances(cat, columns=distance_columns)
    return cat, win, region, meta, dcols


def build_peculiar_velocity(view: DatasetView, *, tracer: str = "pv",
                            z_range: Tuple[float, float] = (0.0, 0.1),
                            distance_columns: Sequence[str] = (
                                "mu", "mu_err", "v_pec", "sigma_v"),
                            nside_window: int = 128, nside_region: int = 8
                            ) -> MeasurementSet:
    cat, win, region, meta, dcols = _base(
        view, z_range, distance_columns, nside_window, nside_region)
    prov = Provenance(dataset_ids=(view.survey_name,),
                      extra={"distance_columns": tuple(dcols)})
    ps = PointSet(catalog=cat, randoms=None, nz=None, window=win,
                  region_map=region, metadata=meta, provenance=prov)
    spec = MeasurementSpec(tracers=(tracer,), pairs=((tracer, tracer),),
                           statistic="velocity_correlation",
                           estimator_family="velocity")
    return MeasurementSet(products={tracer: ps}, spec=spec, metadata=meta)


def build_sn_hubble(view: DatasetView, *, tracer: str = "sn",
                    z_range: Tuple[float, float] = (0.0, 1.5),
                    distance_columns: Sequence[str] = ("mu", "mu_err"),
                    covariance: Optional[CovarianceHandle] = None,
                    nside_window: int = 64, nside_region: int = 8
                    ) -> MeasurementSet:
    cat, win, region, meta, dcols = _base(
        view, z_range, distance_columns, nside_window, nside_region)
    extra = {"distance_columns": tuple(dcols)}
    if covariance is not None:
        extra["cov_id"] = covariance.cov_id
    prov = Provenance(dataset_ids=(view.survey_name,), extra=extra)
    ps = PointSet(catalog=cat, randoms=None, nz=None, window=win,
                  region_map=region, metadata=meta, provenance=prov)
    spec = MeasurementSpec(tracers=(tracer,), pairs=((tracer, tracer),),
                           statistic="hubble", estimator_family="sn")
    return MeasurementSet(products={tracer: ps}, spec=spec, metadata=meta)
