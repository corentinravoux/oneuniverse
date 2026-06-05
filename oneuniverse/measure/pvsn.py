"""PV + SN connections. Cosmology-free (μ->distance, Hubble fit are P2)."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.measure.covariance import CovarianceHandle
from oneuniverse.measure.dataproduct import PointSet
from oneuniverse.measure.distances import attach_distances
from oneuniverse.measure.measurement_set import MeasurementSet
from oneuniverse.measure.metadata import ProductMetadata, Provenance
from oneuniverse.measure.regions import assign_regions
from oneuniverse.measure.select import select_clean
from oneuniverse.measure.spec import MeasurementSpec
from oneuniverse.measure.window import footprint_from_positions


def _base(view, z_range, distance_columns, nside_window, nside_region):
    cat = select_clean(view, z_range=z_range)
    cat, dcols = attach_distances(cat, columns=distance_columns)
    win = footprint_from_positions(cat["ra"].to_numpy(),
                                   cat["dec"].to_numpy(), nside=nside_window)
    region = assign_regions(cat["ra"].to_numpy(), cat["dec"].to_numpy(),
                            nside=nside_region)
    cat = cat.copy(); cat["region_id"] = region
    meta = ProductMetadata(frame="icrs", epoch=2000.0, length_unit="deg",
                           nside_region=int(nside_region))
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
