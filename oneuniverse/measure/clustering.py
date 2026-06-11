"""build_galaxy_clustering — the galaxy-clustering P1->P2 connection (9 steps)."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

from oneuniverse.combine.weights import Weight
from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.measure._pipeline import prepare_pointset
from oneuniverse.measure.dataproduct import PointSet
from oneuniverse.measure.measurement_set import MeasurementSet
from oneuniverse.measure.metadata import Provenance
from oneuniverse.measure.nz import nz_from_spec_z
from oneuniverse.measure.randoms import generate_randoms, randoms_from_view
from oneuniverse.measure.regions import assign_regions
from oneuniverse.measure.spec import MeasurementSpec


def build_galaxy_clustering(
    view: DatasetView, *, tracer: str = "gal",
    z_range: Tuple[float, float],
    weights: Sequence[Weight],
    nz_edges,
    nside_window: int = 256,
    nside_region: int = 8,
    quality_column: Optional[str] = "quality", quality_min: float = 1.0,
    randoms: Union[str, DatasetView] = "generate",
    n_randoms: int = 0, seed: int = 0,
    statistic: str = "pk_multipole",
) -> MeasurementSet:
    """OUF POINT view -> galaxy-clustering MeasurementSet (cosmology-free)."""
    # 1-3, 5, 8: the shared spine (select+clean -> weights -> window -> region)
    cat, win, region, meta, recipe = prepare_pointset(
        view, z_range=z_range, weights=weights, nside_window=nside_window,
        nside_region=nside_region, quality_column=quality_column,
        quality_min=quality_min)
    # 6 n(z) (weighted)
    nz = nz_from_spec_z(cat["z"].to_numpy(), edges=nz_edges,
                        weights=cat["weight"].to_numpy())
    # 4 randoms (ingest | generate | none) — explicit, no silent fall-through
    if isinstance(randoms, DatasetView):
        rnd, source = randoms_from_view(randoms)
        # B4: the data catalog is z-cut above; ingested randoms must match the
        # same radial window, and carry a weight column like generated ones.
        if "z" in rnd.columns:
            rnd = rnd[(rnd["z"] >= z_range[0])
                      & (rnd["z"] <= z_range[1])].reset_index(drop=True)
        if "weight" not in rnd.columns:
            rnd = rnd.copy()
            rnd["weight"] = 1.0
    elif randoms == "generate":
        rnd, source = generate_randoms(win, nz, n_randoms=n_randoms, seed=seed)
    elif randoms == "none":
        rnd, source = None, None
    else:
        raise ValueError(
            f"build_galaxy_clustering: randoms must be a DatasetView (ingest), "
            f"'generate', or 'none'; got {randoms!r}")
    if rnd is not None:
        rnd = rnd.copy()
        rnd["region_id"] = assign_regions(rnd["ra"].to_numpy(),
                                          rnd["dec"].to_numpy(),
                                          nside=nside_region)
    prov = Provenance(dataset_ids=(view.survey_name,), weight_recipe=recipe,
                      randoms_source=source, nz_method=nz.method)
    ps = PointSet(catalog=cat, randoms=rnd, nz=nz, window=win,
                  region_map=region, metadata=meta, provenance=prov)
    spec = MeasurementSpec(tracers=(tracer,), pairs=((tracer, tracer),),
                           statistic=statistic, estimator_family="clustering")
    return MeasurementSet(products={tracer: ps}, spec=spec, metadata=meta)
