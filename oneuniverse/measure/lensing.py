"""Weak-lensing connections: cosmic shear + 3x2pt. Cosmology-free."""
from __future__ import annotations

from typing import Tuple

from oneuniverse.data.dataset_view import DatasetView
from oneuniverse.measure.dataproduct import PointSet
from oneuniverse.measure.measurement_set import MeasurementSet
from oneuniverse.measure.metadata import ProductMetadata, Provenance
from oneuniverse.measure.photoz import attach_photoz
from oneuniverse.measure.regions import assign_regions
from oneuniverse.measure.select import select_clean
from oneuniverse.measure.shapes import attach_shear
from oneuniverse.measure.spec import MeasurementSpec
from oneuniverse.measure.tomography import tomographic_nz
from oneuniverse.measure.window import footprint_from_positions


def build_cosmic_shear(view: DatasetView, *, tracer: str = "src",
                       kind: str = "metacal", tomo_column: str = "tomo_bin",
                       z_grid, nside_window: int = 256, nside_region: int = 8,
                       statistic: str = "xi_pm") -> MeasurementSet:
    """Source shape catalog -> cosmic-shear MeasurementSet (cosmology-free)."""
    cat = select_clean(view)                                  # 1-2
    cat, srecipe = attach_shear(cat, kind=kind)               # 3 (shear weight)
    kernel = attach_photoz(view)                              # 7 photo-z kernel
    nzs = tomographic_nz(cat, kernel, bin_column=tomo_column, # 6 per-bin n(z)
                         z_grid=z_grid)
    win = footprint_from_positions(cat["ra"].to_numpy(),      # 5
                                   cat["dec"].to_numpy(), nside=nside_window)
    region = assign_regions(cat["ra"].to_numpy(),             # 8
                            cat["dec"].to_numpy(), nside=nside_region)
    cat = cat.copy(); cat["region_id"] = region
    meta = ProductMetadata(frame="icrs", epoch=2000.0, length_unit="deg",
                           nside_region=int(nside_region))
    prov = Provenance(dataset_ids=(view.survey_name,),
                      weight_recipe=(srecipe,), nz_method="photo_stack")
    ps = PointSet(catalog=cat, randoms=None, nz=nzs, window=win,
                  region_map=region, metadata=meta, provenance=prov,
                  photoz=kernel, tomo_bin=cat[tomo_column].to_numpy())
    spec = MeasurementSpec(tracers=(tracer,), pairs=((tracer, tracer),),
                           statistic=statistic, estimator_family="lensing")
    return MeasurementSet(products={tracer: ps}, spec=spec, metadata=meta)


def build_3x2pt(lens_view: DatasetView, source_view: DatasetView, *, z_grid,
                nside_region: int = 8, nside_window: int = 256,
                lens_z_range: Tuple[float, float] = (0.0, 2.0),
                lens_weights_columns: Tuple[str, ...] = ("weight_comp",),
                kind: str = "metacal", tomo_column: str = "tomo_bin"
                ) -> MeasurementSet:
    """Lens clustering + source shear sharing one region map (3x2pt bundle)."""
    from oneuniverse.combine.weights import ColumnWeight
    from oneuniverse.measure.weighting import assemble_weight
    lcat = select_clean(lens_view, z_range=lens_z_range)
    lcat, lrec = assemble_weight(
        lcat, [ColumnWeight(c) for c in lens_weights_columns])
    lreg = assign_regions(lcat["ra"].to_numpy(), lcat["dec"].to_numpy(),
                          nside=nside_region)
    lcat = lcat.copy(); lcat["region_id"] = lreg
    meta = ProductMetadata(frame="icrs", epoch=2000.0, length_unit="deg",
                           nside_region=int(nside_region))
    lens_ps = PointSet(
        catalog=lcat, randoms=None, nz=None,
        window=footprint_from_positions(lcat["ra"].to_numpy(),
                                        lcat["dec"].to_numpy(),
                                        nside=nside_window),
        region_map=lreg, metadata=meta,
        provenance=Provenance(dataset_ids=(lens_view.survey_name,),
                              weight_recipe=lrec))
    src_ms = build_cosmic_shear(source_view, tracer="src", kind=kind,
                                tomo_column=tomo_column, z_grid=z_grid,
                                nside_window=nside_window,
                                nside_region=nside_region)
    src_ps = src_ms.products["src"]
    spec = MeasurementSpec(
        tracers=("lens", "src"),
        pairs=(("lens", "lens"), ("lens", "src"), ("src", "src")),
        statistic="mixed", estimator_family="lensing",
        pair_statistics={("lens", "lens"): "w_theta",
                         ("lens", "src"): "gamma_t",
                         ("src", "src"): "xi_pm"})
    return MeasurementSet(products={"lens": lens_ps, "src": src_ps},
                          spec=spec, metadata=meta)
