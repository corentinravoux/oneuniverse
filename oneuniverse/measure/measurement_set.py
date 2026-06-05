"""The joint-analysis bundle handed to Pillar 2. Cosmology-free."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from oneuniverse.measure.dataproduct import DataProduct
from oneuniverse.measure.metadata import ProductMetadata
from oneuniverse.measure.spec import MeasurementSpec


@dataclass
class MeasurementSet:
    products: Dict[str, DataProduct]
    spec: MeasurementSpec
    metadata: ProductMetadata

    def summary(self) -> Dict[str, Any]:
        """A structured, cosmology-free description of the set + its products.

        For downstream inspection: which atoms each product carries, the
        measurement spec, and the shared region scheme. Carries no cosmology.
        """
        prods: Dict[str, Any] = {}
        for name, p in self.products.items():
            entry: Dict[str, Any] = {
                "kind": p.kind,
                "dataset_ids": list(p.provenance.dataset_ids),
                "randoms_source": p.provenance.randoms_source,
                "links": [lk.role for lk in (p.links or [])],
                "has_covariance": p.covariance is not None,
            }
            if p.kind == "pointset":
                cat = p.catalog
                win = p.window
                entry.update(
                    n=int(len(cat)) if cat is not None else 0,
                    columns=list(cat.columns) if cat is not None else [],
                    has_randoms=p.randoms is not None,
                    has_nz=p.nz is not None,
                    has_window=win is not None,
                    has_photoz=p.photoz is not None,
                    has_named_weights=p.weights is not None,
                    has_dndz_external=p.dndz_external is not None,
                    n_tomo=(len(p.nz) if isinstance(p.nz, dict) else None),
                    window_systematics=(list(win.systematics)
                                        if win is not None and win.systematics
                                        else []),
                )
            elif p.kind == "sightline":
                entry.update(n_sightlines=int(p.n_sightlines),
                             has_continuum=p.continuum is not None)
            elif p.kind == "fieldmap":
                entry.update(nside=int(p.nside), npix=int(p.npix),
                             covered_pixels=int(p.mask.sum()),
                             has_beam=p.beam is not None,
                             has_interloper=p.interloper is not None,
                             has_distance_extras=p.distance_extras is not None,
                             is_cube=p.axes is not None)
            prods[name] = entry
        ps = self.spec.pair_statistics
        return {
            "n_products": len(self.products),
            "spec": {"statistic": self.spec.statistic,
                     "estimator_family": self.spec.estimator_family,
                     "pairs": [list(pair) for pair in self.spec.pairs],
                     # JSON-safe: tuple pair keys -> "a×b" strings
                     "pair_statistics": (
                         {f"{a}×{b}": v for (a, b), v in ps.items()}
                         if ps else None)},
            "region_nside": self.metadata.nside_region,
            "frame": self.metadata.frame,
            "cosmology_free": True,
            "products": prods,
        }

    def __repr__(self) -> str:
        kinds = ", ".join(f"{n}:{p.kind}" for n, p in self.products.items())
        return (f"MeasurementSet({self.spec.estimator_family}/"
                f"{self.spec.statistic}; {kinds})")

    # -- on-disk handoff form (the P1->P2 boundary across processes) -------
    def to_dir(self, path):
        """Persist to a self-describing directory (parquet + npy + JSON)."""
        from oneuniverse.measure.io import save_measurement_set
        return save_measurement_set(self, path)

    @classmethod
    def from_dir(cls, path) -> "MeasurementSet":
        """Reconstruct a MeasurementSet written by :meth:`to_dir`."""
        from oneuniverse.measure.io import load_measurement_set
        return load_measurement_set(path)

    #: catalog column names that imply a cosmology was applied (z->distance).
    #: Their presence violates the cosmology-free contract.
    _FORBIDDEN_COLUMNS = frozenset({
        "comoving_distance", "r_comoving", "comoving_dist", "d_comoving",
        "dist_mpc_h", "chi", "chi_mpc", "distance_mpc", "luminosity_distance",
        "angular_diameter_distance",
    })

    def check_invariants(self, *, _inject_cosmology: bool = False) -> None:
        if _inject_cosmology or hasattr(self.metadata, "cosmology"):
            raise ValueError(
                "MeasurementSet must be cosmology-free (no cosmology in "
                "metadata); cosmology enters at the Pillar-2 estimator call")
        nside = self.metadata.nside_region
        for name, p in self.products.items():
            n = len(p.region_map)
            catalog = getattr(p, "catalog", None)
            if catalog is not None:
                # the load-bearing rule, enforced on *contents*: no
                # cosmology-derived column may have leaked into the catalog.
                leaked = self._FORBIDDEN_COLUMNS & {c.lower()
                                                    for c in catalog.columns}
                if leaked:
                    raise ValueError(
                        f"product {name!r}: cosmology-derived column(s) "
                        f"{sorted(leaked)} present — the MeasurementSet must be "
                        f"cosmology-free (z->distance happens in Pillar 2)")
            if catalog is not None and len(catalog) != n:
                raise ValueError(
                    f"product {name!r}: region_map length {n} != catalog "
                    f"length {len(catalog)}")
            # Sky maps (FieldMap) are not per-object jackknifed: empty
            # region_map exempts them from the shared-NSIDE invariant.
            if n > 0 and p.metadata.nside_region != nside:
                raise ValueError(
                    f"product {name!r}: region NSIDE {p.metadata.nside_region}"
                    f" != set NSIDE {nside} (shared region_map invariant)")
