"""Step 9: the aimed-measurement declaration. Cosmology deferred to P2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Union

if TYPE_CHECKING:  # annotation only — no runtime import
    from oneuniverse.measure.covariance import CovariancePlan


@dataclass(frozen=True)
class MeasurementSpec:
    tracers: Tuple[str, ...]
    pairs: Tuple[Tuple[str, str], ...]
    statistic: str                    # "pk_multipole" | "xi_smu" | "w_theta" | ...
    estimator_family: str             # "clustering" | "field_level" | ...
    binning: Optional[dict] = None
    coords: str = "on_sky"            # comoving conversion happens in P2
    covariance: Union[str, "CovariancePlan"] = "jackknife"
    pair_statistics: Optional[dict] = None   # per-pair statistic (3x2pt etc.)
