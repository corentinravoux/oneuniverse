"""
oneuniverse.weight
~~~~~~~~~~~~~~~~~~
Cross-survey weighting and concurrence resolution for the oneuniverse
package.

Quick start
-----------
>>> from oneuniverse.weight import (
...     WeightedCatalog, InverseVarianceWeight, ConstantWeight,
... )
>>> wc = WeightedCatalog({"eboss": eboss_df, "desi": desi_df})
>>> wc.add_weight("eboss", InverseVarianceWeight("z_spec_err"))
>>> wc.add_weight("desi",  InverseVarianceWeight("z_spec_err"))
>>> wc.crossmatch(sky_tol_arcsec=1.0, dz_tol=1e-3)
>>> combined = wc.combine(
...     value_col="z", variance_col="z_var",
...     strategy="hyperparameter",
...     survey_alpha={"desi": 2.0, "eboss": 1.0},
... )
>>> wc.concurrences(universal_id=42)
"""

from oneuniverse.weight.base import (  # noqa: F401
    ColumnWeight,
    ConstantWeight,
    FKPWeight,
    InverseVarianceWeight,
    ProductWeight,
    QualityMaskWeight,
    Weight,
)
from oneuniverse.weight.catalog import WeightedCatalog  # noqa: F401
from oneuniverse.weight.combine import (  # noqa: F401
    CombinedMeasurements,
    combine_weights,
)
from oneuniverse.weight.crossmatch import (  # noqa: F401
    CrossMatchResult,
    cross_match_surveys,
)

__all__ = [
    "Weight",
    "ProductWeight",
    "ConstantWeight",
    "ColumnWeight",
    "InverseVarianceWeight",
    "FKPWeight",
    "QualityMaskWeight",
    "cross_match_surveys",
    "CrossMatchResult",
    "combine_weights",
    "CombinedMeasurements",
    "WeightedCatalog",
]
