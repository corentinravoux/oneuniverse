"""
oneuniverse.weight
~~~~~~~~~~~~~~~~~~
.. deprecated:: Phase 6
    Moved to :mod:`oneuniverse.combine`. This module re-exports the new
    API and emits :class:`DeprecationWarning` on import.
"""
import warnings

warnings.warn(
    "`oneuniverse.weight` is deprecated, use `oneuniverse.combine` instead. "
    "This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from oneuniverse.combine import (  # noqa: F401,E402
    ColumnWeight,
    CombinedMeasurements,
    ConstantWeight,
    FKPWeight,
    InverseVarianceWeight,
    ProductWeight,
    QualityMaskWeight,
    Weight,
    WeightedCatalog,
    combine_weights,
    default_weight_for,
)
from oneuniverse.data.oneuid_crossmatch import (  # noqa: F401,E402
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
    "default_weight_for",
    "cross_match_surveys",
    "CrossMatchResult",
    "combine_weights",
    "CombinedMeasurements",
    "WeightedCatalog",
]
