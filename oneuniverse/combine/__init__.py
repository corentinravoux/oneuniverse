"""
oneuniverse.combine
~~~~~~~~~~~~~~~~~~~
Cross-survey weighting and measurement combination.

Quick start
-----------
>>> from oneuniverse.combine import WeightedCatalog, default_weight_for
>>> wc = WeightedCatalog.from_oneuid(index, db)
>>> wc.add_weight("eboss", default_weight_for("spectroscopic", "spec"))
>>> combined = wc.combine(value_col="z", variance_col="z_var")
"""
from oneuniverse.combine.catalog import WeightedCatalog
from oneuniverse.combine.measurements import CombinedMeasurements
from oneuniverse.combine.strategies import combine_weights
from oneuniverse.combine.weights import (
    ColumnWeight,
    ConstantWeight,
    FKPWeight,
    InverseVarianceWeight,
    ProductWeight,
    QualityMaskWeight,
    Weight,
    default_weight_for,
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
    "combine_weights",
    "CombinedMeasurements",
    "WeightedCatalog",
]
