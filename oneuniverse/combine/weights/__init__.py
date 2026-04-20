"""
oneuniverse.combine.weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Composable per-object weight primitives used by
:class:`oneuniverse.combine.catalog.WeightedCatalog`.
"""
from oneuniverse.combine.weights.base import ProductWeight, Weight
from oneuniverse.combine.weights.fkp import FKPWeight
from oneuniverse.combine.weights.ivar import InverseVarianceWeight
from oneuniverse.combine.weights.quality import (
    ColumnWeight,
    ConstantWeight,
    QualityMaskWeight,
)
from oneuniverse.combine.weights.registry import default_weight_for

__all__ = [
    "Weight",
    "ProductWeight",
    "ConstantWeight",
    "ColumnWeight",
    "InverseVarianceWeight",
    "FKPWeight",
    "QualityMaskWeight",
    "default_weight_for",
]
