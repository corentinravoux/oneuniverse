"""
oneuniverse.combine.weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Composable per-object weight primitives used by
:class:`oneuniverse.combine.catalog.WeightedCatalog`.
"""
from oneuniverse.combine.weights.base import ProductWeight, Weight
from oneuniverse.combine.weights.fkp import FKPWeight
from oneuniverse.combine.weights.hpmap import HealpixMapWeight
from oneuniverse.combine.weights.ivar import InverseVarianceWeight
from oneuniverse.combine.weights.pdf import (
    PdfMeanRedshiftWeight,
    PdfWidthIVarWeight,
)
from oneuniverse.combine.weights.pip import PipBitweightWeight
from oneuniverse.combine.weights.quality import (
    ColumnWeight,
    ConstantWeight,
    QualityMaskWeight,
)
from oneuniverse.combine.weights.registry import (
    default_weight_for, register_default, unregister_default,
)
from oneuniverse.combine.weights.shear import ShearWeight
from oneuniverse.combine.weights.selection import (
    CompletenessWeight,
    FiberCollisionWeight,
    ZFailureWeight,
    boss_total_weight,
)

__all__ = [
    "Weight",
    "ProductWeight",
    "ConstantWeight",
    "ColumnWeight",
    "InverseVarianceWeight",
    "FKPWeight",
    "QualityMaskWeight",
    "HealpixMapWeight",
    "PdfMeanRedshiftWeight",
    "PdfWidthIVarWeight",
    "PipBitweightWeight",
    "ShearWeight",
    "CompletenessWeight",
    "FiberCollisionWeight",
    "ZFailureWeight",
    "boss_total_weight",
    "default_weight_for",
    "register_default",
    "unregister_default",
]
