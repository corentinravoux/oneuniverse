"""Combine datasets: weights, default weights, weighted catalogs."""
from oneuniverse.combine.defaults import default_weight_for, register_default
from oneuniverse.combine.weights import (
    ColumnWeight,
    ConstantWeight,
    FKPWeight,
    HealpixMapWeight,
    InverseVarianceWeight,
    ProductWeight,
    QualityMaskWeight,
    ShearWeight,
    Weight,
)

__all__ = [
    "Weight", "ProductWeight", "ColumnWeight", "ConstantWeight",
    "InverseVarianceWeight", "FKPWeight", "HealpixMapWeight", "ShearWeight",
    "QualityMaskWeight", "default_weight_for", "register_default",
]
