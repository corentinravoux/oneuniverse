"""Default weights keyed by (survey_kind, z_type)."""
from __future__ import annotations

from collections.abc import Callable

from oneuniverse._core.errors import NotFoundError, ValidationError
from oneuniverse.combine.weights import InverseVarianceWeight, Weight
from oneuniverse.data.schema import z_types

_DEFAULTS: dict[tuple[str, str], Callable[[], Weight]] = {
    ("spectroscopic", "spec"): lambda: InverseVarianceWeight("z_err"),
    ("photometric", "phot"): lambda: InverseVarianceWeight("z_err"),
    ("photometric", "phot_pdf"): lambda: InverseVarianceWeight("z_pdf_std"),
    ("peculiar_velocity", "pv"): lambda: InverseVarianceWeight("v_pec_err"),
}


def default_weight_for(survey_kind: str, z_type: str) -> Weight:
    """A fresh default weight for the key; NotFoundError lists known keys."""
    try:
        factory = _DEFAULTS[(survey_kind, z_type)]
    except KeyError:
        raise NotFoundError(
            f"no default weight for ({survey_kind!r}, {z_type!r}); known: {sorted(_DEFAULTS)}"
        ) from None
    return factory()


def register_default(survey_kind: str, z_type: str, factory: Callable[[], Weight],
                     *, replace: bool = False) -> None:
    """Register *factory* for the key; existing keys need ``replace=True``."""
    if z_type not in z_types():
        raise ValidationError(f"unknown z_type {z_type!r}; registered: {sorted(z_types())}")
    if (survey_kind, z_type) in _DEFAULTS and not replace:
        raise ValidationError(f"default for ({survey_kind!r}, {z_type!r}) exists; pass replace=True")
    _DEFAULTS[(survey_kind, z_type)] = factory


def defaults() -> dict[tuple[str, str], Callable[[], Weight]]:
    """Copy of the registry."""
    return dict(_DEFAULTS)
