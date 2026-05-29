"""
oneuniverse.combine.weights.registry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Opinionated default-weight factory keyed on
``(survey_type, sub_kind, z_type)``. ``sub_kind=None`` is the original
two-key behaviour and stays the fallback when no sub-species match is
registered. Sub-kind keys let surveys split a single
``(survey_type, z_type)`` into species like DESI ``BGS_BRIGHT`` vs
``BGS_FAINT`` or DES Y3 ``METACAL`` vs ``MCAL2`` while keeping the
top-level default intact.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

from oneuniverse.combine.weights.base import Weight
from oneuniverse.combine.weights.ivar import InverseVarianceWeight

Key = Tuple[str, Optional[str], str]
Factory = Callable[[], Weight]


def _ivar_spec() -> Weight:
    return InverseVarianceWeight("z_spec_err", name="ivar(z_spec)")


def _ivar_phot() -> Weight:
    return InverseVarianceWeight("z_phot_err", name="ivar(z_phot)")


def _ivar_pec() -> Weight:
    return InverseVarianceWeight("velocity_error", name="ivar(vpec)")


def _ivar_pdf_width() -> Weight:
    from oneuniverse.combine.weights.pdf import PdfWidthIVarWeight
    return PdfWidthIVarWeight(std_column="z_pdf_std")


_DEFAULTS: Dict[Key, Factory] = {
    ("spectroscopic", None, "spec"): _ivar_spec,
    ("photometric", None, "phot"): _ivar_phot,
    ("peculiar_velocity", None, "pec"): _ivar_pec,
    ("photometric", None, "phot_pdf"): _ivar_pdf_width,
}


def default_weight_for(
    survey_type: str,
    z_type: str,
    *,
    sub_kind: Optional[str] = None,
) -> Weight:
    """Return the recommended default :class:`Weight`.

    Resolution order:

    1. ``(survey_type, sub_kind, z_type)`` if ``sub_kind`` is not None.
    2. ``(survey_type, None, z_type)`` (the canonical default).
    """
    if sub_kind is not None:
        key = (survey_type, sub_kind, z_type)
        if key in _DEFAULTS:
            return _DEFAULTS[key]()
    key = (survey_type, None, z_type)
    try:
        return _DEFAULTS[key]()
    except KeyError:
        raise KeyError(
            f"No default weight registered for "
            f"(survey_type={survey_type!r}, sub_kind={sub_kind!r}, "
            f"z_type={z_type!r}). Known keys: {sorted(_DEFAULTS)}"
        ) from None


def register_default(
    survey_type: str,
    z_type: str,
    factory: Factory,
    *,
    sub_kind: Optional[str] = None,
) -> None:
    """Register a default :class:`Weight` factory for
    ``(survey_type, sub_kind, z_type)``. Default ``sub_kind=None``
    matches the canonical pre-Phase-19 contract.
    """
    key = (survey_type, sub_kind, z_type)
    if key in _DEFAULTS:
        raise ValueError(
            f"register_default: {key!r} is already registered "
            f"(call unregister_default first if you intend to replace it)"
        )
    _DEFAULTS[key] = factory


def unregister_default(
    survey_type: str,
    z_type: str,
    *,
    sub_kind: Optional[str] = None,
) -> None:
    """Remove the default factory for
    ``(survey_type, sub_kind, z_type)``.
    """
    key = (survey_type, sub_kind, z_type)
    del _DEFAULTS[key]
