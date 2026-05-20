"""
oneuniverse.combine.weights.registry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Opinionated default-weight factory keyed on ``(survey_type, z_type)``.

The returned :class:`Weight` is the recommended per-object weight for a
survey of that kind. Callers are free to override on a per-survey basis
via :meth:`WeightedCatalog.add_weight`.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

from oneuniverse.combine.weights.base import Weight
from oneuniverse.combine.weights.ivar import InverseVarianceWeight

Key = Tuple[str, str]
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
    ("spectroscopic", "spec"): _ivar_spec,
    ("photometric", "phot"): _ivar_phot,
    ("peculiar_velocity", "pec"): _ivar_pec,
    ("photometric", "phot_pdf"): _ivar_pdf_width,
}


def default_weight_for(survey_type: str, z_type: str) -> Weight:
    """Return the recommended default :class:`Weight` for a given survey.

    Parameters
    ----------
    survey_type : str
        e.g. ``"spectroscopic"``, ``"photometric"``, ``"peculiar_velocity"``.
    z_type : str
        e.g. ``"spec"``, ``"phot"``, ``"pec"``.

    Raises
    ------
    KeyError
        If no default is registered for the pair. Callers should supply
        an explicit :class:`Weight` via ``WeightedCatalog.add_weight``.
    """
    key = (survey_type, z_type)
    try:
        return _DEFAULTS[key]()
    except KeyError:
        raise KeyError(
            f"No default weight registered for (survey_type={survey_type!r}, "
            f"z_type={z_type!r}). Known pairs: {list(_DEFAULTS)}"
        ) from None


def register_default(
    survey_type: str, z_type: str, factory: Factory,
) -> None:
    """Register a default :class:`Weight` factory for ``(survey_type, z_type)``.

    Raises :class:`ValueError` if the key already has a registration —
    callers must explicitly :func:`unregister_default` first to avoid
    silent clobber of the canonical defaults.
    """
    key = (survey_type, z_type)
    if key in _DEFAULTS:
        raise ValueError(
            f"register_default: {key!r} is already registered "
            f"(call unregister_default first if you intend to replace it)"
        )
    _DEFAULTS[key] = factory


def unregister_default(survey_type: str, z_type: str) -> None:
    """Remove the default factory registered for ``(survey_type, z_type)``.

    Raises :class:`KeyError` if no such key is registered.
    """
    key = (survey_type, z_type)
    del _DEFAULTS[key]
