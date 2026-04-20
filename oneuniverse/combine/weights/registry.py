"""
oneuniverse.combine.weights.registry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Opinionated default-weight factory keyed on ``(survey_type, z_type)``.

The returned :class:`Weight` is the recommended per-object weight for a
survey of that kind. Callers are free to override on a per-survey basis
via :meth:`WeightedCatalog.add_weight`.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Callable, Mapping, Tuple

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


_DEFAULTS: Mapping[Key, Factory] = MappingProxyType({
    ("spectroscopic", "spec"): _ivar_spec,
    ("photometric", "phot"): _ivar_phot,
    ("peculiar_velocity", "pec"): _ivar_pec,
})


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
