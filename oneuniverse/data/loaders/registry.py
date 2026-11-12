"""Survey-loader registry (built-ins plus the ``oneuniverse.survey_loaders`` entry points)."""
from __future__ import annotations

from oneuniverse._core.errors import ValidationError
from oneuniverse._core.registry import Registry
from oneuniverse.data.loaders.base import SurveyLoader
from oneuniverse.data.manifest import SurveyMeta

ENTRY_POINT_GROUP = "oneuniverse.survey_loaders"
_registry: Registry[type[SurveyLoader]] = Registry("survey loader", key=lambda cls: cls.meta.name)


def register(cls):
    return _registry.register(cls)


def get_loader(name: str) -> SurveyLoader:
    return _registry.get(name)()


def list_surveys() -> dict[str, SurveyMeta]:
    return {n: _registry.get(n).meta for n in _registry.names()}


def _validate(obj: object) -> None:
    if not (isinstance(obj, type) and issubclass(obj, SurveyLoader)):
        raise ValidationError(f"{obj!r} is not a SurveyLoader subclass")
    if not isinstance(getattr(obj, "meta", None), SurveyMeta):
        raise ValidationError(f"{obj.__name__} has no SurveyMeta `meta`")


def load_entry_points() -> list[str]:
    """Register third-party loaders; broken plugins are skipped with a warning."""
    return _registry.load_entry_points(ENTRY_POINT_GROUP, on_error="warn", validate=_validate)
