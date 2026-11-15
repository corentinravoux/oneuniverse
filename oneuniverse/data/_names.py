"""Shared argument validation for name lists."""
from __future__ import annotations

from collections.abc import Iterable

from oneuniverse._core.errors import ValidationError


def name_list(value: Iterable[str], what: str) -> list[str]:
    """Return *value* as a list of names; a bare ``str`` is a ValidationError."""
    if isinstance(value, (str, bytes)):
        raise ValidationError(f"{what}: pass a list of names, not the string {value!r}")
    return list(value)
