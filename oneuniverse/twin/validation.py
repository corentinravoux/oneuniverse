"""Compat re-export — implementations moved to :mod:`oneuniverse.twin.metrics`
(S10 consolidation). Kept so existing ``from oneuniverse.twin.validation import …``
imports keep working. New code should import from ``oneuniverse.twin.metrics``.
"""
from __future__ import annotations

from oneuniverse.twin.metrics import (  # noqa: F401
    RecoveryMetrics,
    recover_metrics,
)
