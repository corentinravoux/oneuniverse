"""
oneuniverse.data.surveys
~~~~~~~~~~~~~~~~~~~~~~~~~
Survey sub-packages organised by type.

::

    surveys/
    ├── test/              dummy (in-memory, for testing)
    ├── spectroscopic/     sdss_mgs, desi_bgs, sixdfgs
    ├── peculiar_velocity/ cosmicflows4, desi_pv
    ├── photometric/       des_dr2
    └── snia/              pantheonplus

Importing this package auto-imports every survey sub-sub-package,
which triggers their ``@register`` decorators.
"""

# ── Auto-import all survey sub-sub-packages ──────────────────────────────
# Each type directory's __init__.py re-exports its children.
from oneuniverse.data.surveys import (  # noqa: F401
    test,
    spectroscopic,
    peculiar_velocity,
    photometric,
    snia,
)
