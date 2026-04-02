"""
oneuniverse — unified galaxy survey catalog package.
"""

__version__ = "0.1.0"

from oneuniverse import data
from oneuniverse.data import (
    load_catalog,
    list_surveys,
    list_survey_types,
    get_survey_config,
    Cone,
    Shell,
    SkyPatch,
    set_data_root,
)

__all__ = [
    "data",
    "load_catalog",
    "list_surveys",
    "list_survey_types",
    "get_survey_config",
    "Cone",
    "Shell",
    "SkyPatch",
    "set_data_root",
]
