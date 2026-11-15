"""Survey loaders: built-ins register on import, then third-party entry points."""
from oneuniverse.data.loaders.base import FitsCatalogLoader, SurveyLoader, read_fits_batches
from oneuniverse.data.loaders.registry import get_loader, list_surveys, load_entry_points, register
from oneuniverse.data.loaders import desi_qso as _desi_qso      # noqa: F401  (registers)
from oneuniverse.data.loaders import eboss_qso as _eboss_qso    # noqa: F401  (registers)

load_entry_points()

__all__ = ["SurveyLoader", "FitsCatalogLoader", "read_fits_batches", "register",
           "get_loader", "list_surveys", "load_entry_points"]
