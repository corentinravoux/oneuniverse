"""``convert``: run a survey loader and write the result as an OUF 3.0 dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from oneuniverse._core.errors import ValidationError
from oneuniverse._core.fs import hash_file
from oneuniverse.data.loaders import FitsCatalogLoader, SurveyLoader, get_loader
from oneuniverse.data.manifest import LoaderInfo, Manifest, OriginalFile
from oneuniverse.data.writer import write_dataset

_DEFAULT_BATCH_SIZE = 100_000


def _n_rows(loader: SurveyLoader, path: Path) -> int | None:
    if isinstance(loader, FitsCatalogLoader):
        return int(fits.getheader(path, loader.hdu)["NAXIS2"])
    return None


def convert(survey: str | SurveyLoader, *, source: Path, out: Path, overwrite: bool = False,
            partition_nside: int | None = None, batch_size: int | None = None,
            **loader_kwargs) -> Manifest:
    """Convert the raw files of *survey* found in *source* into an OUF dataset at *out*.

    ``survey`` is a registered survey name or a loader instance. ``batch_size``
    (default 100 000 rows) and ``loader_kwargs`` are forwarded to the loader's
    ``batches``; only what was passed explicitly is recorded in ``manifest.loader.kwargs``.
    """
    if batch_size is not None and (
            isinstance(batch_size, bool) or not isinstance(batch_size, int | np.integer)
            or batch_size < 1):
        raise ValidationError(f"batch_size must be an int >= 1, got {batch_size!r}")
    loader = get_loader(survey) if isinstance(survey, str) else survey
    source, out = Path(source), Path(out)
    recorded = dict(loader_kwargs)
    if batch_size is not None:
        recorded["batch_size"] = batch_size
    files = [OriginalFile(path=str(p.relative_to(source)) if p.is_relative_to(source) else p.name,
                          sha256=hash_file(p), n_rows=_n_rows(loader, p))
             for p in loader.original_files(source)]
    size = _DEFAULT_BATCH_SIZE if batch_size is None else int(batch_size)
    batches = loader.batches(source, batch_size=size, **loader_kwargs)
    return write_dataset(batches, out, geometry=loader.geometry, survey=loader.meta,
                         loader=LoaderInfo(loader.meta.name, loader.version, recorded),
                         original_files=files, partition_nside=partition_nside,
                         extensions=loader.extensions, overwrite=overwrite,
                         batch_size=size)
