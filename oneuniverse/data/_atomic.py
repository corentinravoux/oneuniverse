"""Atomic write helpers.

Write to a per-process .tmp file, then ``os.replace`` to the final path.
Guarantees: readers either see the old file or the new file, never a
half-written one, even on Ctrl-C or a crash between steps.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Union


def _tmp_path_for(path: Path) -> Path:
    return path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )


def atomic_write_bytes(path: Union[str, Path], data: bytes) -> None:
    """Atomically write *data* to *path*.

    Writes to a sibling ``.<name>.tmp.<pid>.<ns>`` file, fsync's it, then
    ``os.replace``. On any exception the tmp file is cleaned up so no
    debris is left on disk.
    """
    path = Path(path)
    tmp = _tmp_path_for(path)
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(
    path: Union[str, Path], text: str, encoding: str = "utf-8"
) -> None:
    """Atomically write *text* to *path* with the given *encoding*."""
    atomic_write_bytes(path, text.encode(encoding))
