"""Atomic file and directory publication helpers, and content hashing."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from oneuniverse._core.errors import FormatError

_CHUNK = 1 << 20


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Write *data* to *path* via a same-directory temp file and ``os.replace``."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_json(path: str | Path, obj: Any) -> None:
    """Atomically write *obj* as indented JSON."""
    atomic_write_bytes(path, json.dumps(obj, indent=2).encode("utf-8"))


@contextlib.contextmanager
def atomic_dir(path: str | Path) -> Iterator[Path]:
    """Build a directory in ``<path>.tmp`` and publish it over *path* on success.

    On exception ``<path>.tmp`` is removed and *path* is left untouched.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    old = path.with_name(path.name + ".old")
    if tmp.exists():
        raise FormatError(f"stale temporary directory exists: {tmp}")
    tmp.mkdir(parents=True)
    try:
        yield tmp
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if path.exists():
        os.rename(path, old)
    os.rename(tmp, path)
    if old.exists():
        shutil.rmtree(old)


def hash_bytes(data: bytes) -> str:
    """First 16 hex chars of the sha256 of *data*."""
    return hashlib.sha256(data).hexdigest()[:16]


def hash_file(path: str | Path) -> str:
    """First 16 hex chars of the sha256 of the file contents (chunked read)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()[:16]
