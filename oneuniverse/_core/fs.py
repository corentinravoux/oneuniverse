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

import numpy as np

from oneuniverse._core.errors import FormatError

_CHUNK = 1 << 20

STAGING_SUFFIXES = (".tmp", ".old")
"""Suffixes of the transient names used while publishing a directory."""

# The process umask can only be read by setting it, which is process-global and
# racy; read it once at import (before worker threads exist) and reuse it.
_UMASK = os.umask(0)
os.umask(_UMASK)


def is_staging(path: str | Path) -> bool:
    """True if *path* names a transient ``<target>.tmp`` / ``<target>.old`` entry."""
    return Path(path).name.endswith(STAGING_SUFFIXES)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Write *data* to *path* via a same-directory temp file and ``os.replace``.

    The published file gets mode ``0o666 & ~umask`` (``mkstemp`` alone would
    leave it owner-only).
    """
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        os.fchmod(fd, 0o666 & ~_UMASK)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _json_default(o: Any) -> Any:
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def atomic_write_json(path: str | Path, obj: Any) -> None:
    """Atomically write *obj* as indented JSON.

    NaN / infinity are rejected (``allow_nan=False``); numpy scalars and arrays
    are converted to plain Python values.
    """
    text = json.dumps(obj, indent=2, allow_nan=False, default=_json_default)
    atomic_write_bytes(path, text.encode("utf-8"))


@contextlib.contextmanager
def atomic_dir(path: str | Path) -> Iterator[Path]:
    """Build a directory in ``<path>.tmp`` and publish it over *path* on success.

    On exception ``<path>.tmp`` is removed and *path* is left untouched.

    Publication is ``rename(path, path.old)``, ``rename(path.tmp, path)``,
    ``rmtree(path.old)``. A hard kill leaves one of three states, each detected
    (and refused with a :class:`FormatError` naming the safe action) by the
    next call:

    (a) killed during the body: *path* intact, ``path.tmp`` stale.
    (b) killed between the two renames: *path* missing, ``path.old`` is the
        complete previous version and ``path.tmp`` the complete new one.
    (c) killed during ``rmtree(old)``: *path* is the new version, ``path.old``
        partial leftover.

    Limitations: the swap is not atomic for concurrent readers (there is a
    window between the two renames where *path* does not exist), and durability
    across power loss is not guaranteed (no directory fsync).
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    old = path.with_name(path.name + ".old")
    if not path.exists() and tmp.exists() and old.exists():
        raise FormatError(
            f"{path} missing while {tmp} and {old} present: a previous publish was "
            f"interrupted after moving the old version aside; rename {tmp} to "
            f"{path} to finish it, then delete {old}")
    if not path.exists() and old.exists():
        raise FormatError(
            f"{path} missing while {old} present: a previous publish was "
            f"interrupted; rename {old} to {path} to restore the previous version")
    if tmp.exists():
        raise FormatError(
            f"stale temporary directory {tmp} (a previous build was interrupted "
            f"or another writer is active; {path} is intact): delete {tmp} "
            f"if no other writer is running")
    if old.exists():
        raise FormatError(
            f"stale previous-version directory {old} ({path} is the new, complete "
            f"version; an earlier cleanup was interrupted): delete {old}")
    try:
        tmp.mkdir(parents=True)
    except FileExistsError:
        raise FormatError(
            f"{tmp} appeared concurrently: another writer is publishing {path}") from None
    try:
        yield tmp
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    try:
        if path.exists():
            os.rename(path, old)
        try:
            os.rename(tmp, path)
        except BaseException:
            if old.exists() and not path.exists():
                os.rename(old, path)  # restore the previous version
            raise
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if old.exists():
        try:
            shutil.rmtree(old)
        except OSError as exc:
            raise FormatError(
                f"the new version IS published at {path}, but the previous version "
                f"could not be removed ({exc}): delete {old}") from exc


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
