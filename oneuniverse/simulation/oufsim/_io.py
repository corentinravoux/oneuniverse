"""Atomic write helpers for the OUF-Sim store.

Duplicated (not imported) from ``oneuniverse.data._atomic`` by design —
Pillar 3 must not depend on Pillar 1 (Rule 1). Same guarantee: readers
see either the old file or the new file, never a half-written one.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Union


def _tmp_path_for(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")


def atomic_write_bytes(path: Union[str, Path], data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def atomic_write_text(path: Union[str, Path], text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def write_json(path: Union[str, Path], payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, default=str))


def read_json(path: Union[str, Path]) -> Any:
    return json.loads(Path(path).read_text())
