"""Content hashing for manifests.

16 hex chars (64-bit) prefix of sha256 is enough to fingerprint files
for change detection and reproducibility audits without bloating
manifests.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union

HASH_LEN: int = 16  # hex chars; 64-bit prefix of sha256
_CHUNK: int = 1 << 20  # 1 MiB


def hash_bytes(data: bytes) -> str:
    """Return the 16-hex-char prefix of sha256(data)."""
    return hashlib.sha256(data).hexdigest()[:HASH_LEN]


def hash_file(path: Union[str, Path]) -> str:
    """Stream *path* through sha256 and return the 16-hex-char prefix."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:HASH_LEN]
