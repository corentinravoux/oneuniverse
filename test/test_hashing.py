"""Tests for sha256 content-hash helpers."""
import hashlib
from pathlib import Path

from oneuniverse.data._hashing import HASH_LEN, hash_bytes, hash_file


def test_hash_bytes_known_value():
    expected = hashlib.sha256(b"hello").hexdigest()[:HASH_LEN]
    assert hash_bytes(b"hello") == expected
    assert len(hash_bytes(b"hello")) == HASH_LEN


def test_hash_file_matches_bytes(tmp_path: Path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello")
    assert hash_file(f) == hash_bytes(b"hello")


def test_hash_file_streams_large_input(tmp_path: Path):
    f = tmp_path / "big.bin"
    # 10 MB, chunk-boundary crossings.
    payload = b"abcdefgh" * (10_000_000 // 8)
    f.write_bytes(payload)
    assert hash_file(f) == hash_bytes(payload)


def test_hash_len_is_16():
    assert HASH_LEN == 16
