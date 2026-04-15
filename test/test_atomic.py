"""Tests for the atomic-write helpers."""
import os
from pathlib import Path

import pytest

from oneuniverse.data._atomic import atomic_write_bytes, atomic_write_text


def test_atomic_write_bytes_creates_final_file(tmp_path):
    target = tmp_path / "a.bin"
    atomic_write_bytes(target, b"hello")
    assert target.read_bytes() == b"hello"
    # No .tmp leftovers in the directory.
    assert [p.name for p in tmp_path.iterdir()] == ["a.bin"]


def test_atomic_write_text_overwrites_existing(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("old")
    atomic_write_text(target, "new")
    assert target.read_text() == "new"
    assert [p.name for p in tmp_path.iterdir()] == ["a.txt"]


def test_atomic_write_bytes_leaves_no_tmp_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "a.bin"

    def boom(src, dst):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_bytes(target, b"hello")
    # No leftover tmp and no partial final file.
    assert list(tmp_path.iterdir()) == []
