"""Diagnostic figure scripts (slow) and a --help smoke test of the CLI scripts (fast)."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = ["sky_coverage", "partition_pruning", "crossmatch_separations",
               "measurement_set_overview", "resim_buffer_convergence", "mock_challenge"]
CLI = ["convert_surveys", "build_reference_store"]


@pytest.mark.slow
@pytest.mark.parametrize("name", DIAGNOSTICS)
def test_diagnostic_script_writes_one_png(name, tmp_path):
    script = ROOT / "examples" / "diagnostics" / f"{name}.py"
    subprocess.run([sys.executable, str(script), "--out", str(tmp_path)], check=True, timeout=120)
    pngs = list(tmp_path.glob("*.png"))
    assert [p.name for p in pngs] == [f"{name}.png"]
    assert pngs[0].stat().st_size > 30_000


@pytest.mark.parametrize("name", CLI)
def test_cli_script_help(name):
    script = ROOT / "scripts" / f"{name}.py"
    out = subprocess.run([sys.executable, str(script), "--help"], check=True, timeout=60,
                         capture_output=True, text=True)
    assert out.stdout.startswith("usage:")


@pytest.mark.slow
@pytest.mark.parametrize("name", ["sky_coverage", "partition_pruning", "crossmatch_separations",
                                  "measurement_set_overview"])
def test_data_root_is_left_unchanged(name, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "examples" / "diagnostics"))
    from _common import synthetic_database

    root = tmp_path / "db"
    root.mkdir()
    synthetic_database(root, n=3_000, n_common=1_000)

    def listing():
        return sorted((str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns)
                      for p in root.rglob("*"))

    before = listing()
    out = tmp_path / "out"
    subprocess.run([sys.executable, str(ROOT / "examples" / "diagnostics" / f"{name}.py"),
                    "--out", str(out), "--data-root", str(root), "--dataset", "a", "--dataset", "b"],
                   check=True, timeout=120)
    assert listing() == before
    assert len(list(out.glob("*.png"))) == 1
