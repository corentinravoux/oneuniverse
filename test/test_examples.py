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
