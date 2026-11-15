"""Diagnostic figure scripts (slow), the CLI scripts end to end (slow) and their --help (fast)."""
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
from fixtures.fits_like import write_fake_desi_dr1_fits, write_fake_eboss_dr16q_fits

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


def _script_main(name):
    return runpy.run_path(str(ROOT / "scripts" / f"{name}.py"))["main"]


@pytest.mark.slow
def test_convert_surveys_main_on_fake_archive(tmp_path, capsys):
    from oneuniverse.data import Database
    from oneuniverse.data.loaders import get_loader

    archive = tmp_path / "archive"
    eboss = write_fake_eboss_dr16q_fits(archive / get_loader("eboss_qso").archive_path, 2_000)
    write_fake_desi_dr1_fits(archive / get_loader("desi_qso").archive_path, 2_000)
    main = _script_main("convert_surveys")
    main(["--data-root", str(archive), "--db-root", str(tmp_path / "db")])
    db = Database(tmp_path / "db")
    assert sorted(db.names()) == ["desi_qso", "eboss_qso"]
    assert "eboss_qso:" in capsys.readouterr().out

    # --source overrides the archive layout for one survey
    moved = tmp_path / "elsewhere"
    moved.mkdir()
    eboss.rename(moved / eboss.name)
    main(["--data-root", str(archive), "--db-root", str(tmp_path / "db2"),
          "--surveys", "eboss_qso", "--source", f"eboss_qso={moved}"])
    assert Database(tmp_path / "db2").names() == ["eboss_qso"]
    with pytest.raises(SystemExit, match="NAME=PATH"):
        main(["--data-root", str(archive), "--db-root", str(tmp_path / "db3"), "--source", "x"])


@pytest.mark.slow
def test_build_reference_store_main_tiny_grid(tmp_path):
    from oneuniverse.simulation import SimStore

    out = tmp_path / "store"
    main = _script_main("build_reference_store")
    main(["--out", str(out), "--n-grid", "16", "--box", "64", "--redshifts", "0.0", "1.0"])
    assert (out / "manifest.json").is_file()
    assert SimStore(out).products
    with pytest.raises(SystemExit, match="already exists"):
        main(["--out", str(out), "--n-grid", "16", "--box", "64"])
