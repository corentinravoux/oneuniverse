import subprocess
import sys
from pathlib import Path

import pytest
from fixtures.fits_like import write_fake_desi_dr1_fits, write_fake_eboss_dr16q_fits

from oneuniverse._core.errors import FormatError
from oneuniverse.data import DatasetView, convert, get_loader


def test_convert_desi_end_to_end(tmp_path):
    src = write_fake_desi_dr1_fits(tmp_path / "raw", 6_000, 0)
    m = convert("desi_qso", source=src.parent, out=tmp_path / "db/desi_qso", batch_size=1_000)
    assert m.survey.name == "desi_qso" and m.loader.kwargs == {"batch_size": 1_000}
    assert m.original_files[0].path == "QSO_full.dat.fits" and m.original_files[0].n_rows == 6_000
    v = DatasetView.from_path(tmp_path / "db/desi_qso")
    assert v.n_rows == m.n_rows and v.manifest.extension("spectrum")["wavelength_convention"] == "vacuum"


def test_convert_default_kwargs_empty(tmp_path):
    src = write_fake_eboss_dr16q_fits(tmp_path / "raw", 500, 0)
    m = convert("eboss_qso", source=src.parent, out=tmp_path / "e")
    assert m.loader.kwargs == {}


def test_convert_accepts_loader_instance_and_overwrite(tmp_path):
    src = write_fake_eboss_dr16q_fits(tmp_path / "raw", 1_000, 0)
    convert(get_loader("eboss_qso"), source=src.parent, out=tmp_path / "e")
    with pytest.raises(FormatError):
        convert("eboss_qso", source=src.parent, out=tmp_path / "e")
    m = convert("eboss_qso", source=src.parent, out=tmp_path / "e", overwrite=True)
    assert m.validity.version == 2


def test_convert_bad_kwarg_leaves_nothing(tmp_path):
    src = write_fake_eboss_dr16q_fits(tmp_path / "raw", 100, 0)
    with pytest.raises(TypeError):
        convert("eboss_qso", source=src.parent, out=tmp_path / "e", bogus=1)
    assert not (tmp_path / "e").exists() and not (tmp_path / "e.tmp").exists()


@pytest.mark.parametrize("bad", [0, -5, 2.5, "10", True])
def test_convert_rejects_bad_batch_size(tmp_path, bad):
    from oneuniverse._core.errors import ValidationError
    src = write_fake_eboss_dr16q_fits(tmp_path / "raw", 100, 0)
    with pytest.raises(ValidationError, match="batch_size"):
        convert("eboss_qso", source=src.parent, out=tmp_path / "e", batch_size=bad)

_CONVERT_MEM_SCRIPT = """
import sys, threading, time, pathlib
sys.path[:0] = [sys.argv[1], sys.argv[2]]
from fixtures.fits_like import write_fake_desi_dr1_fits, write_fake_eboss_dr16q_fits
from oneuniverse.data import convert

def anon():
    for line in open("/proc/self/status"):
        if line.startswith("RssAnon:"):
            return int(line.split()[1]) / 1024
raw = pathlib.Path(sys.argv[3]) / "raw"
write = write_fake_desi_dr1_fits if sys.argv[5] == "desi_qso" else write_fake_eboss_dr16q_fits
src = write(raw, int(sys.argv[4]), 0)
import gc; gc.collect()
base, peak, stop = anon(), [0.0], threading.Event()
def poll():
    while not stop.is_set():
        peak[0] = max(peak[0], anon()); time.sleep(0.05)
t = threading.Thread(target=poll); t.start()
m = convert(sys.argv[5], source=src.parent, out=pathlib.Path(sys.argv[3]) / "db")
stop.set(); t.join()
print(peak[0] - base, m.n_rows)
"""


def _convert_rss_growth_mib(tmp_path, survey, n_rows):
    root = Path(__file__).resolve().parents[2]
    r = subprocess.run(
        [sys.executable, "-c", _CONVERT_MEM_SCRIPT, str(root), str(root / "test"),
         str(tmp_path), str(n_rows), survey], capture_output=True, text=True, check=True)
    growth, n = r.stdout.split()
    assert int(n) > 0
    print(f"convert {survey} {n_rows} rows: peak RssAnon growth {float(growth):.0f} MiB")
    return float(growth)


@pytest.mark.slow
def test_convert_desi_2m_bounded_memory(tmp_path):
    assert _convert_rss_growth_mib(tmp_path, "desi_qso", 2_000_000) < 400  # MiB


@pytest.mark.slow
def test_convert_eboss_750k_bounded_memory(tmp_path):
    assert _convert_rss_growth_mib(tmp_path, "eboss_qso", 750_000) < 400  # MiB
