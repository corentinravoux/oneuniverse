"""The README example runs verbatim on synthetic survey files."""
import re
from pathlib import Path

from fixtures.fits_like import write_fake_desi_dr1_fits, write_fake_eboss_dr16q_fits

README = Path(__file__).resolve().parents[1] / "README.md"


def _block() -> str:
    text = README.read_text()
    m = re.search(r"<!-- example:start -->\s*```python\n(.*?)```\s*<!-- example:end -->", text, re.DOTALL)
    assert m, "README example markers not found"
    return m.group(1)


def test_readme_example_runs(tmp_path):
    raw, db_root = tmp_path / "raw", tmp_path / "db"
    eboss = write_fake_eboss_dr16q_fits(raw / "eboss_qso", 3000)
    write_fake_desi_dr1_fits(raw / "desi_qso", 3000, share_with=eboss, n_shared=500)
    ns = {"raw": raw, "db_root": db_root, "out_path": tmp_path / "gc"}
    exec(compile(_block(), "README.md", "exec"), ns)  # noqa: S102
    ms = ns["ms"]
    assert ms.products["gal"].randoms is not None
    assert len(ns["df"]) > 0
    assert ns["ms2"].spec == ms.spec
    assert ns["idx"].n_multi > 0
    both = ns["both"].to_pandas()
    per = both.groupby("oneuid")["dataset"].nunique()
    assert (per > 1).any(), "no object with rows from both surveys"
