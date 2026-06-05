"""H1 fix — registered survey loaders declare a ready/planned status.

Discovery must not silently advertise scaffold loaders that raise
NotImplementedError when loaded.
"""
import pytest

from oneuniverse.data import list_surveys, survey_status

_READY = {"eboss_qso", "desi_qso", "dummy"}
_PLANNED = {"cosmicflows4", "des_dr2", "desi_bgs", "desi_pv", "pantheonplus",
            "sdss_mgs", "sixdfgs"}


def test_ready_surveys_are_exactly_the_implemented_ones():
    ready = set(list_surveys(status="ready"))
    assert ready == _READY


def test_planned_surveys_are_flagged():
    planned = set(list_surveys(status="planned"))
    assert planned == _PLANNED
    for name in _PLANNED:
        assert survey_status(name) == "planned"
    for name in _READY:
        assert survey_status(name) == "ready"


def test_unfiltered_list_marks_planned_in_description():
    allsurv = list_surveys()
    assert "[planned" in allsurv["sixdfgs"]
    assert "[planned" not in allsurv["eboss_qso"]


def test_survey_status_unknown_raises():
    with pytest.raises(KeyError):
        survey_status("not_a_survey")
