"""Entry-point loader discovery. We monkeypatch importlib.metadata.entry_points
so the test needs no external package installed."""
from oneuniverse._registry import Registry


class _FakeEP:
    def __init__(self, name, obj):
        self.name = name
        self._obj = obj

    def load(self):
        return self._obj


def test_load_entry_points_registers_plugin(monkeypatch):
    import oneuniverse._registry as rmod
    sentinel = object()
    monkeypatch.setattr(
        rmod, "entry_points",
        lambda group=None: [_FakeEP("plugin_survey", sentinel)]
        if group == "oneuniverse.survey_loaders" else [],
        raising=False,
    )
    reg = Registry("survey loader")
    added = reg.load_entry_points("oneuniverse.survey_loaders")
    assert added == ["plugin_survey"]
    assert reg.get("plugin_survey") is sentinel


def test_builtin_wins_over_entry_point_of_same_name(monkeypatch):
    import oneuniverse._registry as rmod
    monkeypatch.setattr(
        rmod, "entry_points",
        lambda group=None: [_FakeEP("dup", object())],
        raising=False,
    )
    reg = Registry("survey loader")
    builtin = object()
    reg.register(builtin, name="dup")
    added = reg.load_entry_points("oneuniverse.survey_loaders")
    assert added == []                 # plugin skipped
    assert reg.get("dup") is builtin   # built-in retained
