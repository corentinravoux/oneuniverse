def test_public_surface():
    import oneuniverse.data as d
    for name in ["Geometry", "SurveyMeta", "Manifest", "read_manifest", "DatasetView", "write_dataset",
                 "convert", "list_surveys", "get_loader", "register", "SurveyLoader", "FitsCatalogLoader",
                 "PdfSpec", "ProbabilisticRedshift", "Cone", "SkyPatch", "ZRange", "Database",
                 "CrossMatchRules", "OneuidIndex", "OneuidQuery", "SubobjectRules", "SubobjectLinks",
                 "VersionedSidecar"]:
        assert hasattr(d, name)
    assert len(d.__all__) <= 32
    assert all(hasattr(d, n) for n in d.__all__)


def test_top_level_surface():
    import oneuniverse as ou
    for name in ["data", "convert", "DatasetView", "Cone", "SkyPatch", "ZRange", "__version__"]:
        assert hasattr(ou, name)
