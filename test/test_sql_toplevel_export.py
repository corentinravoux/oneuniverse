def test_export_sql_is_discoverable_from_top_level():
    import oneuniverse
    assert hasattr(oneuniverse, "export_sql")
