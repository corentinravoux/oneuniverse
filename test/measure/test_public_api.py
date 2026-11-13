import oneuniverse.measure as m

NAMES = ["Nz", "Window", "Regions", "Product", "FieldMap", "Sightline", "MeasurementSpec",
         "MeasurementSet", "build_pointset", "build_sightlines", "attach_photoz",
         "tomographic_nz", "galaxy_clustering", "cosmic_shear", "three_by_two", "save", "load"]


def test_public_surface_is_exactly_the_17_names():
    assert len(NAMES) == 17 and sorted(m.__all__) == sorted(NAMES)
    for n in NAMES:
        assert hasattr(m, n)
