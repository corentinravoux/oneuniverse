"""Phase S2 T6 — ProductDecl."""
import pytest

from oneuniverse.simulation.product import ProductDecl


def test_ok():
    p = ProductDecl(
        product="snapshots", native_format="ASDF/pack9",
        indexes=("healpix_tiles", "halo_particle_ptr"),
        fields=("Coordinates", "Velocities"),
    )
    assert p.product == "snapshots"


def test_rejects_unknown_product():
    with pytest.raises(ValueError, match="product"):
        ProductDecl(
            product="not_a_product", native_format="x",
            indexes=(), fields=(),
        )


def test_roundtrip():
    p = ProductDecl(
        product="lightcone", native_format="FITS HEALPix",
        indexes=("lightcone_shell",), fields=("kappa", "gamma1", "gamma2"),
    )
    assert ProductDecl.from_dict(p.to_dict()) == p
