"""Phase S3 T9 — linear-sim public API."""
import oneuniverse.simulation.linear as lin


def test_public_exports_present():
    for name in (
        "transfer_eh_nowiggle", "linear_power", "sigma_R",
        "growth_factor", "growth_rate",
        "generate_density_field", "zeldovich_particles", "find_peaks",
        "generate_linear_sim",
    ):
        assert hasattr(lin, name), f"missing export: {name}"
