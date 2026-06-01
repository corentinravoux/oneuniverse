"""Phase S2 T3 — BackendCapabilities."""
from oneuniverse.simulation.capabilities import BackendCapabilities
from oneuniverse.simulation.execution import ExecutionMode


def test_defaults():
    cap = BackendCapabilities(name="dummy", native_format="Gadget HDF5")
    assert cap.supports_mpi is False
    assert cap.supports_gpu_direct is False
    assert cap.supports_random_access is False
    assert cap.supports_streaming is True
    assert cap.requires_extra == ()
    assert cap.heavy_step_modes == {}


def test_modes_for_default_is_sequential():
    cap = BackendCapabilities(name="x", native_format="f")
    assert cap.modes_for("region_extract") == (ExecutionMode.SEQUENTIAL,)


def test_modes_for_declared_step():
    cap = BackendCapabilities(
        name="x", native_format="f",
        heavy_step_modes={
            "region_extract": (ExecutionMode.SEQUENTIAL, ExecutionMode.MPI),
        },
    )
    assert cap.modes_for("region_extract") == (
        ExecutionMode.SEQUENTIAL, ExecutionMode.MPI,
    )


def test_supports_mode():
    cap = BackendCapabilities(
        name="x", native_format="f",
        heavy_step_modes={"index_build": (ExecutionMode.MPI,)},
    )
    assert cap.supports("index_build", ExecutionMode.MPI) is True
    assert cap.supports("index_build", ExecutionMode.GPU) is False
    # Undeclared step defaults to SEQUENTIAL-only.
    assert cap.supports("foo", ExecutionMode.SEQUENTIAL) is True
    assert cap.supports("foo", ExecutionMode.MPI) is False
