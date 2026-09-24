"""
Shared pytest fixtures/setup.

rule_ranker.ranker and rule_ranker.ranker_postprocess both `import
pyopencl as cl` at module level. No OpenCL ICD is available in a plain
test/CI environment, so importing those modules would fail before a
single test even runs -- even though the vast majority of their code
(rule validators, popcount helpers, CSV/rule parsing, argument parsing,
etc.) is pure Python/NumPy and perfectly testable without a GPU.

This conftest installs a minimal stub module for `pyopencl` into
sys.modules *before* any test imports the real package, so `import
pyopencl as cl` succeeds at import time. The stub only needs to exist --
nothing here calls into an actual GPU kernel, so the stub's attributes
are never exercised, only its presence as an importable module is.
"""
import sys
import types


def _install_pyopencl_stub():
    if "pyopencl" in sys.modules:
        return  # real pyopencl is installed; use it
    stub = types.ModuleType("pyopencl")

    class _DeviceType:
        GPU = object()
        CPU = object()
        ALL = object()

    class _MemFlags:
        READ_ONLY = 1
        WRITE_ONLY = 2
        READ_WRITE = 4
        COPY_HOST_PTR = 8
        ALLOC_HOST_PTR = 16

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "pyopencl is stubbed out for testing (no real OpenCL/GPU "
            "available in this environment); this call path requires "
            "an actual GPU and is out of scope for unit tests."
        )

    stub.device_type = _DeviceType
    stub.mem_flags = _MemFlags
    stub.get_platforms = _unavailable
    stub.Context = _unavailable
    stub.CommandQueue = _unavailable
    stub.Program = _unavailable
    stub.Buffer = _unavailable
    stub.enqueue_copy = _unavailable
    stub.enqueue_fill_buffer = _unavailable

    sys.modules["pyopencl"] = stub


_install_pyopencl_stub()
