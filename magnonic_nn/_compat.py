"""Import-order sensitive setup for magnum.np.

``magnum.np`` does three global things at import time that we need to control:

1. ``torch.set_default_dtype(torch.float64)``
2. ``torch.set_default_device(<cuda if available else cpu>)``
3. decorates ``ExchangeField.h`` and ``LinearFieldTerm.E`` with ``torch.compile``

The ``torch.compile`` decorators are applied while the module body executes, so
the only way to opt out is to patch ``torch.compile`` *before* the import. On
CPU the inductor backend costs more in compile time than it returns for the
mesh sizes used here, and it interacts poorly with ``torch.utils.checkpoint``
re-entrancy, so compilation is off by default. Set
``MAGNONIC_NN_COMPILE=1`` to leave ``torch.compile`` in place.

Every module in this package imports magnum.np through :func:`import_magnumnp`
(or simply imports this module first) so the patch is always in effect.
"""

from __future__ import annotations

import os
import warnings

import torch

__all__ = [
    "import_magnumnp",
    "set_precision",
    "set_device",
    "get_device",
    "compile_enabled",
]

_ORIGINAL_COMPILE = torch.compile


def compile_enabled() -> bool:
    """Whether ``torch.compile`` is left active inside magnum.np."""
    return os.environ.get("MAGNONIC_NN_COMPILE", "0") == "1"


def _identity_compile(model=None, **_kwargs):
    """Drop-in no-op replacement for :func:`torch.compile`."""
    if model is None:  # used as ``@torch.compile(...)``
        return lambda fn: fn
    return model


def import_magnumnp():
    """Import and return the ``magnumnp`` module with our global settings applied.

    Safe to call repeatedly; Python caches the module and we only patch on the
    first import.
    """
    import sys

    if "magnumnp" not in sys.modules and not compile_enabled():
        torch.compile = _identity_compile
    try:
        import magnumnp  # noqa: PLC0415
    finally:
        if not compile_enabled():
            # Restore the real symbol so user code outside magnum.np is unaffected.
            torch.compile = _ORIGINAL_COMPILE
    return magnumnp


# Trigger the import (and therefore the patch) at package import time.
magnumnp = import_magnumnp()


def set_precision(dtype="float32"):
    """Set the global torch default dtype used to build meshes and fields.

    magnum.np defaults to ``float64``. Spin-wave rollouts are dominated by FFTs
    and elementwise work, so ``float32`` roughly halves both runtime and the
    memory held by gradient checkpoints. It is accurate enough for the
    amplitudes used here (see ``tests/test_solver.py::test_float32_matches_float64``),
    but ``float64`` remains available for reference runs.

    Must be called *before* constructing a :class:`~magnonic_nn.model.SpinWaveNetwork`,
    because :class:`magnumnp.Mesh` freezes the dtype of its coordinate tensors
    at construction time.
    """
    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
        "single": torch.float32,
        "double": torch.float64,
        torch.float32: torch.float32,
        torch.float64: torch.float64,
    }
    if dtype not in mapping:
        raise ValueError(f"unsupported precision {dtype!r}; use 'float32' or 'float64'")
    torch.set_default_dtype(mapping[dtype])
    return mapping[dtype]


def set_device(device):
    """Set the global torch default device (``'cpu'``, ``'cuda'``, ``'cuda:1'``, ...)."""
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available; falling back to CPU", stacklevel=2)
        device = torch.device("cpu")
    torch.set_default_device(device)
    return device


def get_device():
    """Return the current torch default device."""
    return torch.empty(0).device
