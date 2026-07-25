"""Shared fixtures.

``torch``'s default dtype and device are process-global, and magnum.np freezes
the dtype into ``Mesh`` at construction time, so a test that switches precision
must put it back or it silently changes every test that runs after it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

import magnonic_nn as mnn


@pytest.fixture(autouse=True)
def _restore_torch_globals():
    dtype = torch.get_default_dtype()
    device = torch.empty(0).device
    threads = torch.get_num_threads()
    torch.set_num_threads(min(threads, 4))
    yield
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    torch.set_num_threads(threads)


@pytest.fixture
def f32():
    mnn.set_precision("float32")
    return torch.float32


@pytest.fixture
def f64():
    mnn.set_precision("float64")
    return torch.float64


@pytest.fixture
def tiny_cfg(f32):
    """Smallest configuration that still exercises the whole pipeline."""
    cfg = mnn.get_preset("tiny")
    cfg.mesh.nx = cfg.mesh.ny = 24
    cfg.material.abc_width = 4
    cfg.solver.timesteps = 24
    # Enough relax steps that the equilibrium has actually settled. Too few and
    # every probe reading carries a drive-independent relaxation transient --
    # see LLGRollout.relax.
    cfg.solver.relax_steps = 60
    return cfg


@pytest.fixture
def grad_cfg(f64):
    """Double precision and tiny, so finite differences are meaningful.

    Central differences on a float32 rollout are dominated by rounding: the
    step has to be large enough to beat 1e-7 relative noise, by which point the
    difference quotient is no longer measuring the derivative.
    """
    cfg = mnn.get_preset("tiny")
    cfg.mesh.nx = cfg.mesh.ny = 16
    cfg.material.abc_width = 3
    cfg.solver.timesteps = 12
    cfg.solver.relax_steps = 20
    cfg.solver.demag = False
    return cfg
