"""Scatterer parameterisations."""

from __future__ import annotations

import pytest
import torch

import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.geometry import binarize


def test_freeform_reduces_to_uniform_bias_at_init(f32):
    cfg = mnn.get_preset("tiny")
    geom = mnn.FreeFormFieldGeometry(cfg.mesh, cfg.fields, cfg.material)

    h = geom.static_field()
    assert h.shape == (*cfg.mesh.n, 3)
    b = MU_0 * h
    assert torch.allclose(b[..., 1], torch.full_like(b[..., 1], cfg.fields.B0), atol=1e-9)
    assert float(b[..., 0].abs().max()) == 0.0
    assert float(b[..., 2].abs().max()) == 0.0


def test_freeform_offset_is_bounded_by_B1(f32):
    """``tanh`` bounding must keep the local field inside ``B0 +/- B1``.

    Without a bound the optimiser happily converges on local fields no magnet
    could produce, and the result looks fine until someone tries to build it.
    """
    cfg = mnn.get_preset("tiny")
    geom = mnn.FreeFormFieldGeometry(cfg.mesh, cfg.fields, cfg.material, bound="tanh")
    with torch.no_grad():
        geom.rho.copy_(torch.randn_like(geom.rho) * 50)  # absurdly large request

    b = MU_0 * geom.static_field()[..., 1]
    assert float(b.max()) <= cfg.fields.B0 + cfg.fields.B1 + 1e-9
    assert float(b.min()) >= cfg.fields.B0 - cfg.fields.B1 - 1e-9


def test_unbounded_mode_is_unbounded(f32):
    """``bound='none'`` reproduces the reference behaviour, warts included."""
    cfg = mnn.get_preset("tiny")
    geom = mnn.FreeFormFieldGeometry(cfg.mesh, cfg.fields, cfg.material, bound="none")
    with torch.no_grad():
        geom.rho.fill_(10.0)
    b = MU_0 * geom.static_field()[..., 1]
    assert float(b.max()) > cfg.fields.B0 + 5 * cfg.fields.B1


def test_design_mask_freezes_cells_outside_it(f32):
    cfg = mnn.get_preset("tiny")
    nx, ny, _ = cfg.mesh.n
    mask = torch.zeros(nx, ny)
    mask[5:10, 5:10] = 1.0

    geom = mnn.FreeFormFieldGeometry(cfg.mesh, cfg.fields, cfg.material, design_mask=mask)
    with torch.no_grad():
        geom.rho.fill_(1.0)

    b = MU_0 * geom.static_field()[:, :, 0, 1]
    inside = b[5:10, 5:10]
    outside = b.clone()
    outside[5:10, 5:10] = cfg.fields.B0

    assert float((inside - cfg.fields.B0).abs().min()) > 0
    assert torch.allclose(outside, torch.full_like(outside, cfg.fields.B0), atol=1e-9)


def test_masked_cells_receive_no_gradient(f32):
    cfg = mnn.get_preset("tiny")
    nx, ny, _ = cfg.mesh.n
    mask = torch.zeros(nx, ny)
    mask[5:10, 5:10] = 1.0

    geom = mnn.FreeFormFieldGeometry(cfg.mesh, cfg.fields, cfg.material, design_mask=mask)
    geom.static_field().sum().backward()

    grad = geom.rho.grad
    assert float(grad[mask == 0].abs().max()) == 0.0
    assert float(grad[mask > 0].abs().max()) > 0.0


def test_ms_geometry_scales_saturation_magnetisation(f32):
    cfg = mnn.get_preset("tiny")
    geom = mnn.MsGeometry(cfg.mesh, cfg.fields, cfg.material)

    assert torch.allclose(geom.Ms_field(), torch.full_like(geom.Ms_field(), cfg.material.Ms))

    with torch.no_grad():
        geom.rho.fill_(0.5)
    assert torch.allclose(geom.Ms_field(), torch.full_like(geom.Ms_field(), 0.5 * cfg.material.Ms))


def test_ms_geometry_clamps_to_physical_range(f32):
    """``Ms`` below zero is meaningless; the clamp must hold in the forward pass."""
    cfg = mnn.get_preset("tiny")
    geom = mnn.MsGeometry(cfg.mesh, cfg.fields, cfg.material, bound="clamp")
    with torch.no_grad():
        geom.rho.fill_(-3.0)
    assert float(geom.Ms_field().min()) >= 0.0

    with torch.no_grad():
        geom.rho.fill_(7.0)
    assert float(geom.Ms_field().max()) <= cfg.material.Ms + 1e-6


def test_binarize_is_sign_forward_identity_backward(f32):
    x = torch.tensor([-2.0, -0.1, 0.4, 3.0], requires_grad=True)
    y = binarize(x)
    assert torch.equal(y, torch.tensor([-1.0, -1.0, 1.0, 1.0]))

    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_nanomagnet_pattern_is_binary_valued(f32):
    cfg = mnn.get_preset("tiny")
    cfg.mesh.nx = cfg.mesh.ny = 48
    geom = mnn.NanomagnetArrayGeometry(
        cfg.mesh, cfg.fields, cfg.material,
        lattice_origin=8, lattice_pitch=4, magnet_size=1, z_offset_cells=5,
    )
    pattern = geom.design_image()
    assert set(torch.unique(pattern).tolist()) <= {-1.0, 0.0, 1.0}
    assert int((pattern != 0).sum()) == geom.nmag[0] * geom.nmag[1]


def test_nanomagnet_stray_field_flips_with_the_design(f32):
    """Reversing every nanomagnet must reverse the stray field exactly."""
    cfg = mnn.get_preset("tiny")
    cfg.mesh.nx = cfg.mesh.ny = 48
    geom = mnn.NanomagnetArrayGeometry(
        cfg.mesh, cfg.fields, cfg.material,
        lattice_origin=8, lattice_pitch=4, magnet_size=2, z_offset_cells=5,
    )
    with torch.no_grad():
        geom.rho.fill_(1.0)
    up = geom.stray_field_image().clone()

    with torch.no_grad():
        geom.rho.fill_(-1.0)
    down = geom.stray_field_image().clone()

    assert torch.allclose(up, -down, atol=1e-12)
    assert float(up.abs().max()) > 0


def test_nanomagnet_stray_field_decays_with_stand_off(f32):
    """Further away is weaker -- a basic check that the offset kernel is real."""
    cfg = mnn.get_preset("tiny")
    cfg.mesh.nx = cfg.mesh.ny = 48

    def peak(z_off):
        geom = mnn.NanomagnetArrayGeometry(
            cfg.mesh, cfg.fields, cfg.material,
            lattice_origin=8, lattice_pitch=6, magnet_size=2, z_offset_cells=z_off,
        )
        with torch.no_grad():
            geom.rho.fill_(1.0)
        return float(geom.stray_field_image().abs().max())

    assert peak(2) > peak(5) > peak(12)


def test_nanomagnet_stray_field_is_a_sensible_magnitude(f32):
    """A CoPt dot a few tens of nm away should produce a few mT, not micro- or kilo-tesla."""
    cfg = mnn.get_preset("tiny")
    cfg.mesh.nx = cfg.mesh.ny = 48
    geom = mnn.NanomagnetArrayGeometry(
        cfg.mesh, cfg.fields, cfg.material,
        lattice_origin=8, lattice_pitch=4, magnet_size=2, z_offset_cells=3,
        Ms_magnet=723e3,
    )
    with torch.no_grad():
        geom.rho.fill_(1.0)
    peak_mT = float(geom.stray_field_image().abs().max()) * 1e3
    assert 0.05 < peak_mT < 500


def test_interior_mask_excludes_the_absorbing_layer(f32):
    cfg = mnn.get_preset("tiny")
    mask = mnn.interior_mask(cfg.mesh, cfg.material)
    w = cfg.material.abc_width
    assert float(mask[:w, :].max()) == 0.0
    assert float(mask[-w:, :].max()) == 0.0
    assert float(mask[w:-w, w:-w].min()) == 1.0


def test_interior_mask_rejects_an_over_thick_boundary(f32):
    cfg = mnn.get_preset("tiny")
    cfg.mesh.nx = cfg.mesh.ny = 8
    cfg.material.abc_width = 6
    with pytest.raises(ValueError, match="no design region"):
        mnn.interior_mask(cfg.mesh, cfg.material)


def test_build_geometry_dispatches_on_config(f32):
    cfg = mnn.get_preset("tiny")
    for name, cls in [
        ("freeform", mnn.FreeFormFieldGeometry),
        ("ms", mnn.MsGeometry),
        ("nanomagnets", mnn.NanomagnetArrayGeometry),
    ]:
        cfg.geometry = name
        assert isinstance(mnn.build_geometry(cfg), cls)

    cfg.geometry = "nope"
    with pytest.raises(KeyError):
        mnn.build_geometry(cfg)
