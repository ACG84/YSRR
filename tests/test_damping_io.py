"""Absorbing boundary, sources and probes."""

from __future__ import annotations

import pytest
import torch

import magnonic_nn as mnn
from magnonic_nn.config import MU_0
from magnonic_nn.sources import _bresenham


# ------------------------------------------------------------------- damping


def test_damping_profile_shape_and_bulk_value(f32):
    cfg = mnn.get_preset("tiny")
    alpha = mnn.absorbing_damping(cfg.mesh, cfg.material)

    assert alpha.shape == (*cfg.mesh.n, 1)
    mid = cfg.mesh.nx // 2, cfg.mesh.ny // 2
    assert float(alpha[mid[0], mid[1], 0, 0]) == pytest.approx(cfg.material.alpha)


def test_damping_is_maximal_at_the_edge(f32):
    cfg = mnn.get_preset("tiny")
    alpha = mnn.absorbing_damping(cfg.mesh, cfg.material)[:, :, 0, 0]
    assert float(alpha[0, :].min()) == pytest.approx(cfg.material.alpha_max)
    assert float(alpha[-1, :].min()) == pytest.approx(cfg.material.alpha_max)
    assert float(alpha[:, 0].min()) == pytest.approx(cfg.material.alpha_max)


def test_damping_decreases_monotonically_inwards(f32):
    """A step in damping is itself a reflecting discontinuity; the taper must be smooth."""
    cfg = mnn.get_preset("tiny")
    alpha = mnn.absorbing_damping(cfg.mesh, cfg.material)[:, cfg.mesh.ny // 2, 0, 0]
    half = alpha[: cfg.mesh.nx // 2]
    assert all(b <= a + 1e-12 for a, b in zip(half, half[1:]))


def test_no_damping_layer_on_a_periodic_axis(f32):
    """A periodic direction has no edge; damping it would attenuate a wrap-around wave."""
    cfg = mnn.get_preset("tiny")
    cfg.mesh.pbc = (0, 1, 0)
    alpha = mnn.absorbing_damping(cfg.mesh, cfg.material)[:, :, 0, 0]

    interior_x = cfg.mesh.nx // 2
    assert float(alpha[interior_x, :].max()) == pytest.approx(cfg.material.alpha)
    assert float(alpha[0, :].min()) == pytest.approx(cfg.material.alpha_max)


def test_zero_width_boundary_is_uniform(f32):
    cfg = mnn.get_preset("tiny")
    cfg.material.abc_width = 0
    alpha = mnn.absorbing_damping(cfg.mesh, cfg.material)
    assert torch.allclose(alpha, torch.full_like(alpha, cfg.material.alpha))


# ------------------------------------------------------------------- sources


def test_bresenham_endpoints_and_continuity():
    pts = list(_bresenham(0, 0, 5, 3))
    assert pts[0] == (0, 0)
    assert pts[-1] == (5, 3)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        assert max(abs(x1 - x0), abs(y1 - y0)) == 1


def test_line_source_spans_the_film(f32):
    cfg = mnn.get_preset("tiny")
    ny = cfg.mesh.ny
    src = mnn.LineSource(cfg.mesh, cfg.fields, 5, 0, 5, ny - 1)
    assert src.n_cells == ny
    xs, ys = src.coordinates()
    assert set(xs.tolist()) == {5}


def test_source_field_amplitude_and_direction(f32):
    """Drive is applied on the source cells only, along ``drive_axis``, at ``Bt``."""
    cfg = mnn.get_preset("tiny")
    src = mnn.PointSource(cfg.mesh, cfg.fields, 6, 7)

    field = src.field(torch.tensor(1.0), nz=cfg.mesh.nz)
    assert field.shape == (*cfg.mesh.n, 3)

    b = MU_0 * field
    assert float(b[6, 7, 0, cfg.fields.drive_axis]) == pytest.approx(cfg.fields.Bt, rel=1e-6)
    for axis in range(3):
        if axis != cfg.fields.drive_axis:
            assert float(b[..., axis].abs().max()) == 0.0

    b[6, 7, 0, cfg.fields.drive_axis] = 0.0
    assert float(b.abs().max()) == 0.0


def test_source_field_is_linear_in_the_signal(f32):
    cfg = mnn.get_preset("tiny")
    src = mnn.PointSource(cfg.mesh, cfg.fields, 6, 7)
    a = src.field(torch.tensor(0.3), nz=1)
    b = src.field(torch.tensor(0.6), nz=1)
    assert torch.allclose(2 * a, b, atol=1e-12)


def test_antenna_taper_is_symmetric_and_widest_in_the_middle(f32):
    cfg = mnn.get_preset("tiny")
    ant = mnn.AntennaSource(cfg.mesh, cfg.fields, x=10, width=5, taper=True)
    column = ant.mask[8:13, 0]
    assert float(column[2]) == pytest.approx(float(column.max()))
    assert float(column[0]) == pytest.approx(float(column[4]), abs=1e-6)


def test_source_rejects_a_bad_axis(f32):
    cfg = mnn.get_preset("tiny")
    with pytest.raises(ValueError, match="axis must be"):
        mnn.PointSource(cfg.mesh, cfg.fields, 3, 3, axis=7)


# -------------------------------------------------------------------- probes


def test_disk_probe_covers_a_disk(f32):
    cfg = mnn.get_preset("tiny")
    probe = mnn.DiskProbe(cfg.mesh, 12, 12, r=2.0)
    xs, ys = probe.coordinates()
    for x, y in zip(xs.tolist(), ys.tolist()):
        assert (x - 12) ** 2 + (y - 12) ** 2 <= 4
    assert probe.area_cells == 13  # the r=2 discrete disk


def test_probe_sums_before_squaring(f32):
    """Coherent readout: two out-of-phase contributions must cancel, not add.

    This is what makes the probe phase sensitive, and therefore what lets the
    trained interference pattern steer energy at all.
    """
    cfg = mnn.get_preset("tiny")
    probe = mnn.DiskProbe(cfg.mesh, 12, 12, r=1.0)

    field = torch.zeros(*cfg.mesh.n, 3)
    cells = probe.coordinates()
    xs, ys = cells[0].tolist(), cells[1].tolist()

    # equal and opposite contributions on two cells
    field[xs[0], ys[0], 0, 0] = 1.0
    field[xs[1], ys[1], 0, 0] = -1.0
    assert float(probe(field)) == pytest.approx(0.0, abs=1e-12)

    # in phase, they add and then square
    field[xs[1], ys[1], 0, 0] = 1.0
    assert float(probe(field)) == pytest.approx(4.0, rel=1e-9)


def test_probe_reads_the_configured_component(f32):
    cfg = mnn.get_preset("tiny")
    probe = mnn.PointProbe(cfg.mesh, 5, 5, component=2)
    field = torch.zeros(*cfg.mesh.n, 3)
    field[5, 5, 0, 0] = 3.0
    assert float(probe(field)) == 0.0
    field[5, 5, 0, 2] = 3.0
    assert float(probe(field)) == pytest.approx(9.0)


def test_probe_array_is_ordered_and_inside_the_film(f32):
    cfg = mnn.get_preset("tiny")
    probes = mnn.linear_probe_array(cfg.mesh, 5, x=18, r=1.0, margin=cfg.material.abc_width)
    ys = [p.y for p in probes]
    assert ys == sorted(ys)
    assert all(cfg.material.abc_width <= y < cfg.mesh.ny - cfg.material.abc_width for y in ys)
    assert all(p.x == 18 for p in probes)


def test_empty_disk_probe_is_rejected(f32):
    cfg = mnn.get_preset("tiny")
    with pytest.raises(ValueError, match="covers no cells"):
        mnn.DiskProbe(cfg.mesh, 200, 200, r=1.0)
