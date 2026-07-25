"""Spin-wave sources.

A source is a set of cells plus a direction: at every timestep the drive
signal, scaled by :attr:`FieldConfig.Bt`, is added to the external field on
those cells along that direction. Physically this stands in for the Oersted
field of a microwave antenna patterned on the film.

Sources hold *where* and *how strongly* to drive; the waveform itself is passed
to the model at call time, so the same network can be driven with a tone, a
chirp, or a vowel without rebuilding anything.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import MU_0, FieldConfig, MeshConfig

__all__ = ["Source", "PointSource", "LineSource", "AntennaSource"]


class Source(nn.Module):
    """Base class: a boolean cell mask driven along one Cartesian component.

    :param mask: ``(nx, ny)`` array of weights. Non-binary weights are allowed
        and act as an apodisation of the antenna.
    :param axis: component driven (0=x, 1=y, 2=z).
    :param amplitude_T: peak drive field in tesla, converted internally to A/m.
    """

    def __init__(self, mask: torch.Tensor, axis: int, amplitude_T: float):
        super().__init__()
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1 or 2, got {axis}")
        self.axis = int(axis)
        self.amplitude_T = float(amplitude_T)
        self.register_buffer("mask", mask.to(torch.get_default_dtype()))
        self.register_buffer("amplitude", torch.tensor(amplitude_T / MU_0))

    @property
    def n_cells(self) -> int:
        return int((self.mask != 0).sum().item())

    def field(self, value: torch.Tensor, nz: int = 1) -> torch.Tensor:
        """Return the drive field for scalar signal ``value``.

        :returns: ``(nx, ny, nz, 3)`` tensor of A/m, zero everywhere except on
            the source cells.
        """
        nx, ny = self.mask.shape
        out = torch.zeros(nx, ny, nz, 3, dtype=self.mask.dtype, device=self.mask.device)
        out[..., self.axis] = (self.amplitude * value * self.mask).reshape(nx, ny, 1)
        return out

    def coordinates(self):
        """Indices of the driven cells, for plotting."""
        return torch.nonzero(self.mask, as_tuple=True)


class PointSource(Source):
    """Single-cell source. The magnonic analogue of a point emitter."""

    def __init__(self, mesh: MeshConfig, fields: FieldConfig, x: int, y: int, axis: int | None = None):
        nx, ny, _ = mesh.n
        mask = torch.zeros(nx, ny)
        mask[int(x), int(y)] = 1.0
        super().__init__(mask, fields.drive_axis if axis is None else axis, fields.Bt)
        self.x, self.y = int(x), int(y)


class LineSource(Source):
    """Straight line of cells between two grid points (Bresenham).

    A line perpendicular to the propagation direction launches a nearly plane
    wave, which is what the reference implementation's ``WaveLineSource`` does.
    """

    def __init__(
        self,
        mesh: MeshConfig,
        fields: FieldConfig,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        axis: int | None = None,
    ):
        nx, ny, _ = mesh.n
        mask = torch.zeros(nx, ny)
        for x, y in _bresenham(int(x0), int(y0), int(x1), int(y1)):
            if 0 <= x < nx and 0 <= y < ny:
                mask[x, y] = 1.0
        super().__init__(mask, fields.drive_axis if axis is None else axis, fields.Bt)
        self.endpoints = (int(x0), int(y0), int(x1), int(y1))


class AntennaSource(Source):
    """Finite-width stripe antenna, optionally apodised.

    A stripe of width ``w`` acts as a low-pass filter in wavenumber: it couples
    efficiently only to wavelengths longer than about ``2w``. Use it when you
    want to suppress short-wavelength exchange modes that a single-cell-wide
    line source would otherwise excite.

    :param taper: if ``True``, weight the stripe with a Hann window across its
        width, which cuts the side lobes of the excited ``k`` spectrum.
    """

    def __init__(
        self,
        mesh: MeshConfig,
        fields: FieldConfig,
        x: int,
        width: int = 3,
        y_range: tuple[int, int] | None = None,
        axis: int | None = None,
        taper: bool = True,
    ):
        nx, ny, _ = mesh.n
        mask = torch.zeros(nx, ny)
        y0, y1 = (0, ny) if y_range is None else y_range
        x0 = int(x) - width // 2
        xs = [xi for xi in range(x0, x0 + width) if 0 <= xi < nx]
        if taper and width > 1:
            # periodic=False: torch's default periodic window is asymmetric (it
            # drops the last sample so the window tiles seamlessly for FFT use),
            # which would put the antenna's peak weight off-centre and steer the
            # beam a fraction of a degree off axis.
            window = torch.hann_window(width + 2, periodic=False)[1:-1]
        else:
            window = torch.ones(width)
        for k, xi in enumerate(xs):
            mask[xi, y0:y1] = window[k]
        super().__init__(mask, fields.drive_axis if axis is None else axis, fields.Bt)


def _bresenham(x0: int, y0: int, x1: int, y1: int):
    """Integer line rasterisation, yielding ``(x, y)`` pairs inclusive of both ends."""
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        yield x, y
        if x == x1 and y == y1:
            return
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
