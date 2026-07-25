"""Output probes.

A probe integrates spin-wave intensity over a small region of the film, which
is what a pickup antenna or an inductive loop of that size would measure. The
paper reads out "the time-integrated wave intensity" over ~300 nm areas, and
those integrated intensities are the network's output vector.

The intensity convention follows the reference implementation:

1. subtract the equilibrium ``m0`` so only the wave contributes;
2. weight by ``Ms`` so a probe over a region with suppressed magnetisation
   reports less signal;
3. sum the transverse component over the probe area -- summing *before*
   squaring is what makes the probe phase sensitive, so waves arriving out of
   phase cancel rather than add;
4. square, and accumulate over time.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import MeshConfig

__all__ = ["Probe", "PointProbe", "DiskProbe", "linear_probe_array"]


class Probe(nn.Module):
    """Base class: weighted spatial integral of one magnetisation component.

    :param mask: ``(nx, ny)`` weights defining the probe area.
    :param component: which component of ``m`` is measured. Default 0 (x),
        which is transverse to the default ``+y`` bias and therefore carries
        the wave.
    """

    def __init__(self, mask: torch.Tensor, component: int = 0):
        super().__init__()
        if component not in (0, 1, 2):
            raise ValueError(f"component must be 0, 1 or 2, got {component}")
        self.component = int(component)
        self.register_buffer("mask", mask.to(torch.get_default_dtype()))

    @property
    def area_cells(self) -> int:
        return int((self.mask != 0).sum().item())

    def amplitude(self, dm_Ms: torch.Tensor) -> torch.Tensor:
        """Coherent (phase-sensitive) amplitude for one timestep.

        :param dm_Ms: ``(nx, ny, nz, 3)`` field of ``(m - m0) * Ms``.
        :returns: scalar tensor.
        """
        weighted = dm_Ms[..., self.component] * self.mask.reshape(*self.mask.shape, 1)
        return weighted.sum()

    def forward(self, dm_Ms: torch.Tensor) -> torch.Tensor:
        """Instantaneous intensity, i.e. squared coherent amplitude."""
        return self.amplitude(dm_Ms).pow(2)

    def coordinates(self):
        return torch.nonzero(self.mask, as_tuple=True)


class PointProbe(Probe):
    """Single-cell probe."""

    def __init__(self, mesh: MeshConfig, x: int, y: int, component: int = 0):
        nx, ny, _ = mesh.n
        mask = torch.zeros(nx, ny)
        mask[int(x), int(y)] = 1.0
        super().__init__(mask, component)
        self.x, self.y = int(x), int(y)


class DiskProbe(Probe):
    """Circular probe of radius ``r`` cells centred on ``(x, y)``.

    ``r = 2`` cells at 50 nm resolution is ~250 nm across, close to the 300 nm
    detector areas quoted in the paper.
    """

    def __init__(self, mesh: MeshConfig, x: int, y: int, r: float = 2.0, component: int = 0):
        nx, ny, _ = mesh.n
        ix = torch.arange(nx).reshape(nx, 1)
        iy = torch.arange(ny).reshape(1, ny)
        dist2 = (ix - int(x)) ** 2 + (iy - int(y)) ** 2
        mask = (dist2 <= r * r).to(torch.get_default_dtype())
        if mask.sum() == 0:
            raise ValueError(f"disk probe at ({x}, {y}) r={r} covers no cells")
        super().__init__(mask, component)
        self.x, self.y, self.r = int(x), int(y), float(r)


def linear_probe_array(
    mesh: MeshConfig,
    n_probes: int,
    x: int | None = None,
    r: float = 2.0,
    component: int = 0,
    margin: int = 0,
):
    """Evenly spaced column of :class:`DiskProbe` at fixed ``x``.

    Probes are placed at ``y = ny * (p + 1) / (n_probes + 1)``, i.e. spread over
    the film's width with a half-gap at each end, matching the reference
    implementation's layout.

    :param x: column index; defaults to 15 cells in from the far edge.
    :param margin: extra cells to keep clear at the top and bottom.
    """
    nx, ny, _ = mesh.n
    if x is None:
        x = nx - 15
    usable = ny - 2 * margin
    probes = []
    for p in range(n_probes):
        y = margin + int(usable * (p + 1) / (n_probes + 1))
        probes.append(DiskProbe(mesh, x, y, r=r, component=component))
    return probes
