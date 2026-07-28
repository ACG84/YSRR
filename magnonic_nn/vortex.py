"""Micromagnetic vortex disk: the modal reservoir, replacing the Thiele model.

Why this exists. The Thiele description collapses a disk to two degrees of
freedom -- the core position -- so twelve disks carried 24 state variables and
a core reversal erased two of them outright. That is the whole explanation for
the measured lag-1 autocorrelation of -0.006: the *only* state variable was
precisely the one the spike destroyed, leaving nothing for memory to live in.

A real vortex disk carries dozens of radial and azimuthal magnon eigenmodes.
Three-magnon scattering above a power threshold redistributes energy among
them, each mode relaxes at its own linewidth (a timescale bank, rather than
the twelve hand-tuned dampings ``alpha_spread`` was faking), and a core
reversal does not erase the mode populations. That is the reservoir Körber et
al. demonstrate experimentally (Nat Commun 14, 3954, 2023): pulse sequences
"AB" and "BA" with *identical* average input spectra produce different output
spectra, which is memory and nonlinearity with no delay line at all.

The physics that forces the numerics: a vortex core is 10-20 nm, so cells must
be <= 5 nm, and at 5 nm the exchange field reaches ~1e6 A/m -- eigenfrequencies
to tens of GHz, needing a timestep near 1 ps rather than the 20 ps the 50 nm
film used. A 200 nm disk is only 40 cells across, so the mesh is cheap; the
timestep is what costs.

Geometry is imposed through ``Ms``: zero outside the disk, which is how
magnum.np represents a patterned element on a rectangular mesh.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .config import MU_0, MeshConfig, SolverConfig
from .solver import LLGRollout

__all__ = ["VortexConfig", "VortexDisk", "vortex_state", "disk_mask",
           "PortedVortexConfig", "PortedVortexDisk", "ported_mask", "port_alpha"]


@dataclass
class VortexConfig:
    """A permalloy vortex disk and the mesh that resolves it."""

    radius: float = 100e-9       # disk radius
    thickness: float = 20e-9
    dx: float = 5e-9             # <= 5 nm: the core is 10-20 nm across
    margin_cells: int = 4        # vacuum ring, keeps demag off the boundary
    Ms: float = 800e3            # permalloy
    A: float = 1.3e-11
    alpha: float = 0.008
    dt: float = 1e-12            # exchange at 5 nm sets this, not the drive
    polarity: int = 1            # core out-of-plane direction
    chirality: int = 1           # in-plane circulation sense
    core_width: float = 10e-9    # core profile scale

    @property
    def n_cells(self) -> int:
        return 2 * (int(round(self.radius / self.dx)) + self.margin_cells)

    def exchange_field(self) -> float:
        """Rough exchange field at this cell size, in A/m -- sets the timestep."""
        return 2 * self.A / (MU_0 * self.Ms * self.dx**2)

    def max_frequency(self) -> float:
        """Highest eigenfrequency the mesh can host, Hz."""
        from .dispersion import GAMMA
        return GAMMA * self.exchange_field() / (2 * math.pi)


def disk_mask(cfg: VortexConfig, device="cpu", dtype=torch.float64) -> torch.Tensor:
    """``(nx, ny, 1, 1)`` mask: 1 inside the disk, 0 in the surrounding vacuum."""
    n = cfg.n_cells
    idx = (torch.arange(n, device=device, dtype=dtype) - (n - 1) / 2) * cfg.dx
    X, Y = torch.meshgrid(idx, idx, indexing="ij")
    return ((X**2 + Y**2) <= cfg.radius**2).to(dtype).reshape(n, n, 1, 1)


def vortex_state(cfg: VortexConfig, device="cpu", dtype=torch.float64) -> torch.Tensor:
    """Analytic vortex ansatz, ``(nx, ny, 1, 3)``.

    In-plane curling with the chosen chirality, plus an out-of-plane core whose
    profile is the usual ``exp(-(r/w)^2)``. It is only a starting guess -- the
    relaxation below finds the true ground state -- but a good one keeps the
    relaxation short and, more importantly, lands in the intended
    (polarity, chirality) sector rather than a random one.
    """
    n = cfg.n_cells
    idx = (torch.arange(n, device=device, dtype=dtype) - (n - 1) / 2) * cfg.dx
    X, Y = torch.meshgrid(idx, idx, indexing="ij")
    r = torch.sqrt(X**2 + Y**2).clamp_min(1e-18)

    mz = cfg.polarity * torch.exp(-(r / cfg.core_width) ** 2)
    inplane = torch.sqrt((1 - mz**2).clamp_min(0.0))
    mx = -cfg.chirality * inplane * Y / r
    my = cfg.chirality * inplane * X / r

    m = torch.stack([mx, my, mz], dim=-1).reshape(n, n, 1, 3)
    return m * disk_mask(cfg, device, dtype)


class VortexDisk:
    """A relaxed vortex disk that can be driven and read out modally."""

    def __init__(self, cfg: VortexConfig, timesteps: int, device="cpu",
                 dtype=torch.float64, checkpoint: bool = False):
        self.cfg = cfg
        n = cfg.n_cells
        mesh = MeshConfig(nx=n, ny=n, nz=1, dx=cfg.dx, dy=cfg.dx, dz=cfg.thickness)
        solver = SolverConfig(dt=cfg.dt, timesteps=timesteps, checkpoint=checkpoint,
                              renormalize=True, demag=True)

        self.mask = disk_mask(cfg, device, dtype)
        alpha = torch.full((n, n, 1, 1), cfg.alpha, device=device, dtype=dtype)
        self.rollout = LLGRollout(mesh, solver, A=cfg.A, alpha=alpha, Ms_ref=cfg.Ms)
        self.rollout.set_Ms(cfg.Ms * self.mask)

        self.h_zero = torch.zeros(n, n, 1, 3, device=device, dtype=dtype)
        self.m0 = None

    def relax(self, steps: int = 4000, alpha_relax: float = 0.5) -> torch.Tensor:
        """Find the vortex ground state from the analytic ansatz."""
        m = vortex_state(self.cfg, self.h_zero.device, self.h_zero.dtype)
        self.m0 = self.rollout.relax(m, self.h_zero, steps, alpha_relax)
        return self.m0

    def core_position(self, m: torch.Tensor | None = None) -> tuple[float, float]:
        """Core centroid in metres, weighted by out-of-plane magnetisation.

        The natural observable for gyration, and the bridge back to the Thiele
        picture: it is the *only* thing that description could see, whereas the
        mode spectrum below is everything it could not.
        """
        m = self.m0 if m is None else m
        n = self.cfg.n_cells
        w = (m[:, :, 0, 2] * self.mask[:, :, 0, 0]).abs()
        idx = (torch.arange(n, device=m.device, dtype=m.dtype)
               - (n - 1) / 2) * self.cfg.dx
        total = w.sum().clamp_min(1e-30)
        return (float((w.sum(1) * idx).sum() / total),
                float((w.sum(0) * idx).sum() / total))


@dataclass
class PortedVortexConfig(VortexConfig):
    """A vortex disk with waveguide ports, patterned as one connected element.

    The guides are not a separate model: they are magnetic material attached
    to the disk, so spin waves propagate out through them by the same LLG the
    disk obeys. Mode selectivity is then geometry, not an assumption -- a
    guide supports only the modes its width and dispersion admit.

    The arrangement matters more than it first appears. A port at angle theta
    couples to azimuthal mode n with weight ``exp(i n theta)``, so N ports
    evenly spaced around the disk sample the edge magnetisation at N angles:
    their discrete Fourier transform IS the azimuthal mode decomposition, for
    ``|n| < N/2``. The modal readout that ``mode_spectrum`` computes
    numerically is what the port geometry does physically -- which is the
    whole point of encasing the disk rather than measuring it in software.
    """

    n_ports: int = 6
    guide_width: float = 40e-9     # supports the low-n modes; wider admits more
    guide_length: float = 150e-9
    absorb_frac: float = 0.35      # fraction of guide length used as an
                                   # absorbing taper, so the far end does not
                                   # reflect the wave back into the disk
    absorb_alpha: float = 0.5

    @property
    def n_cells(self) -> int:
        reach = self.radius + self.guide_length
        return 2 * (int(np.ceil(reach / self.dx)) + self.margin_cells)

    def port_angles(self):
        return [2 * math.pi * k / self.n_ports for k in range(self.n_ports)]


def ported_mask(cfg: PortedVortexConfig, device="cpu", dtype=torch.float64):
    """``(nx, ny, 1, 1)`` mask for the disk plus its radial guides."""
    n = cfg.n_cells
    idx = (torch.arange(n, device=device, dtype=dtype) - (n - 1) / 2) * cfg.dx
    X, Y = torch.meshgrid(idx, idx, indexing="ij")

    mask = (X**2 + Y**2) <= cfg.radius**2
    for th in cfg.port_angles():
        # guide-local coordinates: u along the guide, v across it
        u = X * math.cos(th) + Y * math.sin(th)
        v = -X * math.sin(th) + Y * math.cos(th)
        mask = mask | ((u >= 0) & (u <= cfg.radius + cfg.guide_length)
                       & (v.abs() <= cfg.guide_width / 2))
    return mask.to(dtype).reshape(n, n, 1, 1)


def port_alpha(cfg: PortedVortexConfig, device="cpu", dtype=torch.float64):
    """Damping field: uniform everywhere, ramped up at the guide far ends.

    Without this the guides are resonators rather than ports -- the wave
    reflects off the open end and returns, so what the tap measures is a
    standing wave set by guide length instead of what the disk emitted.
    """
    n = cfg.n_cells
    idx = (torch.arange(n, device=device, dtype=dtype) - (n - 1) / 2) * cfg.dx
    X, Y = torch.meshgrid(idx, idx, indexing="ij")
    r = torch.sqrt(X**2 + Y**2)

    outer = cfg.radius + cfg.guide_length
    start = outer - cfg.absorb_frac * cfg.guide_length
    ramp = ((r - start) / (outer - start)).clamp(0.0, 1.0) ** 2
    return (cfg.alpha + (cfg.absorb_alpha - cfg.alpha) * ramp).reshape(n, n, 1, 1)


class PortedVortexDisk(VortexDisk):
    """Vortex disk read out through its waveguide ports."""

    def __init__(self, cfg: PortedVortexConfig, timesteps: int, device="cpu",
                 dtype=torch.float64):
        self.cfg = cfg
        n = cfg.n_cells
        mesh = MeshConfig(nx=n, ny=n, nz=1, dx=cfg.dx, dy=cfg.dx, dz=cfg.thickness)
        solver = SolverConfig(dt=cfg.dt, timesteps=timesteps, checkpoint=False,
                             renormalize=True, demag=True)

        self.mask = ported_mask(cfg, device, dtype)
        self.disk_only = disk_mask(cfg, device, dtype)
        alpha = port_alpha(cfg, device, dtype) * self.mask
        self.rollout = LLGRollout(mesh, solver, A=cfg.A, alpha=alpha, Ms_ref=cfg.Ms)
        self.rollout.set_Ms(cfg.Ms * self.mask)

        self.h_zero = torch.zeros(n, n, 1, 3, device=device, dtype=dtype)
        self.m0 = None
        self._tap_masks = self._build_taps(device, dtype)

    def _build_taps(self, device, dtype):
        """One tap per guide, placed before the absorbing taper begins."""
        cfg = self.cfg
        n = cfg.n_cells
        idx = (torch.arange(n, device=device, dtype=dtype) - (n - 1) / 2) * cfg.dx
        X, Y = torch.meshgrid(idx, idx, indexing="ij")
        outer = cfg.radius + cfg.guide_length
        tap_r = outer - cfg.absorb_frac * cfg.guide_length - 2 * cfg.dx

        taps = []
        for th in cfg.port_angles():
            u = X * math.cos(th) + Y * math.sin(th)
            v = -X * math.sin(th) + Y * math.cos(th)
            sel = ((u - tap_r).abs() <= 1.5 * cfg.dx) & (v.abs() <= cfg.guide_width / 2)
            taps.append(sel.to(dtype))
        return torch.stack(taps)                 # (n_ports, nx, ny)

    def relax(self, steps: int = 4000, alpha_relax: float = 0.5) -> torch.Tensor:
        m = vortex_state(self.cfg, self.h_zero.device, self.h_zero.dtype)
        m = m * self.mask + torch.tensor(
            [0.0, 0.0, 1.0], device=m.device, dtype=m.dtype
        ) * (self.mask - self.disk_only)          # guides start along +z
        m = m / m.norm(dim=-1, keepdim=True).clamp_min(1e-12) * self.mask
        self.m0 = self.rollout.relax(m, self.h_zero, steps, alpha_relax)
        return self.m0

    def port_signals(self, m: torch.Tensor) -> torch.Tensor:
        """Mean out-of-plane deviation at each tap: what each port carries."""
        dm = (m - self.m0)[:, :, 0, 2]
        w = self._tap_masks
        return (w * dm).sum(dim=(1, 2)) / w.sum(dim=(1, 2)).clamp_min(1e-30)
