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
           "PortedVortexConfig", "PortedVortexDisk", "ported_mask", "port_alpha",
           "CoupledPortedConfig", "CoupledPortedArray", "coupled_ported_mask"]


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
    # 80 nm, not 40. At the disk's own 20 nm thickness a 40 nm guide is
    # evanescent below ~14 GHz (Q = decay/wavelength runs 10.0 at 14 GHz but
    # is unmeasurable at 10); 80 nm brings the cutoff down to 11 GHz with
    # Q = 8.0, and the AB/BA discrimination driven at 11.7/13.7 GHz scores 15.0
    # in the >=11 GHz passband against 1.2 for the old 6/9 GHz drive. Six of
    # these subtend 46 degrees each, 275 of 360 -- they fit, and measured mode
    # selectivity is slightly BETTER than at 40 nm (|n| concentration
    # 0.700/0.829/0.864 against 0.549/0.855/0.795), so the arc-averaging
    # penalty the width was originally chosen to avoid does not bite here.
    guide_width: float = 80e-9
    # Short on purpose. The port is a near-field tap, not a transport link: it
    # only has to sample the edge so the cross-port DFT can do the
    # decomposition, and over 150 nm even an evanescent mode delivers ~30%.
    # Transport between disks is a separate problem with a separate budget.
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


@dataclass
class CoupledArrayConfig(VortexConfig):
    """Two ported vortex disks joined by a shared waveguide.

    The open risk after the single-disk results: every measurement so far is
    one disk in isolation, and the Thiele array failed precisely when disks
    were coupled -- strong coupling near a threshold produced chaos rather
    than computation. A guide that carries signal from A to B also carries it
    back, so the coupled geometry has to be checked for stability before any
    task result from it means anything.

    Geometry: disks at +-separation/2 along x, joined by a guide of
    ``link_width``, each with one outward port for readout. Absorbing tapers
    sit only at the OUTWARD ends -- the link between the disks is deliberately
    lossless, since attenuating it would hide the very instability this is
    built to look for.
    """

    separation: float = 500e-9     # centre to centre
    link_width: float = 40e-9      # the APERTURE between disks; 0 disconnects
                                   # them entirely, which is the differential
                                   # control for how much the link actually
                                   # carries versus the dipolar field
    out_width: float = 40e-9       # readout guides, held fixed while the
                                   # aperture is swept
    out_length: float = 120e-9
    absorb_frac: float = 0.4
    absorb_alpha: float = 0.5

    @property
    def grid(self) -> tuple[int, int]:
        half_x = self.separation / 2 + self.radius + self.out_length
        nx = 2 * (int(np.ceil(half_x / self.dx)) + self.margin_cells)
        ny = 2 * (int(np.ceil(self.radius / self.dx)) + self.margin_cells)
        return nx, ny

    def centres(self):
        return [(-self.separation / 2, 0.0), (self.separation / 2, 0.0)]


def coupled_mask(cfg: CoupledArrayConfig, device="cpu", dtype=torch.float64):
    """Two disks, the link between them, and one outward port each."""
    nx, ny = cfg.grid
    x = (torch.arange(nx, device=device, dtype=dtype) - (nx - 1) / 2) * cfg.dx
    y = (torch.arange(ny, device=device, dtype=dtype) - (ny - 1) / 2) * cfg.dx
    X, Y = torch.meshgrid(x, y, indexing="ij")

    mask = torch.zeros_like(X, dtype=torch.bool)
    for cx, cy in cfg.centres():
        mask = mask | (((X - cx) ** 2 + (Y - cy) ** 2) <= cfg.radius**2)
    # link along x between the two disks -- omitted entirely at zero aperture
    if cfg.link_width > 0:
        mask = mask | ((X.abs() <= cfg.separation / 2)
                       & (Y.abs() <= cfg.link_width / 2))
    # outward readout guides, independent of the aperture
    outer = cfg.separation / 2 + cfg.radius + cfg.out_length
    mask = mask | ((X.abs() >= cfg.separation / 2) & (X.abs() <= outer)
                   & (Y.abs() <= cfg.out_width / 2))
    return mask.to(dtype).reshape(nx, ny, 1, 1)


def coupled_alpha(cfg: CoupledArrayConfig, device="cpu", dtype=torch.float64):
    """Damping ramped only at the OUTWARD ends; the link stays lossless."""
    nx, ny = cfg.grid
    x = (torch.arange(nx, device=device, dtype=dtype) - (nx - 1) / 2) * cfg.dx
    y = (torch.arange(ny, device=device, dtype=dtype) - (ny - 1) / 2) * cfg.dx
    X, _ = torch.meshgrid(x, y, indexing="ij")

    outer = cfg.separation / 2 + cfg.radius + cfg.out_length
    start = outer - cfg.absorb_frac * cfg.out_length
    ramp = ((X.abs() - start) / (outer - start)).clamp(0.0, 1.0) ** 2
    return (cfg.alpha + (cfg.absorb_alpha - cfg.alpha) * ramp).reshape(nx, ny, 1, 1)


class CoupledDiskArray:
    """Two guide-coupled vortex disks, driven and read at either end."""

    def __init__(self, cfg: CoupledArrayConfig, timesteps: int, device="cpu",
                 dtype=torch.float64):
        self.cfg = cfg
        nx, ny = cfg.grid
        mesh = MeshConfig(nx=nx, ny=ny, nz=1, dx=cfg.dx, dy=cfg.dx, dz=cfg.thickness)
        solver = SolverConfig(dt=cfg.dt, timesteps=timesteps, checkpoint=False,
                              renormalize=True, demag=True)

        self.mask = coupled_mask(cfg, device, dtype)
        alpha = coupled_alpha(cfg, device, dtype) * self.mask
        self.rollout = LLGRollout(mesh, solver, A=cfg.A, alpha=alpha, Ms_ref=cfg.Ms)
        self.rollout.set_Ms(cfg.Ms * self.mask)
        self.h_zero = torch.zeros(nx, ny, 1, 3, device=device, dtype=dtype)
        self.m0 = None

        x = (torch.arange(nx, device=device, dtype=dtype) - (nx - 1) / 2) * cfg.dx
        y = (torch.arange(ny, device=device, dtype=dtype) - (ny - 1) / 2) * cfg.dx
        X, Y = torch.meshgrid(x, y, indexing="ij")
        self.disk_masks = torch.stack([
            (((X - cx) ** 2 + (Y - cy) ** 2) <= cfg.radius**2).to(dtype)
            for cx, cy in cfg.centres()])

    def relax(self, steps: int = 900, alpha_relax: float = 0.5):
        cfg = self.cfg
        nx, ny = cfg.grid
        x = (torch.arange(nx, dtype=self.h_zero.dtype) - (nx - 1) / 2) * cfg.dx
        y = (torch.arange(ny, dtype=self.h_zero.dtype) - (ny - 1) / 2) * cfg.dx
        X, Y = torch.meshgrid(x, y, indexing="ij")

        m = torch.zeros(nx, ny, 1, 3, dtype=self.h_zero.dtype)
        m[:, :, 0, 2] = 1.0                       # guides start out of plane
        for k, (cx, cy) in enumerate(cfg.centres()):
            dX, dY = X - cx, Y - cy
            r = torch.sqrt(dX**2 + dY**2).clamp_min(1e-18)
            mz = cfg.polarity * torch.exp(-(r / cfg.core_width) ** 2)
            ip = torch.sqrt((1 - mz**2).clamp_min(0.0))
            sel = self.disk_masks[k] > 0
            m[:, :, 0, 0] = torch.where(sel, -cfg.chirality * ip * dY / r, m[:, :, 0, 0])
            m[:, :, 0, 1] = torch.where(sel, cfg.chirality * ip * dX / r, m[:, :, 0, 1])
            m[:, :, 0, 2] = torch.where(sel, mz, m[:, :, 0, 2])
        m = m / m.norm(dim=-1, keepdim=True).clamp_min(1e-12) * self.mask
        self.m0 = self.rollout.relax(m, self.h_zero, steps, alpha_relax)
        return self.m0

    def disk_energy(self, m: torch.Tensor) -> torch.Tensor:
        """Deviation energy inside each disk: (2,)."""
        dm = ((m - self.m0) ** 2).sum(-1)[:, :, 0]
        w = self.disk_masks
        return (w * dm).sum(dim=(1, 2)) / w.sum(dim=(1, 2)).clamp_min(1e-30)


@dataclass
class CoupledPortedConfig(PortedVortexConfig):
    """Two disks, each with a FULL port complement, joined through one port.

    The previous coupled build gave each disk a single outward stub and was
    violently unstable -- lambda +5.5/ns, independent of separation, because
    the instability was never about separation. A driven vortex needs enough
    open aperture to shed the energy pumped into it: measured at 30 mT, one
    port gives +6.494/ns, two give +4.506, six give -0.668.

    So here each disk keeps its six ports and the link is one of them, joined
    tip to tip. The stability budget is met per disk, and the coupling
    question becomes answerable for the first time.

    CHIRALITY MATTERS, and not the way it first appears. With both disks at
    the same chirality the link nucleates a Bloch-type 180 degree domain wall
    at its midpoint -- measured directly along the centreline: mx runs +1.00
    across the left half, reverses through zero, runs -1.00 across the right,
    with mz peaking at +0.80 at the crossing. Each vortex drives the strip's
    magnetisation in the same ROTATIONAL sense, which means one pushes it
    toward +x and the other toward -x; the strip cannot satisfy both.

    The wall is real but it is NOT what destabilised the earlier runs. With
    1800 relaxation steps instead of 900, the same walled configuration
    measures lambda -0.171/ns where it previously measured +0.422: the
    instability was unconverged relaxation, the state still settling when the
    drive began, read as exponential divergence. ``relax()`` warns only below
    |dm| = 0.02, which is ample for probe readouts and far too loose for a
    Lyapunov measurement. Every lambda from a coupled run at 900 steps --
    the separation, aperture and link-loss sweeps -- is suspect for that
    reason.

    Set ``chirality_b = -chirality`` to remove it anyway: opposite chirality
    makes both disks drive the strip the same way in the lab frame, so it
    magnetises uniformly. Measured at 30 mT with 1800 relax steps:

        same chirality      wall, mz peak 0.98   lambda -0.171   B/A 0.00275
        opposite chirality  no wall, mz 0.00     lambda -0.292   B/A 0.00380

    Better on both axes -- more stable and 38% better coupled -- so it is the
    configuration to build, even though the wall was not the instability.
    """

    chirality_b: int | None = None   # chirality of the second disk; None
                                     # copies the first, which nucleates the
                                     # wall described above

    separation: float = 700e-9     # centre to centre
    link_width: float = 40e-9
    link_alpha: float | None = None   # damping inside the link corridor.
                                      # None keeps the base alpha, i.e. a
                                      # lossless feedback path -- which is
                                      # what measured lambda +0.422/ns. Raising
                                      # it terminates the loop, at the cost of
                                      # attenuating the coupling it provides.

    @property
    def grid(self) -> tuple[int, int]:
        reach_x = self.separation / 2 + self.radius + self.guide_length
        reach_y = self.radius + self.guide_length
        return (2 * (int(np.ceil(reach_x / self.dx)) + self.margin_cells),
                2 * (int(np.ceil(reach_y / self.dx)) + self.margin_cells))

    def centres(self):
        return [(-self.separation / 2, 0.0), (self.separation / 2, 0.0)]


def _radial_guides(X, Y, cx, cy, cfg):
    """Disk at (cx, cy) plus its radial guides."""
    dX, dY = X - cx, Y - cy
    m = (dX**2 + dY**2) <= cfg.radius**2
    for th in cfg.port_angles():
        u = dX * math.cos(th) + dY * math.sin(th)
        v = -dX * math.sin(th) + dY * math.cos(th)
        m = m | ((u >= 0) & (u <= cfg.radius + cfg.guide_length)
                 & (v.abs() <= cfg.guide_width / 2))
    return m


def coupled_ported_mask(cfg: CoupledPortedConfig, device="cpu", dtype=torch.float64):
    nx, ny = cfg.grid
    x = (torch.arange(nx, device=device, dtype=dtype) - (nx - 1) / 2) * cfg.dx
    y = (torch.arange(ny, device=device, dtype=dtype) - (ny - 1) / 2) * cfg.dx
    X, Y = torch.meshgrid(x, y, indexing="ij")
    m = torch.zeros_like(X, dtype=torch.bool)
    for cx, cy in cfg.centres():
        m = m | _radial_guides(X, Y, cx, cy, cfg)
    # bridge whatever gap remains between the facing guide tips
    m = m | ((X.abs() <= cfg.separation / 2) & (Y.abs() <= cfg.link_width / 2))
    return m.to(dtype).reshape(nx, ny, 1, 1)


def coupled_ported_alpha(cfg: CoupledPortedConfig, device="cpu", dtype=torch.float64):
    """Absorb at the OUTWARD guide ends; the link corridor stays lossless."""
    nx, ny = cfg.grid
    x = (torch.arange(nx, device=device, dtype=dtype) - (nx - 1) / 2) * cfg.dx
    y = (torch.arange(ny, device=device, dtype=dtype) - (ny - 1) / 2) * cfg.dx
    X, Y = torch.meshgrid(x, y, indexing="ij")

    outer = cfg.radius + cfg.guide_length
    start = outer - cfg.absorb_frac * cfg.guide_length
    ramp = torch.ones_like(X)
    for cx, cy in cfg.centres():
        r = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        ramp = torch.minimum(ramp, ((r - start) / (outer - start)).clamp(0.0, 1.0) ** 2)
    # The link is handled separately: the outward tapers must not spill into it
    # (that would attenuate the coupling under test), but the link may carry a
    # deliberate loss of its own to terminate the feedback loop.
    link = (X.abs() <= cfg.separation / 2) & (Y.abs() <= cfg.link_width / 2)
    ramp = torch.where(link, torch.zeros_like(ramp), ramp)
    alpha = cfg.alpha + (cfg.absorb_alpha - cfg.alpha) * ramp
    if cfg.link_alpha is not None:
        alpha = torch.where(link, torch.full_like(alpha, cfg.link_alpha), alpha)
    return alpha.reshape(nx, ny, 1, 1)


class CoupledPortedArray:
    """Two fully-ported vortex disks joined through one port each."""

    def __init__(self, cfg: CoupledPortedConfig, timesteps: int, device="cpu",
                 dtype=torch.float64):
        self.cfg = cfg
        nx, ny = cfg.grid
        mesh = MeshConfig(nx=nx, ny=ny, nz=1, dx=cfg.dx, dy=cfg.dx, dz=cfg.thickness)
        solver = SolverConfig(dt=cfg.dt, timesteps=timesteps, checkpoint=False,
                              renormalize=True, demag=True)
        self.mask = coupled_ported_mask(cfg, device, dtype)
        alpha = coupled_ported_alpha(cfg, device, dtype) * self.mask
        self.rollout = LLGRollout(mesh, solver, A=cfg.A, alpha=alpha, Ms_ref=cfg.Ms)
        self.rollout.set_Ms(cfg.Ms * self.mask)
        self.h_zero = torch.zeros(nx, ny, 1, 3, device=device, dtype=dtype)
        self.m0 = None

        x = (torch.arange(nx, device=device, dtype=dtype) - (nx - 1) / 2) * cfg.dx
        y = (torch.arange(ny, device=device, dtype=dtype) - (ny - 1) / 2) * cfg.dx
        X, Y = torch.meshgrid(x, y, indexing="ij")
        self.disk_masks = torch.stack([
            (((X - cx) ** 2 + (Y - cy) ** 2) <= cfg.radius**2).to(dtype)
            for cx, cy in cfg.centres()])
        self.guide_cells_per_disk = int(
            (self.mask[:, :, 0, 0].sum() - self.disk_masks.sum()) / 2)

    def relax(self, steps: int = 1800, alpha_relax: float = 0.5,
              require_tol: float | None = None):
        """Relax to the ground state.

        Default raised from 900 to 1800. At 900 the state was still settling,
        and that residual drift read as exponential divergence in the Lyapunov
        estimator -- the actual cause of an instability I attributed to four
        other mechanisms first. Pass ``require_tol`` (e.g. 1e-4) to make drift
        a hard failure: ``LLGRollout.relax`` warns only below |dm| = 0.02,
        which is ample for probe readouts and far too loose for a stability
        measurement.
        """
        cfg = self.cfg
        nx, ny = cfg.grid
        x = (torch.arange(nx, dtype=self.h_zero.dtype) - (nx - 1) / 2) * cfg.dx
        y = (torch.arange(ny, dtype=self.h_zero.dtype) - (ny - 1) / 2) * cfg.dx
        X, Y = torch.meshgrid(x, y, indexing="ij")
        m = torch.zeros(nx, ny, 1, 3, dtype=self.h_zero.dtype)
        m[:, :, 0, 2] = 1.0
        chir = [cfg.chirality,
                cfg.chirality if cfg.chirality_b is None else cfg.chirality_b]
        for k, (cx, cy) in enumerate(cfg.centres()):
            dX, dY = X - cx, Y - cy
            r = torch.sqrt(dX**2 + dY**2).clamp_min(1e-18)
            mz = cfg.polarity * torch.exp(-(r / cfg.core_width) ** 2)
            ip = torch.sqrt((1 - mz**2).clamp_min(0.0))
            sel = self.disk_masks[k] > 0
            c = chir[k]
            m[:, :, 0, 0] = torch.where(sel, -c * ip * dY / r, m[:, :, 0, 0])
            m[:, :, 0, 1] = torch.where(sel, c * ip * dX / r, m[:, :, 0, 1])
            m[:, :, 0, 2] = torch.where(sel, mz, m[:, :, 0, 2])
        m = m / m.norm(dim=-1, keepdim=True).clamp_min(1e-12) * self.mask
        self.m0 = self.rollout.relax(m, self.h_zero, steps, alpha_relax)
        if require_tol is not None:
            probe = self.rollout.relax(self.m0.clone(), self.h_zero, 20, alpha_relax)
            drift = float((probe - self.m0).norm() / max(float(self.m0.norm()), 1e-30))
            if drift > require_tol:
                raise RuntimeError(
                    f"relaxation still drifting: {drift:.2e} > {require_tol:.0e} "
                    f"after {steps} steps. A stability measurement on this state "
                    f"would report the drift as divergence.")
        return self.m0

    def link_wall(self) -> dict:
        """Is there a domain wall in the link? Measured, not assumed.

        Scans the link centreline strictly between the disk edges and reports
        the mx reversal and the peak out-of-plane component -- the two
        signatures of a Bloch wall.
        """
        cfg = self.cfg
        nx, ny = cfg.grid
        x = (torch.arange(nx, dtype=self.m0.dtype) - (nx - 1) / 2) * cfg.dx
        y = (torch.arange(ny, dtype=self.m0.dtype) - (ny - 1) / 2) * cfg.dx
        X, Y = torch.meshgrid(x, y, indexing="ij")
        gap = cfg.separation / 2 - cfg.radius
        row = ((X.abs() < gap) & (Y.abs() < cfg.dx)
               & (self.mask[:, :, 0, 0] > 0))
        if row.sum() < 4:
            return {"has_link": False}
        order = X[row].argsort()
        mx = self.m0[:, :, 0, 0][row][order]
        mz = self.m0[:, :, 0, 2][row][order]
        reverses = bool((mx[:3].mean() * mx[-3:].mean()) < -0.25)
        return {"has_link": True, "reverses": reverses,
                "mx_left": float(mx[:3].mean()), "mx_right": float(mx[-3:].mean()),
                "mz_peak": float(mz.abs().max())}

    def disk_energy(self, m: torch.Tensor) -> torch.Tensor:
        dm = ((m - self.m0) ** 2).sum(-1)[:, :, 0]
        w = self.disk_masks
        return (w * dm).sum(dim=(1, 2)) / w.sum(dim=(1, 2)).clamp_min(1e-30)
