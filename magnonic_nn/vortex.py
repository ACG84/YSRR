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

def _dev(device):
    """Resolve a device argument to the CURRENT torch default.

    These constructors defaulted to "cpu" literally, so the geometry -- mask,
    alpha, the vortex state -- stayed on the host even after set_device("cuda")
    moved magnum.np's own tensors. The result was a device mismatch deep inside
    magnum.np's exchange field, reported as a bare RuntimeError because its
    Timer re-raises the exception CLASS and discards the message. Following the
    default device means set_device works as the rest of the package assumes.
    """
    return torch.empty(0).device if device is None else device



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


def disk_mask(cfg: VortexConfig, device=None, dtype=torch.float64) -> torch.Tensor:
    """``(nx, ny, 1, 1)`` mask: 1 inside the disk, 0 in the surrounding vacuum."""
    n = cfg.n_cells
    idx = (torch.arange(n, device=_dev(device), dtype=dtype) - (n - 1) / 2) * cfg.dx
    X, Y = torch.meshgrid(idx, idx, indexing="ij")
    return ((X**2 + Y**2) <= cfg.radius**2).to(dtype).reshape(n, n, 1, 1)


def vortex_state(cfg: VortexConfig, device=None, dtype=torch.float64) -> torch.Tensor:
    """Analytic vortex ansatz, ``(nx, ny, 1, 3)``.

    In-plane curling with the chosen chirality, plus an out-of-plane core whose
    profile is the usual ``exp(-(r/w)^2)``. It is only a starting guess -- the
    relaxation below finds the true ground state -- but a good one keeps the
    relaxation short and, more importantly, lands in the intended
    (polarity, chirality) sector rather than a random one.
    """
    n = cfg.n_cells
    idx = (torch.arange(n, device=_dev(device), dtype=dtype) - (n - 1) / 2) * cfg.dx
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

    def __init__(self, cfg: VortexConfig, timesteps: int, device=None,
                 dtype=torch.float64, checkpoint: bool = False):
        self.cfg = cfg
        n = cfg.n_cells
        mesh = MeshConfig(nx=n, ny=n, nz=1, dx=cfg.dx, dy=cfg.dx, dz=cfg.thickness)
        solver = SolverConfig(dt=cfg.dt, timesteps=timesteps, checkpoint=checkpoint,
                              renormalize=True, demag=True)

        self.mask = disk_mask(cfg, device, dtype)
        alpha = torch.full((n, n, 1, 1), cfg.alpha, device=_dev(device), dtype=dtype)
        self.rollout = LLGRollout(mesh, solver, A=cfg.A, alpha=alpha, Ms_ref=cfg.Ms)
        self.rollout.set_Ms(cfg.Ms * self.mask)

        self.h_zero = torch.zeros(n, n, 1, 3, device=_dev(device), dtype=dtype)
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


def ported_mask(cfg: PortedVortexConfig, device=None, dtype=torch.float64):
    """``(nx, ny, 1, 1)`` mask for the disk plus its radial guides."""
    n = cfg.n_cells
    idx = (torch.arange(n, device=_dev(device), dtype=dtype) - (n - 1) / 2) * cfg.dx
    X, Y = torch.meshgrid(idx, idx, indexing="ij")

    mask = (X**2 + Y**2) <= cfg.radius**2
    for th in cfg.port_angles():
        # guide-local coordinates: u along the guide, v across it
        u = X * math.cos(th) + Y * math.sin(th)
        v = -X * math.sin(th) + Y * math.cos(th)
        mask = mask | ((u >= 0) & (u <= cfg.radius + cfg.guide_length)
                       & (v.abs() <= cfg.guide_width / 2))
    return mask.to(dtype).reshape(n, n, 1, 1)


def port_alpha(cfg: PortedVortexConfig, device=None, dtype=torch.float64):
    """Damping field: uniform everywhere, ramped up at the guide far ends.

    Without this the guides are resonators rather than ports -- the wave
    reflects off the open end and returns, so what the tap measures is a
    standing wave set by guide length instead of what the disk emitted.
    """
    n = cfg.n_cells
    idx = (torch.arange(n, device=_dev(device), dtype=dtype) - (n - 1) / 2) * cfg.dx
    X, Y = torch.meshgrid(idx, idx, indexing="ij")
    r = torch.sqrt(X**2 + Y**2)

    outer = cfg.radius + cfg.guide_length
    start = outer - cfg.absorb_frac * cfg.guide_length
    ramp = ((r - start) / (outer - start)).clamp(0.0, 1.0) ** 2
    return (cfg.alpha + (cfg.absorb_alpha - cfg.alpha) * ramp).reshape(n, n, 1, 1)


class PortedVortexDisk(VortexDisk):
    """Vortex disk read out through its waveguide ports."""

    def __init__(self, cfg: PortedVortexConfig, timesteps: int, device=None,
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

        self.h_zero = torch.zeros(n, n, 1, 3, device=_dev(device), dtype=dtype)
        self.m0 = None
        self._tap_masks = self._build_taps(device, dtype)

    def _build_taps(self, device, dtype):
        """One tap per guide, placed before the absorbing taper begins."""
        cfg = self.cfg
        n = cfg.n_cells
        idx = (torch.arange(n, device=_dev(device), dtype=dtype) - (n - 1) / 2) * cfg.dx
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


def coupled_mask(cfg: CoupledArrayConfig, device=None, dtype=torch.float64):
    """Two disks, the link between them, and one outward port each."""
    nx, ny = cfg.grid
    x = (torch.arange(nx, device=_dev(device), dtype=dtype) - (nx - 1) / 2) * cfg.dx
    y = (torch.arange(ny, device=_dev(device), dtype=dtype) - (ny - 1) / 2) * cfg.dx
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


def coupled_alpha(cfg: CoupledArrayConfig, device=None, dtype=torch.float64):
    """Damping ramped only at the OUTWARD ends; the link stays lossless."""
    nx, ny = cfg.grid
    x = (torch.arange(nx, device=_dev(device), dtype=dtype) - (nx - 1) / 2) * cfg.dx
    y = (torch.arange(ny, device=_dev(device), dtype=dtype) - (ny - 1) / 2) * cfg.dx
    X, _ = torch.meshgrid(x, y, indexing="ij")

    outer = cfg.separation / 2 + cfg.radius + cfg.out_length
    start = outer - cfg.absorb_frac * cfg.out_length
    ramp = ((X.abs() - start) / (outer - start)).clamp(0.0, 1.0) ** 2
    return (cfg.alpha + (cfg.absorb_alpha - cfg.alpha) * ramp).reshape(nx, ny, 1, 1)


class CoupledDiskArray:
    """Two guide-coupled vortex disks, driven and read at either end."""

    def __init__(self, cfg: CoupledArrayConfig, timesteps: int, device=None,
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
        self.h_zero = torch.zeros(nx, ny, 1, 3, device=_dev(device), dtype=dtype)
        self.m0 = None

        x = (torch.arange(nx, device=_dev(device), dtype=dtype) - (nx - 1) / 2) * cfg.dx
        y = (torch.arange(ny, device=_dev(device), dtype=dtype) - (ny - 1) / 2) * cfg.dx
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


def coupled_ported_mask(cfg: CoupledPortedConfig, device=None, dtype=torch.float64):
    nx, ny = cfg.grid
    x = (torch.arange(nx, device=_dev(device), dtype=dtype) - (nx - 1) / 2) * cfg.dx
    y = (torch.arange(ny, device=_dev(device), dtype=dtype) - (ny - 1) / 2) * cfg.dx
    X, Y = torch.meshgrid(x, y, indexing="ij")
    m = torch.zeros_like(X, dtype=torch.bool)
    for cx, cy in cfg.centres():
        m = m | _radial_guides(X, Y, cx, cy, cfg)
    # bridge whatever gap remains between the facing guide tips
    m = m | ((X.abs() <= cfg.separation / 2) & (Y.abs() <= cfg.link_width / 2))
    return m.to(dtype).reshape(nx, ny, 1, 1)


def coupled_ported_alpha(cfg: CoupledPortedConfig, device=None, dtype=torch.float64):
    """Absorb at the OUTWARD guide ends; the link corridor stays lossless."""
    nx, ny = cfg.grid
    x = (torch.arange(nx, device=_dev(device), dtype=dtype) - (nx - 1) / 2) * cfg.dx
    y = (torch.arange(ny, device=_dev(device), dtype=dtype) - (ny - 1) / 2) * cfg.dx
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

    def __init__(self, cfg: CoupledPortedConfig, timesteps: int, device=None,
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
        self.h_zero = torch.zeros(nx, ny, 1, 3, device=_dev(device), dtype=dtype)
        self.m0 = None

        x = (torch.arange(nx, device=_dev(device), dtype=dtype) - (nx - 1) / 2) * cfg.dx
        y = (torch.arange(ny, device=_dev(device), dtype=dtype) - (ny - 1) / 2) * cfg.dx
        X, Y = torch.meshgrid(x, y, indexing="ij")
        self.disk_masks = torch.stack([
            (((X - cx) ** 2 + (Y - cy) ** 2) <= cfg.radius**2).to(dtype)
            for cx, cy in cfg.centres()])
        self.guide_cells_per_disk = int(
            (self.mask[:, :, 0, 0].sum() - self.disk_masks.sum()) / 2)
        self._tap_masks = self._build_taps(X, Y, dtype)

    def _build_taps(self, X, Y, dtype):
        """One tap per guide per disk, same geometry as the single disk.

        The class carried no readout at all, so any task run through it was
        impossible rather than merely wrong. Taps sit just inside the absorbing
        taper, as on PortedVortexDisk, but referenced to EACH disk's own centre.

        Restricted to cells that actually carry material: a tap window that
        overhangs empty space would average real signal against structural zeros
        and read low by whatever fraction it overhangs.
        """
        cfg = self.cfg
        outer = cfg.radius + cfg.guide_length
        tap_r = outer - cfg.absorb_frac * cfg.guide_length - 2 * cfg.dx
        solid = self.mask[:, :, 0, 0] > 0.5
        taps = []
        for cx, cy in cfg.centres():
            dX, dY = X - cx, Y - cy
            for th in cfg.port_angles():
                u = dX * math.cos(th) + dY * math.sin(th)
                v = -dX * math.sin(th) + dY * math.cos(th)
                sel = (((u - tap_r).abs() <= 1.5 * cfg.dx)
                       & (v.abs() <= cfg.guide_width / 2) & solid)
                taps.append(sel.to(dtype))
        return torch.stack(taps)          # (n_disks * n_ports, nx, ny)

    def port_signals(self, m: torch.Tensor) -> torch.Tensor:
        """Mean out-of-plane deviation at each tap, disk A's ports first."""
        dm = (m - self.m0)[:, :, 0, 2]
        w = self._tap_masks
        return (w * dm).sum(dim=(1, 2)) / w.sum(dim=(1, 2)).clamp_min(1e-30)

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


@dataclass
class ChainPortedConfig(PortedVortexConfig):
    """N fully-ported vortex disks in a line, each linked to its neighbours.

    CoupledPortedConfig generalised from two stages to a cascade of N, so that
    "how deep can this go" is a question the geometry can express at all.

    The depth budget is set by two measured numbers rather than by taste. At
    700 nm the guided transfer per hop is 0.383 and the dipolar crosstalk that
    reaches any disk directly from the driven one is 0.0043, so the guided
    signal arriving at stage n is 0.383^(n-1) and falls to the crosstalk floor
    at stage 6. Depth 5 is therefore the ceiling at this separation, and depth 4
    puts the deepest stage's response at lag ~10.4 -- which is where NARMA-10's
    product term lives.

    CHIRALITY ALTERNATES, and that is not cosmetic. Two neighbours of the SAME
    chirality drive the strip between them in the same rotational sense, which
    is opposite senses in the lab frame, and the strip nucleates a Bloch wall at
    the midpoint it cannot avoid. Measured on the two-disk build at 30 mT:

        same chirality      wall, mz peak 0.98   lambda -0.171   B/A 0.00275
        opposite chirality  no wall, mz 0.00     lambda -0.292   B/A 0.00380

    Alternating +1/-1 along the chain makes EVERY adjacent pair opposite, which
    is the only assignment with that property.
    """

    n_disks: int = 3
    separation: float = 700e-9        # centre to centre, adjacent stages
    link_width: float = 80e-9         # 80 to match the ports: 40 is evanescent
                                      # at 12 GHz and would not be a waveguide
    link_alpha: float | None = None   # damping inside the link corridors

    @property
    def grid(self) -> tuple[int, int]:
        half_x = ((self.n_disks - 1) * self.separation / 2
                  + self.radius + self.guide_length)
        reach_y = self.radius + self.guide_length
        return (2 * (int(np.ceil(half_x / self.dx)) + self.margin_cells),
                2 * (int(np.ceil(reach_y / self.dx)) + self.margin_cells))

    def centres(self):
        off = (self.n_disks - 1) / 2.0
        return [((i - off) * self.separation, 0.0) for i in range(self.n_disks)]

    def chiralities(self):
        return [(+1 if i % 2 == 0 else -1) for i in range(self.n_disks)]


def _chain_links(cfg: ChainPortedConfig, X, Y):
    """Boolean mask of the link corridors between consecutive stages."""
    reg = torch.zeros_like(X, dtype=torch.bool)
    cs = cfg.centres()
    for (ax, _), (bx, _) in zip(cs[:-1], cs[1:]):
        reg = reg | ((X >= ax) & (X <= bx) & (Y.abs() <= cfg.link_width / 2))
    return reg


def chain_ported_mask(cfg: ChainPortedConfig, device=None, dtype=torch.float64):
    nx, ny = cfg.grid
    x = (torch.arange(nx, device=_dev(device), dtype=dtype) - (nx - 1) / 2) * cfg.dx
    y = (torch.arange(ny, device=_dev(device), dtype=dtype) - (ny - 1) / 2) * cfg.dx
    X, Y = torch.meshgrid(x, y, indexing="ij")
    m = torch.zeros_like(X, dtype=torch.bool)
    for cx, cy in cfg.centres():
        m = m | _radial_guides(X, Y, cx, cy, cfg)
    if cfg.link_width > 0:
        m = m | _chain_links(cfg, X, Y)
    return m.to(dtype).reshape(nx, ny, 1, 1)


def chain_ported_alpha(cfg: ChainPortedConfig, device=None, dtype=torch.float64):
    """Absorb at the OUTWARD guide ends only; link corridors stay lossless.

    An absorbing taper spilling into a link would attenuate the very coupling
    the cascade runs on, so the corridors are cut out of the ramp explicitly.
    """
    nx, ny = cfg.grid
    x = (torch.arange(nx, device=_dev(device), dtype=dtype) - (nx - 1) / 2) * cfg.dx
    y = (torch.arange(ny, device=_dev(device), dtype=dtype) - (ny - 1) / 2) * cfg.dx
    X, Y = torch.meshgrid(x, y, indexing="ij")

    outer = cfg.radius + cfg.guide_length
    start = outer - cfg.absorb_frac * cfg.guide_length
    ramp = torch.ones_like(X)
    for cx, cy in cfg.centres():
        r = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        ramp = torch.minimum(ramp,
                             ((r - start) / (outer - start)).clamp(0.0, 1.0) ** 2)
    if cfg.link_width > 0:
        link = _chain_links(cfg, X, Y)
        ramp = torch.where(link, torch.zeros_like(ramp), ramp)
        alpha = cfg.alpha + (cfg.absorb_alpha - cfg.alpha) * ramp
        if cfg.link_alpha is not None:
            alpha = torch.where(link, torch.full_like(alpha, cfg.link_alpha),
                                alpha)
    else:
        alpha = cfg.alpha + (cfg.absorb_alpha - cfg.alpha) * ramp
    return alpha.reshape(nx, ny, 1, 1)


class ChainPortedArray:
    """A cascade of N ported vortex disks; stage i taps occupy block i."""

    def __init__(self, cfg: ChainPortedConfig, timesteps: int, device=None,
                 dtype=torch.float64):
        self.cfg = cfg
        nx, ny = cfg.grid
        mesh = MeshConfig(nx=nx, ny=ny, nz=1, dx=cfg.dx, dy=cfg.dx,
                          dz=cfg.thickness)
        solver = SolverConfig(dt=cfg.dt, timesteps=timesteps, checkpoint=False,
                              renormalize=True, demag=True)
        self.mask = chain_ported_mask(cfg, device, dtype)
        alpha = chain_ported_alpha(cfg, device, dtype) * self.mask
        self.rollout = LLGRollout(mesh, solver, A=cfg.A, alpha=alpha,
                                  Ms_ref=cfg.Ms)
        self.rollout.set_Ms(cfg.Ms * self.mask)
        self.h_zero = torch.zeros(nx, ny, 1, 3, device=_dev(device), dtype=dtype)
        self.m0 = None

        x = (torch.arange(nx, device=_dev(device), dtype=dtype) - (nx - 1) / 2) * cfg.dx
        y = (torch.arange(ny, device=_dev(device), dtype=dtype) - (ny - 1) / 2) * cfg.dx
        self._X, self._Y = torch.meshgrid(x, y, indexing="ij")
        self.disk_masks = torch.stack([
            (((self._X - cx) ** 2 + (self._Y - cy) ** 2) <= cfg.radius**2).to(dtype)
            for cx, cy in cfg.centres()])
        self._tap_masks = self._build_taps(dtype)

    def _build_taps(self, dtype):
        cfg = self.cfg
        outer = cfg.radius + cfg.guide_length
        tap_r = outer - cfg.absorb_frac * cfg.guide_length - 2 * cfg.dx
        solid = self.mask[:, :, 0, 0] > 0.5
        taps = []
        for cx, cy in cfg.centres():
            dX, dY = self._X - cx, self._Y - cy
            for th in cfg.port_angles():
                u = dX * math.cos(th) + dY * math.sin(th)
                v = -dX * math.sin(th) + dY * math.cos(th)
                taps.append((((u - tap_r).abs() <= 1.5 * cfg.dx)
                             & (v.abs() <= cfg.guide_width / 2)
                             & solid).to(dtype))
        return torch.stack(taps)

    def port_signals(self, m: torch.Tensor) -> torch.Tensor:
        dm = (m - self.m0)[:, :, 0, 2]
        w = self._tap_masks
        return (w * dm).sum(dim=(1, 2)) / w.sum(dim=(1, 2)).clamp_min(1e-30)

    def relax(self, steps: int = 5000, alpha_relax: float = 0.5,
              require_tol: float | None = None):
        cfg = self.cfg
        nx, ny = cfg.grid
        X, Y = self._X, self._Y
        m = torch.zeros(nx, ny, 1, 3, dtype=self.h_zero.dtype)
        m[:, :, 0, 2] = 1.0
        for k, (cx, cy) in enumerate(cfg.centres()):
            dX, dY = X - cx, Y - cy
            r = torch.sqrt(dX**2 + dY**2).clamp_min(1e-18)
            mz = cfg.polarity * torch.exp(-(r / cfg.core_width) ** 2)
            ip = torch.sqrt((1 - mz**2).clamp_min(0.0))
            sel = self.disk_masks[k] > 0
            c = cfg.chiralities()[k]
            m[:, :, 0, 0] = torch.where(sel, -c * ip * dY / r, m[:, :, 0, 0])
            m[:, :, 0, 1] = torch.where(sel, c * ip * dX / r, m[:, :, 0, 1])
            m[:, :, 0, 2] = torch.where(sel, mz, m[:, :, 0, 2])
        m = m / m.norm(dim=-1, keepdim=True).clamp_min(1e-12) * self.mask
        self.m0 = self.rollout.relax(m, self.h_zero, steps, alpha_relax)
        if require_tol is not None:
            probe = self.rollout.relax(self.m0.clone(), self.h_zero, 20,
                                       alpha_relax)
            drift = float((probe - self.m0).norm()
                          / max(float(self.m0.norm()), 1e-30))
            if drift > require_tol:
                raise RuntimeError(
                    f"relaxation still drifting: {drift:.2e} > "
                    f"{require_tol:.0e} after {steps} steps. A timing "
                    f"measurement on this state would read drift as signal.")
        return self.m0


@dataclass
class PolyTapConfig(PortedVortexConfig):
    """A delay-line BUS tapped in parallel by N vortex disks.

    Why this shape rather than a deeper chain. The capacity accounting on the
    3- and 4-stage chains found 98-99% of measured capacity sitting in degree-1
    targets, with NARMA-10's long-separation products at exactly 0.000 -- and
    the task's value lives entirely in those: products with |i-j| >= 5 score
    0.0401 while every short-separation family sits at the 0.124 linear
    baseline. The chain makes only the worthless kind, and for a structural
    reason. Its nonlinear element is the driven disk, whose own memory spans
    lags 0-6, so the only samples it can multiply are ones a frame or two
    apart. Deep stages remember longer but receive no fresh input to mix
    against. The chain mixes first and delays afterwards; the task needs the
    opposite order.

    So: one guide carries the input past every disk, and each disk ALSO
    receives the fresh sample. Disk i then multiplies u[n] against u[n - k_i],
    with k_i set by how far along the bus it sits -- many separations in
    parallel rather than one in series.

    Three measured numbers fix the geometry, none of them chosen by taste:

        189 nm    one frame of delay on an 80 nm bus (945 m/s, measured)
        951 nm    bus attenuation length (measured on a bare 5 um strip)
        16x       how much better the bus is than the chain over the same
                  delay: ~7x loss across 9.5 frames against the chain's ~118x

    The disks hang off the bus by their OWN 270-degree port guide, so the
    coupling is the tap geometry already characterised rather than a new
    element. That guide is the input; the other five are the readout. Feeding
    a signal into a guide that is also a readout tap would put the input
    straight into the features -- the failure the single-disk work already
    named -- so `port_signals` returns the five, not the six.

    The amplitude balance is the constraint the co-drive arm died on. Driving
    a tap with a full-amplitude fresh sample while its delayed copy arrives
    100x down turns the tap into another input disk: measured, degree-1
    capacity fell 8.91 -> 4.95 and the memory horizon pulled in from lag 12 to
    lag 7. Here the fresh drive per tap is scaled by `fresh_scale()` to track
    the bus decay, so both operands of the intended product arrive comparable.
    """

    n_taps: int = 4
    # Rotate the port ring by half a sector so ONE guide points straight at the
    # bus. The inherited ring sits at 0, 60, ... 300 degrees, which has no
    # guide at 270 and therefore nothing to couple with; the two nearest sit at
    # 240 and 300 and would each meet the bus at a 60-degree skew. The rotation
    # also narrows the disk's x-footprint from 500 to 433 nm, which is what
    # makes 567 nm tap spacing fit at all.
    port_phase: float = math.pi / 6
    # Gap between the bus edge and the tip of each disk's coupling guide.
    #
    # Zero -- a guide touching the bus -- is a STRONG tap, and strong taps drain
    # the line. Measured on the zero-gap build: the bus falls 566x from the
    # injection to the far end where the bare strip over the same span falls
    # 17x, and probing between taps shows each one removing 20-60% of what
    # reaches it on top of the guide's own attenuation. Tap 4 then receives
    # 300x less than tap 1, which is no balance at all.
    #
    # A gap couples evanescently, so a few tens of nm buys orders of magnitude
    # of coupling control: weak taps pass the wave on, at the cost of receiving
    # less of it themselves. That trade is the design's central free parameter
    # and this is the knob for sweeping it.
    coupling_gap: float = 0.0
    # Local damping multiplier on the TAP DISK BODIES only.
    #
    # The bus and the tap are not competing for the same alpha, and that is the
    # whole point. Attenuation length and ring-down are both 1/(alpha*omega), so
    # lowering alpha UNIFORMLY stretches each equally and buys nothing.
    #
    # A tap does not need memory. Its job is to multiply a fresh sample against
    # a delayed one; the memory lives in the bus. So a tap that rings briefly is
    # the correct tap. At the 567 nm spacing already built, a tap must respond
    # in under 0.60 ns to be resolved, which needs alpha_tap > 0.022 -- only
    # 2.8x permalloy's. Physically this is a Pt or Pd cap raising alpha by spin
    # pumping while the uncapped bus keeps its own.
    #
    # MEASURED: this knob does what it claims for RESOLUTION. At 10x, four
    # points in the 20-point sweep resolved the first tap pair -- 3.07, 2.62,
    # 2.64, 2.65 against a designed 3.0 -- which no undamped point ever did.
    #
    # REFUTED, and it was written here as established: that the array's
    # tap-to-tap amplitude spread follows from the same two timescales, giving
    # "taps within a dynamic range R is about ln(R)*(a_tap/a_bus)". That model
    # says spread is propagation loss along the bus, so a thirtyfold longer
    # attenuation length should have flattened it. The bus-damping sweep left
    # the spread at 405x, 458x, 408x, 388x. Whatever sets the spread here, it
    # is not bus attenuation, and the ratio does not govern it.
    #
    # What does set it: see bus_alpha_mult below. The taps read an evanescent
    # field, not a wave, so RESOLUTION as defined here has nothing to resolve.
    tap_alpha_mult: float = 1.0
    # Damping multiplier on the BUS and guides, the other half of the ratio.
    #
    # tap_alpha_mult alone sets alpha_tap/alpha_bus by raising the tap. This
    # lowers the bus instead, which is what a low-damping film would do, and the
    # two are not interchangeable: the ratio governs whether a tap can RESOLVE a
    # delay, while the bus alpha alone governs how far the delay line CARRIES.
    # The 20-point sweep resolved the first tap pair (3.07 against a designed
    # 3.0) and still failed at taps 3 and 4 because the weakest tap never rose
    # above 2.8e-06. Attenuation length is 951 nm at alpha 0.008, so the lag-14
    # tap at 2646 nm sees exp(-2.78) = 0.062 of the injection; at a tenth the
    # bus damping it sees 0.757, a 12x gain that should put it near 3e-05.
    #
    # MEASURED, half right. Reception improves and clears the 1e-5 bar for the
    # first time -- 1.19e-05 at x0.1 and 1.40e-05 at x0.03 -- but at roughly a
    # third of the predicted size. The spread prediction (~400x collapsing to
    # ~1.2x) failed outright: see tap_alpha_mult above. Reception and
    # resolution stay anti-correlated, because turning the tap damping back on
    # to recover resolution halves the weakest tap again (5.88e-06, 6.53e-06).
    #
    # WHY the spread prediction failed, measured afterwards by shifting the
    # whole array ten frames further from the injection at bus x0.03. Tap 1 fell
    # from 5.42e-03 to 9.09e-06. A guided wave over that extra 1890 nm would
    # have fallen 6%; an evanescent field with the ~285 nm decay length implied
    # by the baseline spread predicts 7.2e-06, and that number was registered
    # before the run. Arrival lags across the baseline array are flat to 0.04
    # frames over 1701 nm where 945 m/s demands 9.0 -- attenuation would make a
    # wave small, not punctual. Nothing propagates to these taps at 12 GHz, so
    # neither damping knob was ever addressing the reason the array fails.
    bus_alpha_mult: float = 1.0
    bus_width: float = 80e-9
    bus_absorb: float = 400e-9        # absorbing taper at each bus end
    inject_at: float = 600e-9         # from the left bus end
    inject_len: float = 100e-9
    tap_lags: tuple = (5.0, 8.0, 11.0, 14.0)   # frames of delay, per tap
    frame_nm: float = 189e-9          # MEASURED: one frame of bus delay
    atten_nm: float = 951e-9          # MEASURED: bus attenuation length
    margin: float = 80e-9

    def port_angles(self):
        return [self.port_phase + 2 * math.pi * k / self.n_ports
                for k in range(self.n_ports)]

    def tap_x(self):
        """Distance of each tap from the left bus end."""
        return [self.inject_at + lag * self.frame_nm
                for lag in self.tap_lags[:self.n_taps]]

    def disk_cy(self) -> float:
        """Disk centre height: its 270-degree guide stops `coupling_gap` short."""
        return (self.bus_width / 2 + self.coupling_gap
                + self.radius + self.guide_length)

    def bus_length(self) -> float:
        return (self.tap_x()[-1] + self.radius + self.guide_length
                + self.bus_absorb + self.margin)

    def fresh_scale(self):
        """Per-tap fresh-drive amplitude, tracking the measured bus decay.

        The delayed copy at tap i is attenuated by exp(-d_i / atten_nm); the
        fresh sample is not attenuated at all. Scaling the fresh drive by the
        same factor keeps the two operands of the product comparable at every
        tap, which is precisely what the chain co-drive could not do.
        """
        x0 = self.inject_at
        return [math.exp(-(x - x0) / self.atten_nm) for x in self.tap_x()]

    @property
    def grid(self) -> tuple[int, int]:
        nx = int(np.ceil(self.bus_length() / self.dx)) + 2 * self.margin_cells
        top = self.disk_cy() + self.radius + self.guide_length + self.margin
        bot = self.bus_width / 2 + self.margin
        ny = int(np.ceil((top + bot) / self.dx)) + 2 * self.margin_cells
        return (nx, ny)

    def centres(self):
        """Disk centres in mesh coordinates (origin at the mesh centre)."""
        nx, ny = self.grid
        x_off = -(nx - 1) / 2 * self.dx           # mesh x of the left bus end
        y_bus = self._bus_y()
        return [(x_off + x, y_bus + self.disk_cy()) for x in self.tap_x()]

    def _bus_y(self) -> float:
        """Mesh y of the bus centreline: disks sit above, so the bus sits low."""
        nx, ny = self.grid
        return -(ny - 1) / 2 * self.dx + self.bus_width / 2 + self.margin

    def inject_x(self) -> float:
        nx, ny = self.grid
        return -(nx - 1) / 2 * self.dx + self.inject_at

    def chiralities(self):
        return [(+1 if i % 2 == 0 else -1) for i in range(self.n_taps)]


def polytap_mask(cfg: PolyTapConfig, device=None, dtype=torch.float64):
    nx, ny = cfg.grid
    x = (torch.arange(nx, device=_dev(device), dtype=dtype) - (nx - 1) / 2) * cfg.dx
    y = (torch.arange(ny, device=_dev(device), dtype=dtype) - (ny - 1) / 2) * cfg.dx
    X, Y = torch.meshgrid(x, y, indexing="ij")
    yb = cfg._bus_y()
    x0 = -(nx - 1) / 2 * cfg.dx
    m = (((Y - yb).abs() <= cfg.bus_width / 2)
         & (X >= x0) & (X <= x0 + cfg.bus_length()))
    for cx, cy in cfg.centres():
        m = m | _radial_guides(X, Y, cx, cy, cfg)
    return m.to(dtype).reshape(nx, ny, 1, 1)


def polytap_alpha(cfg: PolyTapConfig, device=None, dtype=torch.float64):
    """Absorb at the bus ends and at the five READOUT guide ends.

    The 270-degree guide is the bus coupling, not a port, so its taper is cut
    out by the vertical corridor below each disk -- an absorber there would
    attenuate the very signal the tap exists to receive, the same mistake the
    chain avoided by cutting its link corridors out of the ramp.
    """
    nx, ny = cfg.grid
    x = (torch.arange(nx, device=_dev(device), dtype=dtype) - (nx - 1) / 2) * cfg.dx
    y = (torch.arange(ny, device=_dev(device), dtype=dtype) - (ny - 1) / 2) * cfg.dx
    X, Y = torch.meshgrid(x, y, indexing="ij")
    yb, x0 = cfg._bus_y(), -(nx - 1) / 2 * cfg.dx
    L = cfg.bus_length()

    outer = cfg.radius + cfg.guide_length
    start = outer - cfg.absorb_frac * cfg.guide_length
    ramp = torch.zeros_like(X)
    for cx, cy in cfg.centres():
        r = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        g = ((r - start) / (outer - start)).clamp(0.0, 1.0) ** 2
        # LOCAL to this disk. The clamp saturates at 1 for every r beyond the
        # guide tip, so without this the taper would extend to infinity and
        # damp the entire bus at absorb_alpha -- which is exactly what the
        # first version did, and what the geometry check caught.
        g = torch.where(r <= outer + cfg.dx, g, torch.zeros_like(g))
        # cut the coupling corridor: directly below this disk, down to the bus
        g = torch.where(((X - cx).abs() <= cfg.guide_width / 2) & (Y < cy),
                        torch.zeros_like(g), g)
        ramp = torch.maximum(ramp, g)
    # bus ends
    bl = ((x0 + cfg.bus_absorb - X) / cfg.bus_absorb).clamp(0.0, 1.0) ** 2
    br = ((X - (x0 + L - cfg.bus_absorb)) / cfg.bus_absorb).clamp(0.0, 1.0) ** 2
    on_bus = (Y - yb).abs() <= cfg.bus_width / 2
    ramp = torch.maximum(ramp, torch.where(on_bus,
                                           torch.maximum(bl, br),
                                           torch.zeros_like(bl)))
    base = cfg.alpha * cfg.bus_alpha_mult
    a = base + (cfg.absorb_alpha - base) * ramp
    a = _tap_damped(cfg, X, Y, a)
    return a.reshape(nx, ny, 1, 1)


def _tap_damped(cfg, X, Y, a):
    """Set alpha on the tap disk BODIES, leaving bus, guides and absorbers alone.

    ABSOLUTE, not a multiplier on whatever is already there: the bus may have
    been scaled by bus_alpha_mult, and the tap's damping should not inherit
    that. Bus alpha is cfg.alpha * bus_alpha_mult, tap alpha is
    cfg.alpha * tap_alpha_mult, and the ratio between them -- the quantity the
    design equation is written in -- is tap_alpha_mult / bus_alpha_mult.
    """
    if cfg.tap_alpha_mult == 1.0 and cfg.bus_alpha_mult == 1.0:
        return a
    tap_a = cfg.alpha * cfg.tap_alpha_mult
    for cx, cy in cfg.centres():
        body = ((X - cx) ** 2 + (Y - cy) ** 2) <= cfg.radius ** 2
        a = torch.where(body, torch.full_like(a, tap_a), a)
    return a


class PolyTapArray:
    """A bus with N tap disks. Tap i contributes five readout ports."""

    BUS_PORT = 4          # index of the 270-degree guide in port_angles()

    def __init__(self, cfg: PolyTapConfig, timesteps: int, device=None,
                 dtype=torch.float64):
        self.cfg = cfg
        nx, ny = cfg.grid
        mesh = MeshConfig(nx=nx, ny=ny, nz=1, dx=cfg.dx, dy=cfg.dx,
                          dz=cfg.thickness)
        solver = SolverConfig(dt=cfg.dt, timesteps=timesteps, checkpoint=False,
                              renormalize=True, demag=True)
        self.mask = polytap_mask(cfg, device, dtype)
        alpha = polytap_alpha(cfg, device, dtype) * self.mask
        self.rollout = LLGRollout(mesh, solver, A=cfg.A, alpha=alpha,
                                  Ms_ref=cfg.Ms)
        self.rollout.set_Ms(cfg.Ms * self.mask)
        self.h_zero = torch.zeros(nx, ny, 1, 3, device=_dev(device), dtype=dtype)
        self.m0 = None

        x = (torch.arange(nx, device=_dev(device), dtype=dtype) - (nx - 1) / 2) * cfg.dx
        y = (torch.arange(ny, device=_dev(device), dtype=dtype) - (ny - 1) / 2) * cfg.dx
        self._X, self._Y = torch.meshgrid(x, y, indexing="ij")
        self.disk_masks = torch.stack([
            (((self._X - cx) ** 2 + (self._Y - cy) ** 2) <= cfg.radius**2).to(dtype)
            for cx, cy in cfg.centres()])
        self.bus_mask = self._bus_region(dtype)
        self.inject_mask = self._inject_region(dtype)
        self._tap_masks = self._build_taps(dtype)

    @property
    def n_readout(self) -> int:
        return self.cfg.n_ports - 1

    def _bus_region(self, dtype):
        cfg = self.cfg
        nx, _ = cfg.grid
        x0 = -(nx - 1) / 2 * cfg.dx
        return (((self._Y - cfg._bus_y()).abs() <= cfg.bus_width / 2)
                & (self._X >= x0)
                & (self._X <= x0 + cfg.bus_length())).to(dtype)

    def _inject_region(self, dtype):
        cfg = self.cfg
        xi = cfg.inject_x()
        return (((self._Y - cfg._bus_y()).abs() <= cfg.bus_width / 2)
                & (self._X >= xi) & (self._X <= xi + cfg.inject_len)).to(dtype)

    def _build_taps(self, dtype):
        """Five taps per disk: every port guide EXCEPT the bus coupling."""
        cfg = self.cfg
        outer = cfg.radius + cfg.guide_length
        tap_r = outer - cfg.absorb_frac * cfg.guide_length - 2 * cfg.dx
        solid = self.mask[:, :, 0, 0] > 0.5
        angles = cfg.port_angles()
        taps = []
        for cx, cy in cfg.centres():
            dX, dY = self._X - cx, self._Y - cy
            for i, th in enumerate(angles):
                if i == self.BUS_PORT:
                    continue
                u = dX * math.cos(th) + dY * math.sin(th)
                v = -dX * math.sin(th) + dY * math.cos(th)
                taps.append((((u - tap_r).abs() <= 1.5 * cfg.dx)
                             & (v.abs() <= cfg.guide_width / 2)
                             & solid).to(dtype))
        return torch.stack(taps)

    def port_signals(self, m: torch.Tensor) -> torch.Tensor:
        dm = (m - self.m0)[:, :, 0, 2]
        w = self._tap_masks
        return (w * dm).sum(dim=(1, 2)) / w.sum(dim=(1, 2)).clamp_min(1e-30)

    def relax(self, steps: int = 8000, alpha_relax: float = 0.5,
              require_tol: float | None = None):
        cfg = self.cfg
        nx, ny = cfg.grid
        X, Y = self._X, self._Y
        m = torch.zeros(nx, ny, 1, 3, dtype=self.h_zero.dtype)
        # Bus and guides start along the bus axis: shape anisotropy puts a long
        # strip's ground state along its own length.
        m[:, :, 0, 0] = 1.0
        for k, (cx, cy) in enumerate(cfg.centres()):
            dX, dY = X - cx, Y - cy
            r = torch.sqrt(dX**2 + dY**2).clamp_min(1e-18)
            mz = cfg.polarity * torch.exp(-(r / cfg.core_width) ** 2)
            ip = torch.sqrt((1 - mz**2).clamp_min(0.0))
            sel = self.disk_masks[k] > 0
            c = cfg.chiralities()[k]
            m[:, :, 0, 0] = torch.where(sel, -c * ip * dY / r, m[:, :, 0, 0])
            m[:, :, 0, 1] = torch.where(sel, c * ip * dX / r, m[:, :, 0, 1])
            m[:, :, 0, 2] = torch.where(sel, mz, m[:, :, 0, 2])
        m = m / m.norm(dim=-1, keepdim=True).clamp_min(1e-12) * self.mask
        self.m0 = self.rollout.relax(m, self.h_zero, steps, alpha_relax)
        if require_tol is not None:
            probe = self.rollout.relax(self.m0.clone(), self.h_zero, 20,
                                       alpha_relax)
            drift = float((probe - self.m0).norm()
                          / max(float(self.m0.norm()), 1e-30))
            if drift > require_tol:
                raise RuntimeError(
                    f"relaxation still drifting: {drift:.2e} > "
                    f"{require_tol:.0e} after {steps} steps.")
        return self.m0


@dataclass
class DirCouplerConfig(PolyTapConfig):
    """Poly-tap bus where each tap is a DIRECTIONAL coupler, not a stub.

    The perpendicular stub was measured and rejected. A galvanic stub drains the
    line (bus through-loss 0.431 at tap 1 against 0.729 for the bare guide), and
    backing it off with a gap fixes the draining (0.912 at 30 nm) while halving
    what the tap receives and collapsing its measured arrival from lag 4.90 to
    lag 1.50 -- the tap stops reading the guided wave and starts reading stray
    field. Both couplings are near-field, so a gap attenuates the signal and the
    contamination together and no gap separates them.

    A co-directional coupler separates them on a different axis. The tap's arm
    runs PARALLEL to the bus, phase-matched (same width, so the same
    dispersion), over a coupling length L_c. Guided power transfers coherently
    along that length, accumulating as it goes; stray dipolar pickup does not
    accumulate, because it has no fixed phase relationship to add along. So
    coupling strength becomes a function of LENGTH -- which trades against
    nothing but floor area -- instead of proximity, which trades directly
    against contrast.

    The arm is co-directional: power crosses into it travelling the same way the
    bus wave travels, runs downstream to the arm's far end, and turns up into the
    disk. The arm's UPSTREAM end is absorbing so nothing reflects back into the
    coupling region.

    The vertical link adds its own transit -- ~1 frame at 945 m/s over 200 nm --
    on top of the bus delay to the tap. Tap lags are therefore measured rather
    than assumed, as they were for the stub build.
    """

    coupler_len: float = 400e-9       # L_c, the coupling length. Bounded above
                                      # by tap spacing: arms must not merge.
    coupler_gap: float = 20e-9        # bus edge to arm edge
    coupler_width: float = 80e-9      # MUST match the bus width, or the two
                                      # guides have different dispersion, are not
                                      # phase matched, and power beats back out
                                      # as fast as it couples in.
    link_len: float = 60e-9           # arm up to the disk's coupling guide
    arm_absorb: float = 100e-9        # taper at the arm's upstream end

    def arm_y(self) -> tuple[float, float]:
        """(bottom, top) of the coupler arm, relative to the bus centreline."""
        b = self.bus_width / 2 + self.coupler_gap
        return (b, b + self.coupler_width)

    def disk_cy(self) -> float:
        _, top = self.arm_y()
        return top + self.link_len + self.radius + self.guide_length

    @property
    def grid(self) -> tuple[int, int]:
        nx = int(np.ceil(self.bus_length() / self.dx)) + 2 * self.margin_cells
        top = self.disk_cy() + self.radius + self.guide_length + self.margin
        bot = self.bus_width / 2 + self.margin
        ny = int(np.ceil((top + bot) / self.dx)) + 2 * self.margin_cells
        return (nx, ny)

    def max_coupler_len(self) -> float:
        """Longest arm that still leaves a gap between adjacent taps."""
        xs = self.tap_x()
        if len(xs) < 2:
            return self.coupler_len
        return min(b - a for a, b in zip(xs[:-1], xs[1:])) - 4 * self.dx


def dircoupler_mask(cfg: DirCouplerConfig, device=None, dtype=torch.float64):
    nx, ny = cfg.grid
    x = (torch.arange(nx, device=_dev(device), dtype=dtype) - (nx - 1) / 2) * cfg.dx
    y = (torch.arange(ny, device=_dev(device), dtype=dtype) - (ny - 1) / 2) * cfg.dx
    X, Y = torch.meshgrid(x, y, indexing="ij")
    yb = cfg._bus_y()
    x0 = -(nx - 1) / 2 * cfg.dx
    lo, hi = cfg.arm_y()

    m = (((Y - yb).abs() <= cfg.bus_width / 2)
         & (X >= x0) & (X <= x0 + cfg.bus_length()))
    for (cx, cy) in cfg.centres():
        # co-directional arm, running INTO the tap from upstream
        m = m | ((X >= cx - cfg.coupler_len) & (X <= cx)
                 & (Y >= yb + lo) & (Y <= yb + hi))
        # vertical link from the arm up to the disk's coupling guide
        m = m | ((X - cx).abs() <= cfg.guide_width / 2) & (Y >= yb + hi) \
                & (Y <= cy - (cfg.radius + cfg.guide_length))
        m = m | _radial_guides(X, Y, cx, cy, cfg)
    return m.to(dtype).reshape(nx, ny, 1, 1)


def dircoupler_alpha(cfg: DirCouplerConfig, device=None, dtype=torch.float64):
    """Absorb at the bus ends, the readout guide ends, and each arm's upstream end.

    The arm's upstream taper is what makes the coupler directional in practice:
    without it the arm is a resonator, power that crossed in reflects off the
    open end, and the tap reads a standing wave whose phase has nothing to do
    with the delay it was placed for.
    """
    nx, ny = cfg.grid
    x = (torch.arange(nx, device=_dev(device), dtype=dtype) - (nx - 1) / 2) * cfg.dx
    y = (torch.arange(ny, device=_dev(device), dtype=dtype) - (ny - 1) / 2) * cfg.dx
    X, Y = torch.meshgrid(x, y, indexing="ij")
    yb, x0 = cfg._bus_y(), -(nx - 1) / 2 * cfg.dx
    L = cfg.bus_length()
    lo, hi = cfg.arm_y()

    outer = cfg.radius + cfg.guide_length
    start = outer - cfg.absorb_frac * cfg.guide_length
    ramp = torch.zeros_like(X)
    for cx, cy in cfg.centres():
        r = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        g = ((r - start) / (outer - start)).clamp(0.0, 1.0) ** 2
        g = torch.where(r <= outer + cfg.dx, g, torch.zeros_like(g))
        # the 270-degree guide is the link to the arm, not a readout port
        g = torch.where(((X - cx).abs() <= cfg.guide_width / 2) & (Y < cy),
                        torch.zeros_like(g), g)
        ramp = torch.maximum(ramp, g)
        # arm upstream taper
        a0 = cx - cfg.coupler_len
        arm = (Y >= yb + lo) & (Y <= yb + hi)
        ta = ((a0 + cfg.arm_absorb - X) / cfg.arm_absorb).clamp(0.0, 1.0) ** 2
        ta = torch.where(arm & (X >= a0), ta, torch.zeros_like(ta))
        ramp = torch.maximum(ramp, ta)
    bl = ((x0 + cfg.bus_absorb - X) / cfg.bus_absorb).clamp(0.0, 1.0) ** 2
    br = ((X - (x0 + L - cfg.bus_absorb)) / cfg.bus_absorb).clamp(0.0, 1.0) ** 2
    on_bus = (Y - yb).abs() <= cfg.bus_width / 2
    ramp = torch.maximum(ramp, torch.where(on_bus,
                                           torch.maximum(bl, br),
                                           torch.zeros_like(bl)))
    base = cfg.alpha * cfg.bus_alpha_mult
    a = base + (cfg.absorb_alpha - base) * ramp
    a = _tap_damped(cfg, X, Y, a)
    return a.reshape(nx, ny, 1, 1)


class DirCouplerArray(PolyTapArray):
    """Poly-tap bus with directional couplers. Five readout ports per tap."""

    def __init__(self, cfg: DirCouplerConfig, timesteps: int, device=None,
                 dtype=torch.float64):
        self.cfg = cfg
        nx, ny = cfg.grid
        mesh = MeshConfig(nx=nx, ny=ny, nz=1, dx=cfg.dx, dy=cfg.dx,
                          dz=cfg.thickness)
        solver = SolverConfig(dt=cfg.dt, timesteps=timesteps, checkpoint=False,
                              renormalize=True, demag=True)
        self.mask = dircoupler_mask(cfg, device, dtype)
        alpha = dircoupler_alpha(cfg, device, dtype) * self.mask
        self.rollout = LLGRollout(mesh, solver, A=cfg.A, alpha=alpha,
                                  Ms_ref=cfg.Ms)
        self.rollout.set_Ms(cfg.Ms * self.mask)
        self.h_zero = torch.zeros(nx, ny, 1, 3, device=_dev(device), dtype=dtype)
        self.m0 = None

        x = (torch.arange(nx, device=_dev(device), dtype=dtype) - (nx - 1) / 2) * cfg.dx
        y = (torch.arange(ny, device=_dev(device), dtype=dtype) - (ny - 1) / 2) * cfg.dx
        self._X, self._Y = torch.meshgrid(x, y, indexing="ij")
        self.disk_masks = torch.stack([
            (((self._X - cx) ** 2 + (self._Y - cy) ** 2) <= cfg.radius**2).to(dtype)
            for cx, cy in cfg.centres()])
        self.bus_mask = self._bus_region(dtype)
        self.inject_mask = self._inject_region(dtype)
        self._tap_masks = self._build_taps(dtype)
