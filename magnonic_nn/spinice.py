"""3D multilayered artificial spin-vortex ice: the element, in this solver.

WHY THIS EXISTS, given that the poly-tap bus already does not work.

The bus architecture put memory in a propagating wave and nonlinearity in a
coupled disk, and every route to keeping them apart is now measured closed:
off-centre coupling makes isolation worse, a lossy barrier is bypassed by the
stray field (6.25x inside itself, 0.87x beyond), and DMI gives a far-field
asymmetry that is non-monotonic and never exceeds 7.3%. The ring that would
have made the coupling the mechanism rather than the fault needs a round-trip
budget the couplers cannot supply: bus->disk 0.201 and disk->bus 0.073 give
kappa_in * kappa_out = 0.0147, -36.6 dB before a nanometre of propagation,
against roughly -8 dB required, and the whole coupler axis buys 1.16x.

Three of those are the same fact: a passive reciprocal element weakly coupled
to a shared line cannot be given a preferred direction. The fourth -- the
readout being blind by parity to the degree-2 family it is scored on, because
the demodulator has no zero-frequency bin -- is a property of coherent lock-in
detection, not of the device.

This architecture is not a repair of that one. It changes where memory lives.

    Dion, Stenning, Vanstone, Holder, Sultana, Alatteili, Martinez, Taghipour
    Kaffash, Kimura, Oulton, Branford, Kurebayashi, Iacocca, Jungfleisch,
    Gartside, "Ultrastrong magnon-magnon coupling and chiral spin-texture
    control in a dipolar 3D multilayered artificial spin-vortex ice",
    Nat. Commun. 15 (2024). doi:10.1038/s41467-024-48080-z

Memory is the MICROSTATE -- hysteretic, non-volatile, with no decay length --
rather than a wave in flight, so the ~1 um loaded-bus decay that capped the
delay line does not apply. Nonlinearity is switching plus magnon-magnon
coupling at a normalised rate of 0.57, which is the ultrastrong regime, against
the 0.0147 that closed the ring. And the readout in this family is the FMR
spectrum ("spin-wave fingerprinting", Gartside et al., Nat. Nanotech. 17, 460
(2022)), which is a POWER spectrum and therefore an even-order detector: the
parity blindness does not arise.

WHAT IS BEING TESTED FIRST, and why it is the right first thing.

This project measured, on the day this module was written, that its ported
vortex disks are not vortices: the core is expelled between relax step 400 and
800 while every run relaxes 8000, and the six 80 nm guides give the flux a
closure path a bare disk does not have. Every interpretation resting on "the
disk's mode" fell with it.

The element here is named for holding vortices -- artificial spin-VORTEX ice,
four states per layer, two macrospin and two vortex. So the first question is
not what it computes. It is whether a 550 x 140 nm stadium in a two-layer
dipolar stack keeps a core through relaxation in THIS solver, where the disk
did not. That is a direct rematch of a measured failure and it gates the rest.

GEOMETRY, from the paper's Methods, verbatim where given:

    islands       550 x 140 x 90 nm, stadium (rectangle with semicircular caps)
    stack         SiO2 / NiFe 30 / Al 35 / NiFe 20 / Al 5 nm, substrate upward
    hard layer    30 nm NiFe, LOWER, switches at higher field
    soft layer    20 nm NiFe, UPPER, switches at lower field
    offset        50 nm lateral displacement in y between the layers, from
                  shadow deposition. This breaks the dipolar coupling symmetry
                  between the two chiralities and is what makes vortex chirality
                  selectable rather than random.
    lattice       square ASI, 125 nm vertex spacing, island-end to vertex-centre
    material      Ms = 800 kA/m, A = 13 pJ/m, alpha = 0.001 (their MuMax3 runs)

Ms and A are identical to the values this project already uses for permalloy.
Their alpha is 0.001 against the 0.008 here, which sharpens the FMR peaks the
readout is made of; it is a modelling choice, and both are carried as config.

CELL SIZE. Theirs is 4.198 nm laterally and 10 nm normal, the lateral figure
chosen to divide their lattice period. A single island has no such constraint,
so this uses 5 nm laterally -- the size the vortex work here already uses, and
at the exchange length of permalloy (~5.3 nm) -- and 5 nm normal, which divides
every layer of the stack exactly: 30/35/20/5 nm becomes 6/7/4/1 cells for 18
in total. A 10 nm normal cell does NOT divide the 35 nm spacer or the 5 nm cap.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .config import MU_0, MeshConfig, SolverConfig
from .solver import LLGRollout
from .vortex import _dev   # resolve device to the CURRENT torch default


@dataclass
class ASVIConfig:
    """One 3D multilayered artificial spin-vortex ice nanoisland."""

    # ---- island, from the paper's Methods
    length: float = 550e-9         # along x, including both caps
    width: float = 140e-9          # along y; cap radius is width/2
    # Stack from the substrate up, as (thickness, is_magnetic). The Al spacer
    # is carried in the mesh with Ms = 0 rather than omitted: it is 35 nm of
    # geometric separation and the demag field has to see it, because the
    # inter-layer dipolar coupling IS the mechanism here.
    stack: tuple = ((30e-9, True), (35e-9, False), (20e-9, True), (5e-9, False))
    layer_offset: float = 50e-9    # lateral shift in y of every layer above
                                   # the first magnetic one

    # ---- mesh
    dx: float = 5e-9               # lateral; permalloy exchange length is 5.3 nm
    dz: float = 5e-9               # normal; divides 30/35/20/5 exactly
    margin: float = 40e-9          # vacuum ring, keeps demag off the boundary

    # ---- material
    Ms: float = 800e3
    A: float = 13e-12
    alpha: float = 0.001
    dt: float = 1e-12

    # ---- initial-state shape
    core_width: float = 10e-9

    def layer_slices(self):
        """(z0, z1, is_magnetic) per stack entry, in cell indices."""
        out, z = [], 0
        for t, mag in self.stack:
            n = int(round(t / self.dz))
            if abs(n * self.dz - t) > 1e-15:
                raise ValueError(
                    f"stack layer {t*1e9:g} nm is not a whole number of "
                    f"{self.dz*1e9:g} nm cells")
            out.append((z, z + n, mag))
            z += n
        return out

    def magnetic_layers(self):
        """Just the magnetic ones, bottom first: (z0, z1, y_offset)."""
        out, seen = [], 0
        for (z0, z1, mag) in self.layer_slices():
            if mag:
                out.append((z0, z1, 0.0 if seen == 0 else self.layer_offset))
                seen += 1
        return out

    @property
    def nz(self) -> int:
        return sum(int(round(t / self.dz)) for t, _ in self.stack)

    def y_span(self):
        """(lo, hi) of every magnetic layer's footprint, before margin."""
        lo = min(-self.width / 2 + o for _, _, o in self.magnetic_layers())
        hi = max(self.width / 2 + o for _, _, o in self.magnetic_layers())
        return lo, hi

    @property
    def grid(self) -> tuple[int, int, int]:
        nx = int(np.ceil((self.length + 2 * self.margin) / self.dx))
        lo, hi = self.y_span()
        ny = int(np.ceil((hi - lo + 2 * self.margin) / self.dx))
        return (nx + nx % 2, ny + ny % 2, self.nz)

    def y_centre(self) -> float:
        """Mesh y of the footprint centre, so both layers fit symmetrically."""
        lo, hi = self.y_span()
        return 0.5 * (lo + hi)

    def layer_cy(self, k: int) -> float:
        """Mesh y of magnetic layer ``k``'s centre.

        One helper rather than `off - y_centre()` repeated at every site. The
        mask, the initial state and the circulation readout must agree on where
        a layer is, and if they drift the vortex gets initialised off its own
        centre and the chirality is read about the wrong axis -- both of which
        look like physics rather than like a bug.
        """
        return self.magnetic_layers()[k][2] - self.y_centre()

    def thickness(self) -> float:
        return self.nz * self.dz

    def aspect(self) -> float:
        return self.length / self.width


def _stadium(X, Y, cx, cy, length, width):
    """Rectangle with semicircular caps: |x| <= L/2, |y| <= W/2, rounded ends."""
    r = width / 2
    flat = length / 2 - r
    dX, dY = X - cx, Y - cy
    body = (dX.abs() <= flat) & (dY.abs() <= r)
    capL = ((dX + flat) ** 2 + dY**2) <= r**2
    capR = ((dX - flat) ** 2 + dY**2) <= r**2
    return body | capL | capR


def asvi_masks(cfg: ASVIConfig, device=None, dtype=torch.float64):
    """Per-cell magnetic mask ``(nx, ny, nz, 1)`` and one mask per magnetic layer.

    The layer masks are returned separately because the whole point of the
    architecture is that the two layers hold INDEPENDENT states -- 4 each, 16
    per island -- so every state-setting and every readout is per layer.
    """
    nx, ny, nz = cfg.grid
    x = (torch.arange(nx, device=_dev(device), dtype=dtype) - (nx - 1) / 2) * cfg.dx
    y = (torch.arange(ny, device=_dev(device), dtype=dtype) - (ny - 1) / 2) * cfg.dx
    X, Y = torch.meshgrid(x, y, indexing="ij")
    yc = cfg.y_centre()

    full = torch.zeros(nx, ny, nz, dtype=dtype, device=_dev(device))
    layers = []
    for (z0, z1, off) in cfg.magnetic_layers():
        # Layer centre is (0, off - yc): the mesh is centred on the FOOTPRINT
        # of both layers, not on the lower one, so a 50 nm offset puts them at
        # -25 and +25 rather than 0 and +50 and neither sits against a wall.
        m2d = _stadium(X, Y, 0.0, cfg.layer_cy(len(layers)),
                       cfg.length, cfg.width).to(dtype)
        lay = torch.zeros(nx, ny, nz, dtype=dtype, device=_dev(device))
        lay[:, :, z0:z1] = m2d.unsqueeze(-1)
        layers.append(lay)
        full = torch.maximum(full, lay)
    return full.unsqueeze(-1), torch.stack(layers), (X, Y)


# ---------------------------------------------------------------- states

MACROSPIN_POS = "macro+"
MACROSPIN_NEG = "macro-"
VORTEX_ACW = "vortex_acw"
VORTEX_CW = "vortex_cw"
STATES = (MACROSPIN_POS, MACROSPIN_NEG, VORTEX_ACW, VORTEX_CW)
"""The four states each magnetic layer may assume, per the paper.

Two macrospin and two vortex; 4 per layer and 4^2 = 16 per 3D island, which is
where the 16^N microstate space comes from. The two vortex states differ only
in circulation sense, and are degenerate WITHOUT the inter-layer offset -- the
50 nm shadow-deposition shift is what lifts that degeneracy and makes chirality
selectable.
"""


def layer_state(state, X, Y, cy, cfg, polarity=1, dtype=torch.float64):
    """In-plane (and core) magnetisation pattern for one layer, as (nx, ny, 3)."""
    mx = torch.zeros_like(X); my = torch.zeros_like(X); mz = torch.zeros_like(X)
    if state == MACROSPIN_POS:
        mx += 1.0
    elif state == MACROSPIN_NEG:
        mx -= 1.0
    elif state in (VORTEX_ACW, VORTEX_CW):
        c = 1.0 if state == VORTEX_ACW else -1.0
        dX, dY = X, Y - cy
        r = torch.sqrt(dX**2 + dY**2).clamp_min(1e-18)
        mz = polarity * torch.exp(-(r / cfg.core_width) ** 2)
        ip = torch.sqrt((1 - mz**2).clamp_min(0.0))
        mx = -c * ip * dY / r
        my = c * ip * dX / r
    else:
        raise ValueError(f"unknown state {state!r}; expected one of {STATES}")
    return torch.stack([mx, my, mz], dim=-1)


class ASVIIsland:
    """One 3D multilayered nanoisland: two magnetic layers, dipolar-coupled."""

    def __init__(self, cfg: ASVIConfig, timesteps: int, device=None,
                 dtype=torch.float64):
        self.cfg = cfg
        nx, ny, nz = cfg.grid
        mesh = MeshConfig(nx=nx, ny=ny, nz=nz, dx=cfg.dx, dy=cfg.dx, dz=cfg.dz)
        solver = SolverConfig(dt=cfg.dt, timesteps=timesteps, checkpoint=False,
                              renormalize=True, demag=True)
        self.mask, self.layer_masks, (self._X, self._Y) = asvi_masks(
            cfg, device, dtype)
        alpha = torch.full_like(self.mask, cfg.alpha) * self.mask
        self.rollout = LLGRollout(mesh, solver, A=cfg.A, alpha=alpha,
                                  Ms_ref=cfg.Ms)
        self.rollout.set_Ms(cfg.Ms * self.mask)
        self.h_zero = torch.zeros(nx, ny, nz, 3, device=_dev(device), dtype=dtype)
        self.m0 = None

    @property
    def n_layers(self) -> int:
        return self.layer_masks.shape[0]

    def initial_state(self, states, polarities=None, dtype=torch.float64):
        """Build ``m`` for a per-layer list of states, e.g. ``["macro+", "vortex_acw"]``."""
        cfg = self.cfg
        if len(states) != self.n_layers:
            raise ValueError(f"{len(states)} states for {self.n_layers} layers")
        pol = polarities or [1] * self.n_layers
        nx, ny, nz = cfg.grid
        m = torch.zeros(nx, ny, nz, 3, dtype=dtype)
        m[:, :, :, 0] = 1.0                      # vacuum: harmless, masked out
        for k, st in enumerate(states):
            pat = layer_state(st, self._X, self._Y, cfg.layer_cy(k), cfg,
                              pol[k], dtype)
            sel = self.layer_masks[k] > 0
            for c in range(3):
                m[:, :, :, c] = torch.where(sel, pat[:, :, c:c+1].expand(-1, -1, nz),
                                            m[:, :, :, c])
        m = m / m.norm(dim=-1, keepdim=True).clamp_min(1e-12) * self.mask
        return m

    def relax(self, states, steps=8000, alpha_relax=0.5, dtype=torch.float64):
        m = self.initial_state(states, dtype=dtype)
        self.m0 = self.rollout.relax(m, self.h_zero, steps, alpha_relax)
        return self.m0

    # ------------------------------------------------------------- readout

    def layer_core(self, m, k):
        """(peak |m_z|, mean m_z) over magnetic layer ``k``.

        The quantity that caught the ported disk. A vortex has |m_z| ~ 1 at its
        core and a mean of order the core area over the island area; a state
        that has gone planar reads 0.00 on both, which is exactly what the
        disks returned after a check that had previously read the whole mask
        and could not fail.
        """
        w = self.layer_masks[k]
        mz = m[:, :, :, 2] * w
        tot = w.sum().clamp_min(1e-30)
        return float(mz.abs().max()), float(mz.sum() / tot)

    def layer_moment(self, m, k):
        """Net in-plane moment (mx, my) of layer ``k``, normalised.

        Separates the two macrospin states from each other and both from a
        vortex: a saturated layer reads |mx| ~ 1, a flux-closed one ~ 0.
        """
        w = self.layer_masks[k]
        tot = w.sum().clamp_min(1e-30)
        return (float((m[:, :, :, 0] * w).sum() / tot),
                float((m[:, :, :, 1] * w).sum() / tot))

    def layer_circulation(self, m, k):
        """Mean of (r_hat x m)_z over the layer: +1 anticlockwise, -1 clockwise.

        Chirality needs its own number. |m_z| says a core is present and the
        net moment says the layer is flux-closed, but neither distinguishes ACW
        from CW -- and the 50 nm inter-layer offset exists precisely to select
        between them, so a run that cannot read chirality cannot see the knob
        the paper is built on.
        """
        cfg = self.cfg
        dX, dY = self._X, self._Y - cfg.layer_cy(k)
        r = torch.sqrt(dX**2 + dY**2).clamp_min(1e-18)
        # (r_hat x m)_z = (x*my - y*mx)/r
        w = self.layer_masks[k]
        circ = ((dX.unsqueeze(-1) * m[:, :, :, 1]
                 - dY.unsqueeze(-1) * m[:, :, :, 0]) / r.unsqueeze(-1)) * w
        return float(circ.sum() / w.sum().clamp_min(1e-30))
