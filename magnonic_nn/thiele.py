"""Thiele-equation vortex disks: the spiking layer of the reservoir.

A magnetic vortex confined in a nanodisk reduces, for rigid-core motion, to a
two-dimensional ODE for the core position **X** (Thiele 1973):

    G x dX/dt - D dX/dt + F(X, t) = 0

with gyrovector ``G = -(2 pi Ms L / gamma) p z_hat`` (polarity ``p = +-1``),
damping ``D``, and total in-plane force ``F``. Solving for the velocity gives a
first-order system: the core *gyrates* around the disk centre at
``omega_0 = k/|G|`` rather than oscillating through it.

What makes this a spiking neuron rather than a linear resonator:

* **Fire.** Above a critical core speed (~320 m/s in permalloy, Guslienko's
  criterion -- weakly dependent on disk geometry) the core polarity reverses.
  The gyration *sense* reverses with it, and the orbit collapses: the stored
  energy is dumped into a spin-wave burst. Threshold, reset, and an emitted
  pulse -- the neuron trifecta, in ferromagnetic resonance hardware.
* **Exhaustion.** Damping grows with orbit radius (``D(r)``), and a hard
  refractory window follows each reversal, so a disk driven relentlessly
  fires, resets, and cannot immediately fire again.
* **Recruitment.** Disks couple through stray dipolar fields. A hard-gyrating
  disk drags its neighbours toward larger orbits -- until it fires, its sense
  flips, and the phase relation it imposed inverts. On top of the quasistatic
  coupling, the reversal burst itself delivers a tangential kick to the
  neighbours (``spike_kick``), which is the spike-propagation path.

The disks are permalloy, not YIG. That is deliberate and physical: Py-on-YIG
hybrids are routine experimentally, and YIG's low Ms puts the gyrotropic
frequency near 100 MHz, where the core cannot reach critical velocity inside
the disk (``v = r omega`` tops out at ~80 m/s). Permalloy at R = 100 nm gyrates
at ~500 MHz and crosses 320 m/s at accessible orbit radii.

Modelling leaps, stated plainly: the incident spin-wave power reaching a disk
is rectified into a *linearly polarised* force at the gyrotropic frequency
(the nonlinear magnon-to-gyration pumping of Philippe & Kim, arXiv:2507.19865,
collapsed to its envelope); a linear polarisation drives either core polarity
with equal efficiency, which keeps a reversed disk in the game. The reversal
is modelled as an instantaneous polarity flip plus orbit contraction rather
than resolved core dynamics. Both are trial-level abstractions to be replaced
by micromagnetics on a graded mesh when this graduates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

__all__ = ["ThieleConfig", "ThieleDisks"]

GAMMA_T = 1.760859630e11  # gyromagnetic ratio, rad/(s*T)
MU_0 = 4.0e-7 * math.pi


@dataclass
class ThieleConfig:
    """Geometry, material and dynamics of the disk array.

    Defaults are a permalloy disk 200 nm across and 10 nm thick; see the
    module docstring for why the disks are not YIG.
    """

    n_disks: int = 12
    R: float = 100e-9            # disk radius, m
    L: float = 10e-9             # thickness, m
    Ms: float = 800e3            # permalloy, A/m
    f_gyro: float | None = None  # Hz; None -> Guslienko thin-disk estimate
    alpha_eff: float = 0.01      # lumped damping ratio D/|G| at small orbit
    beta_nl: float = 0.5         # damping stiffening: D(r) = D0 (1 + beta_nl r^2/R^2)
    kappa_nl: float = 0.3        # potential stiffening: F = -k X (1 + kappa_nl r^2/R^2)
    v_crit: float = 320.0        # core reversal speed, m/s (Py, Guslienko)
    contraction: float = 0.3     # orbit radius retained after a reversal
    refractory: float = 2e-9     # s; no re-fire inside this window
    coupling: float = 0.0        # nearest-neighbour dipolar strength, fraction of k
    spike_kick: float = 0.0      # tangential kick to neighbours per spike, fraction of R
    drive_scale: float = 0.15    # peak drive force, fraction of k*R per unit input
    dt: float | None = None      # s; None -> gyration period / 100

    def derived(self) -> dict:
        """Gyrovector magnitude, stiffness, frequency, timestep -- SI."""
        G = 2 * math.pi * self.Ms * self.L / GAMMA_T
        if self.f_gyro is not None:
            omega0 = 2 * math.pi * self.f_gyro
        else:
            # Guslienko's side-charge-free thin-disk estimate,
            # omega0 = (20/9) gamma Ms L / R in Gaussian form -> SI with mu0/(4 pi).
            omega0 = (20.0 / 9.0) * GAMMA_T * MU_0 * self.Ms * self.L / (4 * math.pi * self.R)
        k = omega0 * G
        dt = self.dt if self.dt is not None else (2 * math.pi / omega0) / 100.0
        return {"G": G, "k": k, "omega0": omega0, "f0": omega0 / (2 * math.pi), "dt": dt}


class ThieleDisks:
    """A row of coupled, spiking vortex disks, integrated with RK4.

    State per disk: core position ``X`` (units of metres), polarity ``p``, and
    time since the last reversal. The core velocity is not a state variable --
    the Thiele equation is first order, and velocity follows algebraically:

        V = (D F + G p z_hat x F) / (D^2 + G^2)

    which for pure harmonic confinement gives circular gyration at
    ``omega = k p / G`` (sense set by polarity), the analytic anchor the tests
    check against.

    The reversal is handled between RK4 steps: it is a discontinuity, and
    letting it inside the integrator stages would poison the derivatives.
    """

    def __init__(self, cfg: ThieleConfig, dtype=torch.float64, device="cpu"):
        self.cfg = cfg
        d = cfg.derived()
        self.G0, self.k, self.omega0, self.dt = d["G"], d["k"], d["omega0"], d["dt"]
        self.dtype, self.device = dtype, device

        n = cfg.n_disks
        self.X = torch.zeros(n, 2, dtype=dtype, device=device)
        self.p = torch.ones(n, dtype=dtype, device=device)
        self.since_switch = torch.full((n,), 1e3, dtype=dtype, device=device)
        self.t = 0.0

        # Nearest-neighbour dipolar coupling along the row. Sukhostavets-form
        # anisotropy: for a bond along y_hat the y-y term carries the opposite
        # sign and twice the weight of x-x. Only the ratio matters here; the
        # magnitude is the `coupling` knob.
        self.coupling_pairs = [(i, i + 1) for i in range(n - 1)]

    # ------------------------------------------------------------------ forces
    def _force(self, X: torch.Tensor, drive: torch.Tensor, t: float) -> torch.Tensor:
        """Total in-plane force on each core at positions ``X`` (n, 2)."""
        cfg, R = self.cfg, self.cfg.R
        r2 = (X * X).sum(-1, keepdim=True) / R**2
        F = -self.k * X * (1.0 + cfg.kappa_nl * r2)

        # Linearly polarised drive at the gyrotropic frequency; amplitude is
        # the per-disk rectified spin-wave power routed to this disk.
        osc = math.cos(self.omega0 * t)
        F[:, 0] += cfg.drive_scale * self.k * R * drive * osc

        if cfg.coupling:
            mu = cfg.coupling * self.k
            for i, j in self.coupling_pairs:
                # bond along y_hat: W = mu (x_i x_j - 2 y_i y_j)
                fx_i = -mu * X[j, 0]
                fy_i = 2 * mu * X[j, 1]
                F[i, 0] += fx_i
                F[i, 1] += fy_i
                F[j, 0] += -mu * X[i, 0]
                F[j, 1] += 2 * mu * X[i, 1]
        return F

    def _velocity(self, X: torch.Tensor, drive: torch.Tensor, t: float) -> torch.Tensor:
        cfg = self.cfg
        F = self._force(X, drive, t)
        r2 = (X * X).sum(-1) / cfg.R**2
        D = cfg.alpha_eff * self.G0 * (1.0 + cfg.beta_nl * r2)
        Gp = self.G0 * self.p
        det = (D * D + Gp * Gp).unsqueeze(-1)
        zxF = torch.stack([-F[:, 1], F[:, 0]], dim=-1)
        return (D.unsqueeze(-1) * F + Gp.unsqueeze(-1) * zxF) / det

    # ------------------------------------------------------------------ stepping
    def step(self, drive: torch.Tensor) -> torch.Tensor:
        """One RK4 step; returns the per-disk spike indicator (0/1)."""
        X, dt, t = self.X, self.dt, self.t
        k1 = self._velocity(X, drive, t)
        k2 = self._velocity(X + 0.5 * dt * k1, drive, t + 0.5 * dt)
        k3 = self._velocity(X + 0.5 * dt * k2, drive, t + 0.5 * dt)
        k4 = self._velocity(X + dt * k3, drive, t + dt)
        self.X = X + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.t = t + dt
        self.since_switch += dt

        speed = self._velocity(self.X, drive, self.t).norm(dim=-1)
        fired = (speed > self.cfg.v_crit) & (self.since_switch > self.cfg.refractory)
        if fired.any():
            idx = fired.nonzero(as_tuple=True)[0]
            self.p[idx] = -self.p[idx]
            self.X[idx] *= self.cfg.contraction
            self.since_switch[idx] = 0.0
            if self.cfg.spike_kick:
                # The reversal burst kicks the neighbours tangentially --
                # tangential because that is the direction that pumps orbit.
                for i in idx.tolist():
                    for j in (i - 1, i + 1):
                        if 0 <= j < self.cfg.n_disks:
                            Xj = self.X[j]
                            r = Xj.norm().clamp_min(1e-3 * self.cfg.R)
                            tangent = torch.stack([-Xj[1], Xj[0]]) / r
                            self.X[j] = Xj + self.cfg.spike_kick * self.cfg.R * tangent
        self._speed = speed
        return fired.to(self.dtype)

    def run_frame(self, drive: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Integrate one input frame under constant ``drive`` (n_disks,).

        Returns the per-disk feature vector, (n_disks, 7):
        ``x/R, y/R, rms orbit/R, end speed/v_crit, polarity, spikes, end r^2/R^2``.
        """
        R = self.cfg.R
        spikes = torch.zeros(self.cfg.n_disks, dtype=self.dtype, device=self.device)
        r2_accum = torch.zeros_like(spikes)
        for _ in range(n_steps):
            spikes += self.step(drive)
            r2_accum += (self.X * self.X).sum(-1) / R**2
        r2_end = (self.X * self.X).sum(-1) / R**2
        return torch.stack([
            self.X[:, 0] / R,
            self.X[:, 1] / R,
            (r2_accum / n_steps).sqrt(),
            self._speed / self.cfg.v_crit,
            self.p.clone(),
            spikes,
            r2_end,
        ], dim=-1)

    def reset(self):
        self.X.zero_()
        self.p.fill_(1.0)
        self.since_switch.fill_(1e3)
        self.t = 0.0
