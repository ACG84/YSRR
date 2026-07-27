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

The disks are permalloy, not YIG, and 20 nm thick rather than the common
10 nm. Both follow from one inequality: on a circular orbit the core speed is
``v = r omega(r)``, and with the Guslienko frequency the small-orbit ceiling
``omega0 R = (20/9) gamma mu0 Ms L / (4 pi)`` depends on **Ms L only** -- disk
radius cancels. Firing requires that ceiling to clear ``v_crit``: YIG misses
by an order of magnitude at any thickness worth patterning, and 10 nm Py sits
at 313 m/s, fractionally *below* 320 -- a knife edge where firing would exist
only at edge-adjacent orbits where the rigid-core ansatz is invalid. At
L = 20 nm the ceiling is 626 m/s and reversal happens at r/R ~ 0.5, squarely
where the model holds.

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

    Defaults are a permalloy disk 200 nm across and 20 nm thick; see the
    module docstring for why the disks are neither YIG nor thinner.
    """

    n_disks: int = 12
    R: float = 100e-9            # disk radius, m
    L: float = 20e-9             # thickness, m; see module docstring -- the
                                 # firing ceiling omega0*R ~ Ms*L must clear v_crit
    Ms: float = 800e3            # permalloy, A/m
    f_gyro: float | None = None  # Hz; None -> Guslienko thin-disk estimate
    alpha_eff: float = 0.005     # lumped damping ratio D/|G| at small orbit;
                                 # sets the memory horizon 1/(alpha_eff*omega0)
                                 # ~ 32 ns ~ 10 frames, matched to timescale,
                                 # not tuned on any task score
    beta_nl: float = 0.5         # damping stiffening: D(r) = D0 (1 + beta_nl r^2/R^2)
    kappa_nl: float = 0.3        # potential stiffening: F = -k X (1 + kappa_nl r^2/R^2)
    v_crit: float = 320.0        # core reversal speed, m/s (Py, Guslienko)
    contraction: float = 0.3     # orbit radius retained after a reversal
    refractory: float = 2e-9     # s; no re-fire inside this window
    # Waveguide-mediated coupling. Encasing each disk in a magnonic guide
    # replaces the near-field dipolar term -- which the 250 nm probe pitch
    # forces on us at a 50 nm edge gap -- with a coupling whose strength AND
    # delay are set by guide geometry. The delay is the point: the disks
    # cannot hold memory because firing resets them inside their own damping
    # horizon, but a delay line holds history OUTSIDE the disk, where a reset
    # cannot reach it. This is the Appeltant delay-reservoir architecture: a
    # nonlinear node plus a delay loop is a reservoir.
    #
    # Near FMR the group velocity is low (~100 m/s at a few mT bias), so a
    # 30 ns delay -- the whole 10-frame memory requirement -- is ~3 um of
    # on-chip guide.
    # Each guide is multi-port, and the ports share ONE emission budget: what
    # leaks to the neighbours cannot also be tapped for the decoder or sent
    # back down the return path. That constraint is the honest part of the
    # architecture -- routing power to the readout costs recruitment strength.
    #
    #   port_neighbour  leaks to the adjacent disks (recruitment)
    #   port_readout    taps to film2, the decoder
    #   port_return     the adjoint/backprop channel
    #
    # The return path carries its own delay, deliberately NOT equal to the
    # forward one. In a biased film the forward and backward group velocities
    # differ, so an adjoint signal arrives on a different schedule than the
    # forward one it must be compared against -- with a separate port that
    # asymmetry becomes a designed delay rather than an uncontrolled error.
    wg_coupling: float = 0.0     # delayed neighbour coupling, fraction of k
    wg_delay: float = 0.0        # s, neighbour propagation delay
    wg_feedback: float = 0.0     # delayed self-feedback, fraction of k
    wg_feedback_delay: float = 0.0   # s, round-trip delay of the self-loop
    port_neighbour: float = 1.0
    port_readout: float = 0.0
    port_return: float = 0.0
    wg_return_delay: float = 0.0     # s; asymmetric by design, see above

    def check_ports(self) -> None:
        total = self.port_neighbour + self.port_readout + self.port_return
        if total > 1.0 + 1e-9:
            raise ValueError(
                f"port fractions sum to {total:.3f} > 1: the guide would emit "
                "more power than the disk delivers to it")
    coupling: float = 0.0        # nearest-neighbour dipolar strength, fraction of k
    spike_kick: float = 0.0      # tangential kick to neighbours per spike, fraction of R
    phase_capture: float = 0.0   # 0..1: how far a reversal burst pulls each
                                 # neighbour's phase toward the burst phase.
                                 # The culling step of the annealing cycle: the
                                 # coherent burst injection-locks the cluster,
                                 # raising its order; radius is untouched, so
                                 # this is a pure correlation projection
    alpha_spread: float = 0.0    # +-fractional spread of alpha_eff across the
                                 # row; identical disks are one filter wearing
                                 # n masks, a spread makes a timescale bank
    temperature: float = 0.0     # K. Langevin noise on the core with variance
                                 # fixed by fluctuation-dissipation from the
                                 # damping already present -- no free knobs.
                                 # At 300 K the thermal orbit is ~1.5% of R and
                                 # velocity jitter ~3% of v_crit: features stay
                                 # informative, but the firing threshold smears
                                 # from a deterministic cliff into a
                                 # probability -- the continuous control
                                 # parameter an edge-of-chaos system needs
    drive_scale: float = 0.10    # peak drive force, fraction of k*R per unit
                                 # input. Calibrated so firing is an event, not
                                 # a carrier: at 0.15 the trial measured 0.44
                                 # spikes/disk/frame -- chatter that scrambles
                                 # the phase memory it is supposed to punctuate
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

        V = (D F + Gz z_hat x F) / (D^2 + Gz^2),   Gz = -G0 p

    which for pure harmonic confinement gives circular gyration at
    ``omega = k p / G0`` -- counterclockwise for p = +1 -- the analytic anchor
    the tests check against.

    The reversal is handled between RK4 steps: it is a discontinuity, and
    letting it inside the integrator stages would poison the derivatives.
    """

    K_B = 1.380649e-23

    def __init__(self, cfg: ThieleConfig, dtype=torch.float64, device="cpu",
                 noise_seed: int = 0):
        self.noise_gen = torch.Generator(device=device).manual_seed(noise_seed)
        self.cfg = cfg
        d = cfg.derived()
        self.G0, self.k, self.omega0, self.dt = d["G"], d["k"], d["omega0"], d["dt"]
        self.dtype, self.device = dtype, device

        n = cfg.n_disks
        self.alpha = cfg.alpha_eff * (
            1.0 + cfg.alpha_spread * torch.linspace(-1.0, 1.0, max(n, 2),
                                                    dtype=dtype, device=device)[:n])
        self.X = torch.zeros(n, 2, dtype=dtype, device=device)
        self.p = torch.ones(n, dtype=dtype, device=device)
        self.since_switch = torch.full((n,), 1e3, dtype=dtype, device=device)
        self.t = 0.0

        # Nearest-neighbour dipolar coupling along the row. Sukhostavets-form
        # anisotropy: for a bond along y_hat the y-y term carries the opposite
        # sign and twice the weight of x-x. Only the ratio matters here; the
        # magnitude is the `coupling` knob.
        self.coupling_pairs = [(i, i + 1) for i in range(n - 1)]
        self._pair_weights = torch.tensor([2.0, -1.0], dtype=dtype, device=device)

        # Circular history buffer for the delayed (waveguide) terms. Sized to
        # the longest delay in use; the buffer starts at the origin, which is
        # the physically right initial condition -- an empty guide.
        cfg.check_ports()
        self.d_nb = int(round(cfg.wg_delay / self.dt)) if cfg.wg_coupling else 0
        self.d_fb = (int(round(cfg.wg_feedback_delay / self.dt))
                     if cfg.wg_feedback else 0)
        self.d_ret = (int(round(cfg.wg_return_delay / self.dt))
                      if cfg.port_return else 0)
        self._hist_len = max(self.d_nb, self.d_fb, self.d_ret) + 1
        self.history = (torch.zeros(self._hist_len, n, 2, dtype=dtype, device=device)
                        if self._hist_len > 1 else None)
        self._hp = 0                      # write cursor

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
            # The displaced-vortex moment is PERPENDICULAR to the core
            # displacement (m ~ z_hat x X), so for a bond along y_hat the
            # dipolar double-weight term lands on the x coordinate:
            # W = mu (y_i y_j - 2 x_i x_j), F = -dW/dX.
            #
            # Vectorised over the chain rather than looped over pairs: the
            # loop was 44 scalar index-assignments per force evaluation, 176
            # per RK4 step, and measured at 84% of total runtime on a state
            # of 24 numbers. Slicing says the same thing in four ops.
            w = self._pair_weights * (cfg.coupling * self.k)
            F[:-1] += w * X[1:]      # each disk from its right neighbour
            F[1:] += w * X[:-1]      # and from its left

        if self.history is not None:
            # Delayed terms are held constant across the RK4 stages: they are
            # history, not a function of the stage state, and tau >> dt.
            if cfg.wg_coupling:
                Xd = self.history[(self._hp - self.d_nb) % self._hist_len]
                g = cfg.wg_coupling * self.k * cfg.port_neighbour
                F[:-1] += g * Xd[1:]
                F[1:] += g * Xd[:-1]
            if cfg.wg_feedback:
                Xf = self.history[(self._hp - self.d_fb) % self._hist_len]
                F += cfg.wg_feedback * self.k * Xf
        return F

    def port_taps(self) -> dict:
        """What each port is carrying right now, in units of R.

        The readout tap is what film2 sees; the return tap is what an adjoint
        scheme would receive. Both are delayed copies of the disk state,
        scaled by their port fraction -- power the disk no longer has.
        """
        cfg, R = self.cfg, self.cfg.R
        out = {}
        if cfg.port_readout:
            out["readout"] = cfg.port_readout * self.X / R
        if cfg.port_return and self.history is not None:
            Xr = self.history[(self._hp - self.d_ret) % self._hist_len]
            out["return"] = cfg.port_return * Xr / R
        return out

    def _velocity(self, X: torch.Tensor, drive: torch.Tensor, t: float) -> torch.Tensor:
        cfg = self.cfg
        F = self._force(X, drive, t)
        r2 = (X * X).sum(-1) / cfg.R**2
        D = self.alpha * self.G0 * (1.0 + cfg.beta_nl * r2)
        # Signed z-component of the gyrovector: G = -(2 pi Ms L / gamma) p z_hat,
        # so Gz = -G0 p. Getting this sign wrong mirror-inverts every orbit --
        # p = +1 must gyrate counterclockwise, as established experimentally.
        Gp = -self.G0 * self.p
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
        if self.cfg.temperature > 0:
            # Euler-Maruyama noise increment after the deterministic RK4 step
            # (RK4 inside the stages would not converge for an SDE). The
            # mobility is a scaled rotation, so the displacement noise is
            # isotropic: sigma^2 = 2 D k_B T dt / (D^2 + G^2), FDT with the
            # same D(r) the deterministic dynamics use.
            r2 = (self.X * self.X).sum(-1) / self.cfg.R**2
            D = self.alpha * self.G0 * (1.0 + self.cfg.beta_nl * r2)
            sigma = (2 * D * self.K_B * self.cfg.temperature * dt
                     / (D * D + self.G0**2)).sqrt()
            self.X = self.X + sigma.unsqueeze(-1) * torch.randn(
                self.X.shape, generator=self.noise_gen, dtype=self.dtype,
                device=self.device)
        if self.history is not None:
            # Advance the guide: what leaves the disk now arrives at the far
            # port tau later. Written after the state update, before the spike
            # test, so a reversal cannot retroactively edit what already left.
            self._hp = (self._hp + 1) % self._hist_len
            self.history[self._hp] = self.X
        self.t = t + dt
        self.since_switch += dt

        speed = self._velocity(self.X, drive, self.t).norm(dim=-1)
        fired = (speed > self.cfg.v_crit) & (self.since_switch > self.cfg.refractory)
        if fired.any():
            idx = fired.nonzero(as_tuple=True)[0]
            # Burst phase, read BEFORE the contraction: the emitted packet is
            # coherent with the core's gyration at the instant of reversal.
            burst_phase = torch.atan2(self.X[idx, 1], self.X[idx, 0])
            self.p[idx] = -self.p[idx]
            self.X[idx] *= self.cfg.contraction
            self.since_switch[idx] = 0.0
            for i, phi_star in zip(idx.tolist(), burst_phase.tolist()):
                for j in (i - 1, i + 1):
                    if not 0 <= j < self.cfg.n_disks:
                        continue
                    Xj = self.X[j]
                    r = Xj.norm().clamp_min(1e-3 * self.cfg.R)
                    if self.cfg.phase_capture:
                        # Injection locking by the coherent burst: pull the
                        # neighbour's ENVELOPE phase toward the burst's
                        # envelope phase, radius untouched. The frame matters:
                        # aligning lab-frame positions is undone within half a
                        # gyration period for counter-rotating (mixed-
                        # polarity) pairs, which is why the lab-frame version
                        # measured no coherence gain at spikes. A common
                        # coherent pulse pins each receiver's envelope
                        # regardless of its rotation sense -- so the cull is
                        # applied to the same I/Q variables the readout sees.
                        w0t = self.omega0 * self.t
                        s_i = -float(self.p[i])         # emitter's sense pre-flip
                        psi_star = phi_star - s_i * w0t
                        phi = math.atan2(float(Xj[1]), float(Xj[0]))
                        psi = phi - float(self.p[j]) * w0t
                        dpsi = math.atan2(math.sin(psi_star - psi),
                                          math.cos(psi_star - psi))
                        new = phi + self.cfg.phase_capture * dpsi
                        self.X[j] = r * torch.tensor(
                            [math.cos(new), math.sin(new)],
                            dtype=self.dtype, device=self.device)
                    if self.cfg.spike_kick:
                        # Energy transfer: tangential, the direction that
                        # pumps orbit.
                        Xj = self.X[j]
                        tangent = torch.stack([-Xj[1], Xj[0]]) / r
                        self.X[j] = Xj + self.cfg.spike_kick * self.cfg.R * tangent
        self._speed = speed
        return fired.to(self.dtype)

    def run_frame(self, drive: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Integrate one input frame under constant ``drive`` (n_disks,).

        Returns the per-disk feature vector, (n_disks, 7):
        ``I/R, Q/R, rms orbit/R, end speed/v_crit, polarity, spikes, end r^2/R^2``.

        I/Q are the core position demodulated into each disk's own rotating
        frame (rotation by -p * omega0 * t) -- lock-in detection against the
        drive clock, which is how an experiment would read the disk anyway.
        Lab-frame sampling only works if every relevant frequency is
        commensurate with the frame length; coupling splits the collective
        modes away from omega0, so lab-frame features turn the stroboscopic
        map time-varying again. Envelopes do not care.
        """
        R = self.cfg.R
        spikes = torch.zeros(self.cfg.n_disks, dtype=self.dtype, device=self.device)
        r2_accum = torch.zeros_like(spikes)
        for _ in range(n_steps):
            spikes += self.step(drive)
            r2_accum += (self.X * self.X).sum(-1) / R**2
        r2_end = (self.X * self.X).sum(-1) / R**2
        theta = self.p * (self.omega0 * self.t)
        c, s = torch.cos(theta), torch.sin(theta)
        I = (c * self.X[:, 0] + s * self.X[:, 1]) / R
        Q = (-s * self.X[:, 0] + c * self.X[:, 1]) / R
        return torch.stack([
            I,
            Q,
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
