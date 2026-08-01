"""Differentiable fixed-step LLG rollout.

The physics -- exchange and demagnetisation fields -- comes from magnum.np's
validated field terms. What this module adds is the time integration:

* **fixed-step RK4**, so a rollout has a deterministic number of steps and
  therefore a deterministic cost and memory footprint. Adaptive solvers make
  the graph depth depend on the design variables, which is awkward to train
  against and makes epochs unpredictably expensive.
* **backpropagation through time with gradient checkpointing**, which gives
  exact gradients of the discrete rollout. magnum.np's ``TorchDiffEqAdjoint``
  is the memory-lean alternative -- it reconstructs the trajectory by
  integrating backwards, trading gradient accuracy for storage. Here the wave
  is driven and oscillatory, precisely the regime where backwards
  reconstruction accumulates error, so exact BPTT is the default.

Memory: with ``T`` steps split into segments of ``c``, peak activation storage
is ``O(T/c + c)`` states, minimised at ``c = sqrt(T)``. For a 100x100 mesh and
600 steps that is ~25 stored states instead of 600.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import torch
from torch.utils.checkpoint import checkpoint

from ._compat import import_magnumnp
from .config import SolverConfig

_magnumnp = import_magnumnp()
from magnumnp import (  # noqa: E402
    DemagField,
    ExchangeField,
    InterfaceDMIField,
    Mesh,
    State,
    constants,
)

__all__ = ["LLGRollout", "RolloutResult", "estimate_max_timestep"]


class RolloutResult:
    """Container for a driven rollout.

    :ivar intensities: ``(n_probes,)`` time-integrated probe intensities.
    :ivar m: final magnetisation, ``(nx, ny, nz, 3)``.
    :ivar traces: ``(T, n_probes)`` per-step probe intensities, or ``None``.
    :ivar snapshots: ``(n_snapshots, nx, ny, nz, 3)`` recorded states, or ``None``.
    :ivar snapshot_steps: step index of each snapshot.
    """

    __slots__ = ("intensities", "m", "traces", "snapshots", "snapshot_steps")

    def __init__(self, intensities, m, traces=None, snapshots=None, snapshot_steps=None):
        self.intensities = intensities
        self.m = m
        self.traces = traces
        self.snapshots = snapshots
        self.snapshot_steps = snapshot_steps


class LLGRollout:
    """Fixed-step RK4 integrator for the LLG equation on a magnum.np mesh.

    :param mesh_cfg: mesh geometry.
    :param solver_cfg: timestep and checkpointing options.
    :param A: exchange stiffness (J/m).
    :param alpha: damping field, ``(nx, ny, nz, 1)``.
    :param Ms_ref: reference saturation magnetisation used to build magnum.np's
        state before the geometry supplies the real (possibly trainable) field.
    """

    def __init__(self, mesh_cfg, solver_cfg: SolverConfig, A: float, alpha: torch.Tensor,
                 Ms_ref: float, Di: float = 0.0):
        self.cfg = solver_cfg
        self.mesh_cfg = mesh_cfg

        self.mesh = Mesh(mesh_cfg.n, mesh_cfg.d, pbc=getattr(mesh_cfg, 'pbc', (0, 0, 0)))
        self.state = State(self.mesh, scale=1e9)
        self.state.material = {"Ms": Ms_ref, "A": A, "Di": Di}
        self.state.m = self.state.Constant([0.0, 0.0, 1.0])

        # gamma before register_alpha: that now precomputes gamma' eagerly
        self.gamma = float(constants.gamma)
        self.register_alpha(alpha)

        self.exchange = ExchangeField()
        self.demag = DemagField() if solver_cfg.demag else None
        # Interfacial DMI is the only non-reciprocal term available here, so it
        # is what a scheme relying on forward/backward asymmetry needs.
        self.dmi = InterfaceDMIField() if Di != 0.0 else None

        self.dt = float(solver_cfg.dt)

        # Thermal (Langevin) field. Off by default -- every deterministic
        # measurement in this project depends on it being off, and a stochastic
        # term would quietly invalidate the AB/BA ratios and the Lyapunov
        # exponents. Enabled only where a device at finite temperature is the
        # thing being modelled, notably any denoising claim: Noise2Noise needs
        # two INDEPENDENT noisy realisations of the same input, and a
        # deterministic rollout gives byte-identical features twice over, so
        # there would be nothing to denoise.
        self.temperature = 0.0
        self._noise_gen = None

    # ------------------------------------------------------------------ setup

    def set_temperature(self, T_kelvin: float, seed: int | None = None):
        """Turn on the fluctuation-dissipation thermal field.

        Fluctuation-dissipation fixes the amplitude given the damping already
        in the model -- it is not a free knob:

            sigma = sqrt(2 alpha k_B T / (gamma mu0 Ms V dt))

        with V the cell volume. Applied as an Euler-Maruyama increment AFTER
        the deterministic RK4 step: RK4 is a smooth-path integrator and feeding
        white noise through its four sub-stages would scale the variance by the
        Butcher weights rather than by dt.
        """
        self.temperature = float(T_kelvin)
        self._noise_gen = None if seed is None else (
            torch.Generator(device="cpu").manual_seed(int(seed)))

    def _thermal_field(self, m: torch.Tensor) -> torch.Tensor:
        k_B, mu0 = 1.380649e-23, 4e-7 * math.pi
        dx, dy, dz = self.mesh_cfg.d
        V = dx * dy * dz
        # Ms is a FIELD, not a scalar: set_Ms patterns the geometry, and cells
        # outside the magnet carry Ms = 0. Since sigma ~ 1/sqrt(Ms), those cells
        # would take an infinite kick -- so the noise is masked to where there
        # is material, which is also physically right (no moments, no
        # fluctuation).
        Ms = self.state.material["Ms"]
        Ms = Ms if torch.is_tensor(Ms) else torch.as_tensor(Ms, dtype=m.dtype)
        Ms = Ms.to(m.dtype).reshape(*Ms.shape[:3], -1)[..., :1] \
            if Ms.dim() >= 3 else Ms.reshape(1, 1, 1, 1)
        live = Ms > 0
        a = self.alpha if torch.is_tensor(self.alpha) else torch.as_tensor(
            self.alpha, dtype=m.dtype)
        a = a.to(m.dtype)
        denom = (self.gamma * mu0 * Ms.clamp_min(1e-30) * V * self.dt)
        sigma = torch.sqrt(2.0 * k_B * self.temperature * a / denom) * live
        eta = torch.randn(m.shape, dtype=m.dtype, generator=self._noise_gen)
        return sigma * eta

    def register_alpha(self, alpha: torch.Tensor):
        """Install the (possibly spatially varying) damping field.

        gamma' is computed EAGERLY here rather than lazily on first use. The
        lazy version allocated it inside the stepper, so under CUDA graph
        capture the graph took ownership of the cached tensor and every replay
        read memory that had since been overwritten -- the failure was reported
        against `self.gamma / (1.0 + alpha**2)` deep in torque(). Precomputing
        makes it a stable input the graph can close over.
        """
        self.alpha = alpha
        self._gamma_prime = (
            (alpha, self.gamma / (1.0 + alpha**2))
            if isinstance(alpha, torch.Tensor) else None)

    def _gp(self, alpha):
        """``gamma / (1 + alpha^2)`` for the spatially varying case."""
        if isinstance(alpha, torch.Tensor):
            cached = self._gamma_prime
            if cached is not None and cached[0] is alpha:
                return cached[1]
            # a caller-supplied alpha (relaxation uses its own); compute it
            # without caching, since caching here is what broke graph replay
            return self.gamma / (1.0 + alpha**2)
        return self.gamma / (1.0 + alpha**2)

    def set_Ms(self, Ms: torch.Tensor):
        """Point magnum.np's material at ``Ms``.

        Assigning a 4-dimensional tensor stores it by reference, so a trainable
        ``Ms`` keeps its autograd history and the demag field stays
        differentiable with respect to the design.
        """
        self.state.material["Ms"] = Ms

    # ------------------------------------------------------------- physics

    def h_eff(self, m: torch.Tensor, h_ext: torch.Tensor) -> torch.Tensor:
        """Effective field for magnetisation ``m`` and external field ``h_ext``."""
        self.state.m = m
        h = h_ext + self.exchange.h(self.state)
        if self.demag is not None:
            h = h + self.demag.h(self.state)
        if self.dmi is not None:
            h = h + self.dmi.h(self.state)
        return h

    def torque(self, m: torch.Tensor, h_ext: torch.Tensor, alpha=None) -> torch.Tensor:
        """Landau-Lifshitz-Gilbert right-hand side, ``dm/dt``.

        ``dm/dt = -gamma' (m x H) - alpha gamma' m x (m x H)``
        with ``gamma' = gamma / (1 + alpha^2)``.
        """
        alpha = self.alpha if alpha is None else alpha
        gp = self._gp(alpha)
        h = self.h_eff(m, h_ext)
        mxh = torch.linalg.cross(m, h)
        return -gp * mxh - alpha * gp * torch.linalg.cross(m, mxh)

    def rk4_step_fields(self, m, h0, h1, h2, h3, alpha=None):
        """RK4 from PRE-EVALUATED substep fields, with no Python in the loop.

        This is the graph-capturable form. ``rk4_step`` takes a Python callable
        and evaluates it per substep, which forces a graph break every time and
        is why the CUDA path spent 64% of wall time on launch overhead: 520
        kernels per step, averaging 4.8 us of work each against a ~4 us launch
        cost. Handing the four fields in as tensors lets the whole step be
        captured once and replayed, measured at 0.63 ms/step against 9.65
        eager -- 15x, and 26x against the CPU path in production use.

        h0..h3 are the total field at theta = 0, 1/2, 1/2, 1 (static + drive).
        """
        dt = self.dt
        k1 = self.torque(m, h0, alpha)
        k2 = self.torque(m + 0.5 * dt * k1, h1, alpha)
        k3 = self.torque(m + 0.5 * dt * k2, h2, alpha)
        k4 = self.torque(m + dt * k3, h3, alpha)
        m_next = m + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if self.cfg.renormalize:
            m_next = m_next / m_next.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return m_next

    def graph_stepper(self, mode: str = "reduce-overhead"):
        """``(step_fn, compiled)`` for a ``(m, h0, h1, h2, h3) -> m`` closure.

        Off CUDA the eager function comes back unchanged, so callers need no
        branch to *run*; the flag is there because callers do need to know
        whether they are driving a replayed graph, which requires
        ``cudagraph_mark_step_begin()`` and a clone of the output each step.

        The flag is returned rather than left to be inferred. The obvious
        inference -- ``stepper is not rollout.rk4_step_fields`` -- is always
        true: every attribute access builds a fresh bound-method object, so the
        eager fallback never compares identical to itself. The caller then takes
        the graph branch on CPU forever. Harmless in that instance (the mark is
        a no-op and the clone is a copy) but only by luck.

        Thermal noise is deliberately excluded: it draws fresh randoms every
        step, which a replayed graph would freeze into a fixed pattern -- noise
        shaped, but not noise.
        """
        if self.temperature > 0.0 or not torch.cuda.is_available():
            return self.rk4_step_fields, False
        try:
            return torch.compile(self.rk4_step_fields, mode=mode,
                                 dynamic=False), True
        except Exception:
            return self.rk4_step_fields, False

    def rk4_step(
        self,
        m: torch.Tensor,
        h_static: torch.Tensor,
        h_drive: Callable[[float], torch.Tensor] | None = None,
        alpha=None,
    ) -> torch.Tensor:
        """One classical RK4 step.

        :param h_drive: callable mapping a sub-step fraction in ``[0, 1]`` to the
            drive field at that instant. ``None`` means undriven.
        """
        dt = self.dt

        def field_at(theta):
            return h_static if h_drive is None else h_static + h_drive(theta)

        k1 = self.torque(m, field_at(0.0), alpha)
        k2 = self.torque(m + 0.5 * dt * k1, field_at(0.5), alpha)
        k3 = self.torque(m + 0.5 * dt * k2, field_at(0.5), alpha)
        k4 = self.torque(m + dt * k3, field_at(1.0), alpha)
        m_next = m + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if self.temperature > 0.0:
            # Euler-Maruyama increment, outside the RK4 stages (see
            # set_temperature for why it cannot go inside them).
            h_th = self._thermal_field(m_next)
            m_next = m_next + self.dt * self.torque(m_next, h_th, alpha)

        if self.cfg.renormalize:
            m_next = m_next / m_next.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return m_next

    # ------------------------------------------------------------ relaxation

    @torch.no_grad()
    def relax(
        self,
        m: torch.Tensor,
        h_static: torch.Tensor,
        steps: int,
        alpha_relax: float,
        warn_tol: float = 0.02,
    ) -> torch.Tensor:
        """Damped integration to the equilibrium magnetisation.

        Run with large uniform damping so the precession dies out in few steps.
        Gradients are deliberately not tracked: the equilibrium is used only as
        the reference the probes subtract, and treating it as a constant is
        what the reference implementation does. The design still receives
        gradients through the driven dynamics, where ``h_static`` enters at
        every step.

        :param warn_tol: warn if the magnetisation is still moving by more than
            this fraction per step when the budget runs out.

        Stopping short here is quietly destructive: the probes report
        ``m - m0``, so a residual relaxation transient rides on every readout
        and is *independent of the drive*. Symptoms are a probe signal that
        barely responds to the excitation amplitude and gradients that point
        nowhere useful. Raise ``relax_steps`` until the warning stops.
        """
        last_step = None
        for _ in range(int(steps)):
            m_next = self.rk4_step(m, h_static, None, alpha=alpha_relax)
            m_next = m_next / m_next.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            last_step = (m_next - m).abs().max()
            m = m_next

        if last_step is not None and float(last_step) > warn_tol:
            import warnings

            warnings.warn(
                f"relaxation did not converge in {int(steps)} steps "
                f"(|dm| = {float(last_step):.3g} on the last step, tolerance {warn_tol:g}). "
                f"Probe readouts will contain a drive-independent transient; "
                f"increase cfg.solver.relax_steps.",
                stacklevel=2,
            )
        return m

    # --------------------------------------------------------------- rollout

    def run(
        self,
        m0: torch.Tensor,
        h_static: torch.Tensor,
        Ms: torch.Tensor,
        signal: torch.Tensor,
        sources: Sequence,
        probes: Sequence,
        record_traces: bool = False,
        snapshot_every: int | None = None,
    ) -> RolloutResult:
        """Drive the film and integrate probe intensities.

        :param m0: equilibrium magnetisation, ``(nx, ny, nz, 3)``.
        :param h_static: static external field, ``(nx, ny, nz, 3)`` in A/m.
        :param Ms: saturation magnetisation field, ``(nx, ny, nz, 1)``.
        :param signal: drive waveform, ``(T, n_sources)``, dimensionless -- the
            per-source amplitude in tesla lives on the source objects.
        :param record_traces: also return per-step probe intensities. Costs
            ``O(T * n_probes)`` memory, negligible next to the field, but it
            disables checkpointing-free operation so keep it off during
            training when you do not need it.
        :param snapshot_every: record ``m`` every ``k`` steps for visualisation.
            Snapshots are detached, and requesting them disables checkpointing
            for this call -- recomputation would either duplicate every
            snapshot or drop the recomputed ones, and this is an analysis path
            that normally runs under ``no_grad`` anyway.
        """
        T = signal.shape[0]
        if signal.shape[1] != len(sources):
            raise ValueError(
                f"signal has {signal.shape[1]} channels but {len(sources)} sources are defined"
            )

        self.set_Ms(Ms)
        nz = self.mesh_cfg.nz

        chunk = self.cfg.chunk_size or max(1, int(round(math.sqrt(T))))
        use_ckpt = self.cfg.checkpoint and torch.is_grad_enabled() and not snapshot_every

        m = m0
        totals = torch.zeros(len(probes), dtype=m0.dtype, device=m0.device)
        traces = [] if record_traces else None
        snapshots, snapshot_steps = ([], []) if snapshot_every else (None, None)

        for start in range(0, T, chunk):
            stop = min(start + chunk, T)
            # A chunk needs one sample past its end for the linear sub-step
            # interpolation of the final step.
            sig = signal[start : min(stop + 1, T)]
            if stop == T:
                sig = torch.cat([sig, torch.zeros_like(sig[-1:])], dim=0)

            args = (m, h_static, Ms, sig, m0)
            if use_ckpt:
                out, m = checkpoint(
                    self._run_chunk, *args, sources, probes, nz, None, 0, None,
                    use_reentrant=False,
                )
            else:
                out, m = self._run_chunk(
                    *args, sources, probes, nz, snapshot_every, start,
                    (snapshots, snapshot_steps) if snapshot_every else None,
                )

            totals = totals + out.sum(dim=0)
            if record_traces:
                traces.append(out)

        return RolloutResult(
            intensities=totals,
            m=m,
            traces=torch.cat(traces, dim=0) if record_traces else None,
            snapshots=torch.stack(snapshots) if snapshot_every and snapshots else None,
            snapshot_steps=snapshot_steps,
        )

    def _run_chunk(self, m, h_static, Ms, sig, m0, sources, probes, nz,
                   snapshot_every=None, global_start=0, sink=None):
        """Integrate ``len(sig) - 1`` steps and return per-step probe intensities.

        Separated out so :func:`torch.utils.checkpoint` can recompute it during
        the backward pass instead of storing every intermediate state.

        ``sink`` collects snapshots at true multiples of ``snapshot_every``
        measured in global steps. Recording them here rather than at chunk
        boundaries matters: the chunk length is ``sqrt(T)``, so boundary-only
        sampling silently quantises the frame interval to that value and the
        recorded times no longer match what the caller asked for.
        """
        self.set_Ms(Ms)
        n_steps = sig.shape[0] - 1
        out = []

        for n in range(n_steps):
            s0, s1 = sig[n], sig[n + 1]

            def h_drive(theta, s0=s0, s1=s1):
                if self.cfg.source_interp == "hold":
                    value = s0
                else:
                    value = s0 + theta * (s1 - s0)
                field = None
                for i, src in enumerate(sources):
                    contrib = src.field(value[i], nz)
                    field = contrib if field is None else field + contrib
                return field

            m = self.rk4_step(m, h_static, h_drive)
            dm_Ms = (m - m0) * Ms
            out.append(torch.stack([p(dm_Ms) for p in probes]))

            if sink is not None:
                step = global_start + n + 1
                if step % snapshot_every == 0:
                    sink[0].append(m.detach().clone())
                    sink[1].append(step)

        return torch.stack(out), m


def estimate_max_timestep(mesh_cfg, material_cfg, fields_cfg, safety: float = 0.2) -> float:
    """Largest RK4 timestep that resolves the fastest precession in the film.

    The stiffest contribution is exchange at the grid scale::

        H_ex,max ~ (2A / (mu_0 Ms)) * (4/dx^2 + 4/dy^2 + 4/dz^2)

    (the Laplacian's largest eigenvalue on a uniform grid). Adding the bias and
    the demagnetising field of a saturated film gives the fastest angular
    frequency ``omega = gamma * H_max``; the step is then ``safety * 2 pi / omega``.

    ``safety = 0.2`` means ~5 samples per period of the fastest mode, which RK4
    handles with a phase error well under a percent. This is a guide, not a
    guarantee -- verify with ``tests/test_solver.py::test_norm_conservation``
    or by checking ``|m|`` drift over a long run.
    """
    MU_0 = 1.2566370614e-6
    gamma = 2.21276157e5

    dx, dy, dz = mesh_cfg.d
    lap = 0.0
    for d, n in zip((dx, dy, dz), mesh_cfg.n):
        if n > 1:
            lap += 4.0 / d**2
    h_ex = 2.0 * material_cfg.A / (MU_0 * material_cfg.Ms) * lap
    h_max = h_ex + fields_cfg.B0 / MU_0 + material_cfg.Ms

    omega = gamma * h_max
    return safety * 2.0 * math.pi / omega
