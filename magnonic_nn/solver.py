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
from magnumnp import DemagField, ExchangeField, Mesh, State, constants  # noqa: E402

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

    def __init__(self, mesh_cfg, solver_cfg: SolverConfig, A: float, alpha: torch.Tensor, Ms_ref: float):
        self.cfg = solver_cfg
        self.mesh_cfg = mesh_cfg

        self.mesh = Mesh(mesh_cfg.n, mesh_cfg.d, pbc=getattr(mesh_cfg, 'pbc', (0, 0, 0)))
        self.state = State(self.mesh, scale=1e9)
        self.state.material = {"Ms": Ms_ref, "A": A}
        self.state.m = self.state.Constant([0.0, 0.0, 1.0])

        self.register_alpha(alpha)

        self.exchange = ExchangeField()
        self.demag = DemagField() if solver_cfg.demag else None

        self.dt = float(solver_cfg.dt)
        self.gamma = float(constants.gamma)

    # ------------------------------------------------------------------ setup

    def register_alpha(self, alpha: torch.Tensor):
        """Install the (possibly spatially varying) damping field."""
        self.alpha = alpha
        self._gamma_prime = None  # invalidate cache

    def _gp(self, alpha):
        """``gamma / (1 + alpha^2)``, cached for the spatially varying case."""
        if isinstance(alpha, torch.Tensor):
            if self._gamma_prime is None or self._gamma_prime[0] is not alpha:
                self._gamma_prime = (alpha, self.gamma / (1.0 + alpha**2))
            return self._gamma_prime[1]
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

        if self.cfg.renormalize:
            m_next = m_next / m_next.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return m_next

    # ------------------------------------------------------------ relaxation

    @torch.no_grad()
    def relax(self, m: torch.Tensor, h_static: torch.Tensor, steps: int, alpha_relax: float) -> torch.Tensor:
        """Damped integration to the equilibrium magnetisation.

        Run with large uniform damping so the precession dies out in few steps.
        Gradients are deliberately not tracked: the equilibrium is used only as
        the reference the probes subtract, and treating it as a constant is
        what the reference implementation does. The design still receives
        gradients through the driven dynamics, where ``h_static`` enters at
        every step.
        """
        for _ in range(int(steps)):
            m = self.rk4_step(m, h_static, None, alpha=alpha_relax)
            m = m / m.norm(dim=-1, keepdim=True).clamp_min(1e-12)
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
            Snapshots are detached; requesting them during a training pass is
            fine but doubles the field memory.
        """
        T = signal.shape[0]
        if signal.shape[1] != len(sources):
            raise ValueError(
                f"signal has {signal.shape[1]} channels but {len(sources)} sources are defined"
            )

        self.set_Ms(Ms)
        nz = self.mesh_cfg.nz

        chunk = self.cfg.chunk_size or max(1, int(round(math.sqrt(T))))
        use_ckpt = self.cfg.checkpoint and torch.is_grad_enabled()

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
                    self._run_chunk, *args, sources, probes, nz, use_reentrant=False
                )
            else:
                out, m = self._run_chunk(*args, sources, probes, nz)

            totals = totals + out.sum(dim=0)
            if record_traces:
                traces.append(out)
            if snapshot_every:
                # record on chunk boundaries that land on the snapshot grid
                if (stop % snapshot_every) < chunk:
                    snapshots.append(m.detach().clone())
                    snapshot_steps.append(stop)

        return RolloutResult(
            intensities=totals,
            m=m,
            traces=torch.cat(traces, dim=0) if record_traces else None,
            snapshots=torch.stack(snapshots) if snapshot_every and snapshots else None,
            snapshot_steps=snapshot_steps,
        )

    def _run_chunk(self, m, h_static, Ms, sig, m0, sources, probes, nz):
        """Integrate ``len(sig) - 1`` steps and return per-step probe intensities.

        Separated out so :func:`torch.utils.checkpoint` can recompute it during
        the backward pass instead of storing every intermediate state.
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
