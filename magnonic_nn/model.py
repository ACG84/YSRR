"""The physical neural network itself.

:class:`SpinWaveNetwork` is a ``torch.nn.Module`` whose *only* trainable
parameter is the scatterer design. There are no matrix multiplications and no
activation functions: a drive waveform goes in, spin waves propagate and
interfere through the trained field pattern, and the time-integrated
intensities at the probes come out. Routing and non-linearity are both done by
the magnetisation dynamics.

The non-linearity is not added anywhere -- it is intrinsic to the LLG equation
and switches on when the precession angle grows past a few degrees. That is why
:attr:`FieldConfig.Bt` is the single most consequential knob in the whole
simulation: at 1 mT the film is a linear interference medium and superposition
holds, so the network can only compute linearly separable functions of the
input spectrum; at tens of mT the same film computes things superposition
forbids.
"""

from __future__ import annotations

import warnings

import torch
from torch import nn

from .config import SimConfig
from .damping import absorbing_damping
from .geometry import Geometry, build_geometry
from .solver import LLGRollout

__all__ = ["SpinWaveNetwork", "normalize_intensities"]


def normalize_intensities(u: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    """Scale probe intensities to sum to one along the last dimension.

    Absolute intensity depends on drive amplitude and on how much energy the
    absorbing boundary swallowed, neither of which carries class information.
    Normalising makes the readout a probability-like vector and stops the
    optimiser from "winning" by simply pumping more power into every probe.
    """
    return u / (u.sum(dim=-1, keepdim=True) + eps)


class SpinWaveNetwork(nn.Module):
    """Spin-wave scatterer with a trainable design region.

    :param cfg: full simulation configuration.
    :param sources: excitation antennas; the drive signal has one channel each.
    :param probes: readout regions; the output vector has one entry each.
    :param geometry: trainable design. Built from ``cfg.geometry`` when omitted.

    Calling the module with a signal of shape ``(T, n_sources)`` returns
    ``(n_probes,)``; a batched signal ``(B, T, n_sources)`` returns
    ``(B, n_probes)``.

    .. note::
       magnum.np's field terms operate on a single ``(nx, ny, nz, 3)`` mesh with
       no batch dimension, so a batch is evaluated as a Python loop over
       samples. Cost is linear in batch size. Use
       :func:`magnonic_nn.train.accumulate_gradients` to keep memory flat across
       a batch.
    """

    def __init__(
        self,
        cfg: SimConfig,
        sources,
        probes,
        geometry: Geometry | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.sources = nn.ModuleList(sources)
        self.probes = nn.ModuleList(probes)
        self.geometry = geometry if geometry is not None else build_geometry(cfg)

        # Sources capture their amplitude at construction, so a cfg.fields.Bt
        # changed afterwards is silently ignored -- and since intensity scales
        # as Bt^2, that shows up as a result which looks plausible and is simply
        # the wrong drive. Catch the mismatch instead of letting it through.
        for i, src in enumerate(self.sources):
            if abs(src.amplitude_T - cfg.fields.Bt) > 1e-12:
                warnings.warn(
                    f"source {i} was built with Bt = {src.amplitude_T * 1e3:g} mT but the "
                    f"config now says {cfg.fields.Bt * 1e3:g} mT. Sources take their "
                    f"amplitude at construction; rebuild them after changing "
                    f"cfg.fields.Bt.",
                    stacklevel=2,
                )

        alpha = absorbing_damping(cfg.mesh, cfg.material)
        self.register_buffer("alpha", alpha)

        self.rollout = LLGRollout(
            cfg.mesh, cfg.solver, cfg.material.A, alpha, cfg.material.Ms,
            Di=getattr(cfg.material, "Di", 0.0),
        )

        # equilibrium magnetisation, recomputed whenever the design changes
        self._m0 = None
        self._m0_design_version = None

    # ------------------------------------------------------------------ state

    @property
    def n_probes(self) -> int:
        return len(self.probes)

    @property
    def n_sources(self) -> int:
        return len(self.sources)

    def initial_magnetisation(self) -> torch.Tensor:
        """Uniform state along the bias axis, the starting point for relaxation."""
        nx, ny, nz = self.cfg.mesh.n
        axis = self.geometry.bias_axis.reshape(1, 1, 1, 3)
        return axis.expand(nx, ny, nz, 3).clone()

    def equilibrium(self, force: bool = False) -> torch.Tensor:
        """Relaxed magnetisation for the current design.

        Cached and invalidated when any design parameter changes, so repeated
        forward passes within one optimiser step do not re-relax. The cache key
        is the parameter version counter, which PyTorch bumps on every in-place
        update -- including the one the optimiser performs.
        """
        version = tuple(p._version for p in self.geometry.parameters())
        if force or self._m0 is None or version != self._m0_design_version:
            with torch.no_grad():
                h_static = self.geometry.static_field()
                self.rollout.set_Ms(self.geometry.Ms_field())
                self._m0 = self.rollout.relax(
                    self.initial_magnetisation(),
                    h_static,
                    self.cfg.solver.relax_steps,
                    self.cfg.material.alpha_max,
                )
            self._m0_design_version = version
        return self._m0

    # ---------------------------------------------------------------- forward

    def forward(self, signal: torch.Tensor, **kwargs) -> torch.Tensor:
        """Propagate ``signal`` and return time-integrated probe intensities."""
        batched = signal.dim() == 3
        signals = signal if batched else signal.unsqueeze(0)

        outputs = [self.run(s, **kwargs).intensities for s in signals]
        stacked = torch.stack(outputs)
        return stacked if batched else stacked[0]

    def run(self, signal: torch.Tensor, record_traces: bool = False, snapshot_every: int | None = None):
        """Single-sample rollout returning the full :class:`~magnonic_nn.solver.RolloutResult`."""
        if signal.dim() != 2:
            raise ValueError(f"expected a (T, n_sources) signal, got shape {tuple(signal.shape)}")

        h_static = self.geometry.static_field()
        Ms = self.geometry.Ms_field()
        m0 = self.equilibrium()

        return self.rollout.run(
            m0=m0,
            h_static=h_static,
            Ms=Ms,
            signal=signal,
            sources=self.sources,
            probes=self.probes,
            record_traces=record_traces,
            snapshot_every=snapshot_every,
        )

    # ------------------------------------------------------------ diagnostics

    def design_field_tesla(self) -> torch.Tensor:
        """Total static field along the bias axis, in tesla, as a ``(nx, ny)`` image.

        Worth looking at before trusting any trained result: an unconstrained
        free-form design can converge on local fields far larger than anything
        a real device could apply.
        """
        from .config import MU_0

        h = self.geometry.static_field()[:, :, 0, :].detach()
        return MU_0 * (h * self.geometry.bias_axis).sum(dim=-1)

    def probe_positions(self):
        return [(p.x, p.y) for p in self.probes if hasattr(p, "x")]

    def extra_repr(self) -> str:
        m = self.cfg.mesh
        return (
            f"mesh={m.nx}x{m.ny}x{m.nz} @ {m.dx * 1e9:g}nm, "
            f"{self.cfg.solver.timesteps} steps x {self.cfg.solver.dt * 1e12:g}ps "
            f"({self.cfg.duration * 1e9:g}ns), "
            f"Bt={self.cfg.fields.Bt * 1e3:g}mT, "
            f"{self.n_sources} source(s), {self.n_probes} probe(s)"
        )
