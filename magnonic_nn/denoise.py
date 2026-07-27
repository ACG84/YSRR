"""A physically-implemented denoising autoencoder for the disk array's readout.

The reservoir's problem, measured: thermal noise buys a smooth firing
threshold (the continuous control parameter an edge-of-chaos system needs) and
costs per-event timing reproducibility, leaving noise/spread ~1.0-1.4 -- the
thermal divergence exceeds the signal's own temporal variation. Individual
spike times are irrecoverable in principle once the threshold is stochastic.
What survives is population and rate structure, and extracting that is a
filtering problem.

So the third stage is another film. **film1 (routing) -> disks (spiking
reservoir) -> film2 (denoising readout)**, all magnonic. The disks emit
through their stray field; each disk's gyration amplitude-modulates film2's
local drive, so film2 sees twelve amplitude-modulated carriers and mixes them
spatially (interference) and temporally (propagation delay). A bottleneck is
simply fewer output probes than input disks.

Why this can work at all: the signal occupies a low-dimensional manifold --
twelve strongly coupled disks, order parameter ~0.93, fed by film channels of
effective rank ~2.7 -- while thermal noise is isotropic and full-rank. That
gap is what any denoiser exploits.

**Trained without clean targets.** No experiment can measure the T = 0 state
of a real device, so the objective is Noise2Noise: run the same input twice
and train film2 to map one noisy realisation onto the *other*. Because the
noise is zero-mean and independent between runs, the regression optimum is
the clean signal -- the noise in the target only adds variance, not bias. Two
repeats, no labels, no ground truth, and a scalar loss that SPSA-style
physical gradients could optimise in hardware.

The autoencoder is trained purely on reconstruction and never on the task.
That is a methodological requirement, not a preference: a denoiser that has
seen the task labels would make the downstream comparison against
ridge-on-raw-features meaningless.

One honest scale caveat. film2 is simulated as a single continuous rollout
across the whole sequence -- the physically correct picture, and the only
affordable one, since a rollout per frame would be thousands of independent
simulations. But the number of film timesteps per input frame is set by
compute, not physics, so the propagation depth per frame is shorter than a
real device's would be. This is a reduced-scale demonstration of the
mechanism, not a device-accurate simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .model import SpinWaveNetwork
from .probes import linear_probe_array
from .sources import PointSource

__all__ = ["DenoiserConfig", "PhysicalDenoiser", "disk_emission"]


@dataclass
class DenoiserConfig:
    nx: int = 20                 # film2 mesh; small, since cost is per timestep
    steps_per_frame: int = 40    # film timesteps per reservoir frame
    n_out: int = 6               # bottleneck: probes out of 12 disks in
    carrier_ghz: float = 3.6     # inside the film's usable band
    probe_radius: float = 1.0


def disk_emission(features: torch.Tensor, cfg: DenoiserConfig, dt: float,
                  n_disks: int = 12) -> torch.Tensor:
    """Disk features -> ``(n_frames * steps_per_frame, n_disks)`` drive signal.

    Each disk amplitude-modulates a shared carrier: magnitude from the orbit
    radius, phase from the envelope phase (the I/Q the lock-in already reads).
    Physically this is the core's stray field modulating the film's local
    drive -- it does not require the film to propagate the 1 GHz gyrotropic
    frequency, which a biased YIG film below its FMR cannot do.
    """
    I, Q = features[:, :n_disks, 0], features[:, :n_disks, 1]
    amp = torch.sqrt(I * I + Q * Q)
    phase = torch.atan2(Q, I)

    n_frames, spf = features.shape[0], cfg.steps_per_frame
    k = torch.arange(n_frames * spf, device=features.device)
    t = k.to(features.dtype) * dt
    frame = torch.div(k, spf, rounding_mode="floor")   # long, for indexing

    w = 2 * math.pi * cfg.carrier_ghz * 1e9
    return amp[frame] * torch.sin(w * t.unsqueeze(-1) + phase[frame])


class PhysicalDenoiser(torch.nn.Module):
    """film2 as a trainable spatial-temporal filter, plus a linear decoder.

    The encoder is the film: twelve modulated sources in, ``n_out`` probes
    out, trained through the micromagnetic rollout by autograd. The decoder is
    linear and exists only to define the reconstruction objective; the readout
    uses the bottleneck ``z``, not the reconstruction.
    """

    def __init__(self, sim_cfg, cfg: DenoiserConfig, n_disks: int = 12,
                 n_features: int = 5):
        super().__init__()
        self.cfg = cfg
        self.n_disks = n_disks

        nx, ny, _ = sim_cfg.mesh.n
        margin = sim_cfg.material.abc_width
        span = ny - 2 * margin
        xs = margin + 2
        sources = [
            PointSource(sim_cfg.mesh, sim_cfg.fields, xs,
                        margin + int((j + 0.5) * span / n_disks))
            for j in range(n_disks)
        ]
        probes = linear_probe_array(sim_cfg.mesh, cfg.n_out, x=nx - margin - 3,
                                    r=cfg.probe_radius, margin=margin)
        self.film = SpinWaveNetwork(sim_cfg, sources, probes)
        self.decoder = torch.nn.Linear(cfg.n_out, n_disks * n_features)
        # Readout gain. Probe intensities come out at ~1e6 while the
        # reconstruction target is unit-variance, so an unnormalised encoder
        # leaves the optimiser twelve orders of magnitude of pure rescaling to
        # traverse before it can learn anything -- which is exactly what the
        # first training run spent all eight epochs doing. Calibrated once
        # from the first batch and then frozen; physically it is the gain of
        # the amplifier any real readout would need anyway.
        self.register_buffer("gain", torch.ones(cfg.n_out))
        self.register_buffer("offset", torch.zeros(cfg.n_out))
        self.register_buffer("calibrated", torch.zeros(1))

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """Reservoir features -> ``(n_frames, n_out)`` bottleneck."""
        dt = self.film.cfg.solver.dt
        signal = disk_emission(features, self.cfg, dt, self.n_disks)
        result = self.film.run(signal, record_traces=True)
        traces = result.traces                       # (T, n_out)
        n_frames = features.shape[0]
        z = traces.reshape(n_frames, self.cfg.steps_per_frame, -1).mean(dim=1)

        if self.calibrated.item() == 0:
            with torch.no_grad():
                self.offset.copy_(z.mean(0))
                self.gain.copy_(z.std(0).clamp_min(1e-30).reciprocal())
                self.calibrated.fill_(1)
        return (z - self.offset) * self.gain

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(features)
        return z, self.decoder(z)
