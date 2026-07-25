"""Configuration dataclasses and ready-made presets.

Default values reproduce the reference implementation released with

    Papp, A., Porod, W. & Csaba, G. "Nanoscale neural network using non-linear
    spin-wave interference." Nat Commun 12, 6422 (2021).
    https://doi.org/10.1038/s41467-021-26711-z

(the ``SpinTorch`` code, https://github.com/a-papp/SpinTorch). Fields are given
in tesla here because that is how the paper quotes them; the solver converts to
the A/m convention magnum.np uses via ``H = B / mu_0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

MU_0 = 1.2566370614e-6  # matches magnumnp.constants.mu_0

__all__ = [
    "MaterialConfig",
    "MeshConfig",
    "FieldConfig",
    "SolverConfig",
    "SimConfig",
    "PRESETS",
    "get_preset",
    "MU_0",
]


@dataclass
class MaterialConfig:
    """Magnetic material parameters.

    Defaults are YIG as used in the reference implementation.
    """

    Ms: float = 140e3
    """Saturation magnetisation (A/m)."""

    A: float = 3.65e-12
    """Exchange stiffness (J/m)."""

    Di: float = 0.0
    """Interfacial DMI strength (J/m^2). Zero by default -- plain YIG has none.

    This is the term that makes the medium **non-reciprocal**: it adds a
    contribution linear in k to the dispersion, so ``omega(+k) != omega(-k)``
    and a wave travels at a different speed forwards than backwards. Everything
    else in this model is reciprocal, which is why a reciprocity measurement on
    the default configuration returns 1.0000 exactly.

    Scale, for the default 20 nm YIG at 4 GHz: 0.01 mJ/m^2 splits the two
    directions by 55 MHz and their group velocities by 455 vs 405 m/s;
    0.05 mJ/m^2 (the upper end of what is reported for YIG/Pt) gives 273 MHz
    and 556 vs 304 m/s, an asymmetry approaching 2x; past ~0.2 mJ/m^2 the
    backward branch stops propagating altogether and the film is a one-way
    medium."""

    alpha: float = 1e-4
    """Bulk Gilbert damping in the propagation region."""

    alpha_max: float = 0.5
    """Damping at the outermost cell of the absorbing boundary, and the value
    used during relaxation."""

    abc_width: int = 10
    """Width of the absorbing boundary layer, in cells."""

    abc_exponent: float = 2.0
    """Exponent of the damping taper. ``2`` reproduces the reference
    implementation's quadratic ramp."""


@dataclass
class MeshConfig:
    """Finite-difference mesh. The film lies in the x-y plane."""

    nx: int = 100
    ny: int = 100
    nz: int = 1
    dx: float = 50e-9
    dy: float = 50e-9
    dz: float = 20e-9
    """``dz`` is the film thickness when ``nz == 1``."""

    pbc: tuple = (0, 0, 0)
    """Periodic images per axis, passed straight to :class:`magnumnp.Mesh`.
    ``(0, 1, 0)`` makes the film infinite along y, which is how you measure a
    clean dispersion relation without edge modes. A periodic axis gets no
    absorbing boundary."""

    @property
    def n(self):
        return (self.nx, self.ny, self.nz)

    @property
    def d(self):
        return (self.dx, self.dy, self.dz)

    @property
    def extent(self):
        """Physical size (m) of the simulated region."""
        return (self.nx * self.dx, self.ny * self.dy, self.nz * self.dz)


@dataclass
class FieldConfig:
    """Static and driven magnetic fields, in tesla."""

    B0: float = 60e-3
    """Uniform bias field magnitude."""

    bias_axis: Sequence[float] = (0.0, 1.0, 0.0)
    """Direction of the bias field. In-plane along +y by default, which puts
    propagation along x in the Damon-Eshbach (surface-wave) geometry."""

    B1: float = 50e-3
    """Scale of the trainable scattering field: the design variable
    ``rho in [-1, 1]`` maps to a local field ``B1 * rho`` added along
    ``bias_axis``."""

    Bt: float = 1e-3
    """Excitation amplitude at the source. ``1 mT`` is the paper's linear
    regime; ``20-50 mT`` drives the non-linear regime where the network gains
    its extra computational power."""

    drive_axis: int = 2
    """Component the sources drive (2 = out-of-plane z). Perpendicular to the
    bias so it torques the equilibrium magnetisation efficiently."""


@dataclass
class SolverConfig:
    """Time integration and gradient bookkeeping."""

    dt: float = 20e-12
    """Fixed RK4 timestep (s)."""

    timesteps: int = 600
    """Number of steps in the driven rollout."""

    relax_steps: int = 100
    """Damped steps used to find the equilibrium ``m0`` before driving."""

    checkpoint: bool = True
    """Recompute activations during backward instead of storing every step.
    Turns O(T) memory into O(sqrt(T))."""

    chunk_size: int | None = None
    """Steps per checkpoint segment. ``None`` picks ``round(sqrt(timesteps))``,
    which minimises peak memory for a fixed recompute budget."""

    renormalize: bool = False
    """Project ``|m|`` back to 1 after every step. The reference implementation
    does not, relying on RK4 to conserve the norm; enable for long rollouts or
    large timesteps."""

    demag: bool = True
    """Include the demagnetisation field. Disabling it is ~3x faster and useful
    for smoke tests, but removes the dipolar contribution to the dispersion."""

    source_interp: str = "linear"
    """How the sampled drive signal is evaluated at RK4 sub-stages.
    ``'linear'`` interpolates between samples (more accurate); ``'hold'``
    reproduces the reference implementation's zero-order hold."""


@dataclass
class SimConfig:
    """Everything needed to build a :class:`~magnonic_nn.model.SpinWaveNetwork`."""

    mesh: MeshConfig = field(default_factory=MeshConfig)
    material: MaterialConfig = field(default_factory=MaterialConfig)
    fields: FieldConfig = field(default_factory=FieldConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)

    geometry: str = "freeform"
    """One of ``'freeform'`` (trainable field pattern), ``'ms'`` (trainable
    saturation-magnetisation pattern) or ``'nanomagnets'`` (binary PMA
    nanomagnet array)."""

    precision: str = "float32"
    device: str = "cpu"
    seed: int = 0

    @property
    def duration(self) -> float:
        """Physical duration of the driven rollout (s)."""
        return self.solver.timesteps * self.solver.dt

    @property
    def sample_rate(self) -> float:
        """Sampling rate of the drive signal (Hz)."""
        return 1.0 / self.solver.dt

    @property
    def nyquist(self) -> float:
        return 0.5 / self.solver.dt

    def replace(self, **kwargs) -> "SimConfig":
        """Return a copy with top-level fields overridden."""
        return replace(self, **kwargs)


def _tiny() -> SimConfig:
    """Small enough to run a full train step in seconds on a laptop CPU.

    Used by the test suite and by ``--preset tiny`` for smoke runs. The mesh is
    2.4 um square, which fits roughly three wavelengths at 3 GHz -- enough for
    the scatterer to do something measurable, not enough for a real task.
    """
    return SimConfig(
        mesh=MeshConfig(nx=48, ny=48, nz=1, dx=50e-9, dy=50e-9, dz=20e-9),
        material=MaterialConfig(abc_width=6),
        solver=SolverConfig(dt=20e-12, timesteps=100, relax_steps=30),
    )


def _focus() -> SimConfig:
    """Exact geometry of the reference implementation's ``focus.py`` example.

    5 um x 5 um YIG, single 4 GHz line source, 19 probes along the far edge.
    """
    return SimConfig(
        mesh=MeshConfig(nx=100, ny=100, nz=1, dx=50e-9, dy=50e-9, dz=20e-9),
        material=MaterialConfig(),
        fields=FieldConfig(B0=60e-3, B1=50e-3, Bt=1e-3),
        solver=SolverConfig(dt=20e-12, timesteps=600, relax_steps=100),
    )


def _demux() -> SimConfig:
    """Frequency demultiplexing: 3.0 / 3.5 / 4.0 GHz to three separate probes.

    Longer rollout than ``focus`` because resolving 500 MHz spacing needs at
    least ~1/(500 MHz) = 2 ns of steady state on top of the transit time.
    """
    return SimConfig(
        mesh=MeshConfig(nx=100, ny=100, nz=1, dx=50e-9, dy=50e-9, dz=20e-9),
        material=MaterialConfig(),
        fields=FieldConfig(B0=60e-3, B1=50e-3, Bt=1e-3),
        solver=SolverConfig(dt=20e-12, timesteps=800, relax_steps=100),
    )


def _vowels() -> SimConfig:
    """Vowel classification in the non-linear regime.

    ``Bt = 20 mT`` puts the peak precession angle around 21 degrees, which
    breaks superposition decisively (133% deviation between the response to a
    sum of tones and the sum of the responses, against 0.6% at 1 mT) without
    driving the film into switching. At 50 mT the peak ``|m - m0|`` reaches
    1.99 out of a possible 2 -- parts of the film reverse, and the result is
    then about domain formation rather than non-linear wave interference.

    ``renormalize`` is on because large-angle precession makes RK4 drift the
    magnetisation norm: 2e-3 per rollout at 20 mT and 2e-2 at 50 mT, against
    1e-7 with the projection. It costs nothing measurable.
    """
    return SimConfig(
        mesh=MeshConfig(nx=100, ny=100, nz=1, dx=50e-9, dy=50e-9, dz=20e-9),
        material=MaterialConfig(),
        fields=FieldConfig(B0=60e-3, B1=50e-3, Bt=20e-3),
        solver=SolverConfig(dt=20e-12, timesteps=800, relax_steps=100,
                            renormalize=True),
    )


def _paper() -> SimConfig:
    """10 um x 10 um scatterer, the size quoted in the paper.

    200x200 cells at 50 nm. GPU strongly recommended: a forward+backward pass
    is ~10x the cost of the ``focus`` preset.
    """
    return SimConfig(
        mesh=MeshConfig(nx=200, ny=200, nz=1, dx=50e-9, dy=50e-9, dz=20e-9),
        material=MaterialConfig(abc_width=16),
        fields=FieldConfig(B0=60e-3, B1=50e-3, Bt=50e-3),
        solver=SolverConfig(dt=20e-12, timesteps=1200, relax_steps=150),
        device="cuda",
    )


PRESETS = {
    "tiny": _tiny,
    "focus": _focus,
    "demux": _demux,
    "vowels": _vowels,
    "paper": _paper,
}


def get_preset(name: str) -> SimConfig:
    """Return a fresh :class:`SimConfig` for the named preset."""
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
    return PRESETS[name]()
