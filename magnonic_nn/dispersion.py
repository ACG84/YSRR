"""Spin-wave dispersion for the tangentially magnetised film.

Which frequencies a task may use, and whether the mesh resolves them, both
follow from the dispersion relation, so it is worth getting right rather than
guessing.

For a film of thickness ``d`` magnetised in plane, with the wavevector
perpendicular to the magnetisation (the Damon-Eshbach / surface-wave geometry
that the default source-probe layout sets up), the dipole-exchange dispersion is

.. math::
    \\omega^2(k) = \\omega_0(k)\\,[\\omega_0(k) + \\omega_M]
                 + \\frac{\\omega_M^2}{4}\\,\\bigl(1 - e^{-2kd}\\bigr)

with

.. math::
    \\omega_0(k) = \\gamma\\left(H_0 + \\frac{2A}{\\mu_0 M_s}k^2\\right),
    \\qquad \\omega_M = \\gamma M_s .

Both limits are the textbook ones: at ``k -> 0`` it reduces to the Kittel
in-plane resonance ``sqrt(omega_H (omega_H + omega_M))``, and for ``kd >> 1``
without exchange it saturates at the Damon-Eshbach surface-mode frequency
``omega_H + omega_M / 2``.

Purely exchange-based estimates are badly wrong in this regime -- at 4 GHz in
20 nm YIG they underestimate the wavelength by roughly a factor of three,
because they charge the entire frequency excess above resonance to exchange
when most of it is dipolar.

:func:`measure_dispersion` checks the analytic curve against the solver itself.
"""

from __future__ import annotations

import math

import torch

from .config import MU_0, FieldConfig, MaterialConfig, MeshConfig, SimConfig

__all__ = [
    "GAMMA",
    "omega_of_k",
    "frequency_of_k",
    "k_of_frequency",
    "wavelength",
    "kittel_fmr",
    "usable_band",
    "measure_dispersion",
]

GAMMA = 2.21276157e5
"""Gyromagnetic ratio in m/(A s), matching ``magnumnp.constants.gamma``."""


def _omegas(fields: FieldConfig, material: MaterialConfig):
    h0 = fields.B0 / MU_0
    return GAMMA * h0, GAMMA * material.Ms


def omega_of_k(k, fields: FieldConfig, material: MaterialConfig, thickness: float) -> float:
    """Angular frequency (rad/s) of the surface mode at wavenumber ``k`` (rad/m)."""
    omega_h, omega_m = _omegas(fields, material)
    omega_0 = omega_h + GAMMA * (2.0 * material.A / (MU_0 * material.Ms)) * k**2
    dipolar = 0.25 * omega_m**2 * (1.0 - math.exp(-2.0 * k * thickness))
    return math.sqrt(omega_0 * (omega_0 + omega_m) + dipolar)


def frequency_of_k(k, fields: FieldConfig, material: MaterialConfig, thickness: float) -> float:
    """Frequency in Hz at wavenumber ``k``."""
    return omega_of_k(k, fields, material, thickness) / (2 * math.pi)


def kittel_fmr(fields: FieldConfig, material: MaterialConfig) -> float:
    """Uniform-mode (``k = 0``) ferromagnetic resonance, in Hz.

    Below this the film does not propagate: a drive there produces an
    evanescent, exponentially decaying response instead of a travelling wave.
    """
    omega_h, omega_m = _omegas(fields, material)
    return math.sqrt(omega_h * (omega_h + omega_m)) / (2 * math.pi)


def k_of_frequency(
    freq: float,
    fields: FieldConfig,
    material: MaterialConfig,
    thickness: float,
    k_max: float = 1e10,
) -> float:
    """Invert the dispersion for ``k`` at frequency ``freq``.

    ``omega(k)`` is strictly increasing, so a bisection is both safe and exact
    to machine precision in a few dozen iterations.

    :returns: wavenumber in rad/m, or ``0.0`` if ``freq`` is at or below the FMR
        (no propagating solution exists).
    """
    if freq <= kittel_fmr(fields, material):
        return 0.0
    if frequency_of_k(k_max, fields, material, thickness) < freq:
        raise ValueError(f"{freq / 1e9:.3f} GHz lies above k_max = {k_max:g} rad/m")

    lo, hi = 0.0, k_max
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if frequency_of_k(mid, fields, material, thickness) < freq:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def wavelength(freq: float, fields: FieldConfig, material: MaterialConfig, thickness: float) -> float:
    """Wavelength in metres at ``freq``; ``inf`` below the FMR."""
    k = k_of_frequency(freq, fields, material, thickness)
    return float("inf") if k <= 0 else 2 * math.pi / k


def usable_band(cfg: SimConfig, cells_per_wavelength: float = 6.0):
    """Frequency range the configured film both propagates and resolves.

    :returns: ``(f_lo, f_hi)`` in Hz.

    The lower edge is 5 % above the FMR, keeping drives clear of the resonance
    where the group velocity vanishes and nothing propagates away from the
    antenna. The upper edge is where the wavelength falls to
    ``cells_per_wavelength * dx``; past that the mesh aliases the wave and the
    simulation stops being trustworthy regardless of what the film would do.
    """
    thickness = cfg.mesh.dz * cfg.mesh.nz
    f_lo = 1.05 * kittel_fmr(cfg.fields, cfg.material)

    min_wavelength = cells_per_wavelength * min(cfg.mesh.dx, cfg.mesh.dy)
    k_max = 2 * math.pi / min_wavelength
    f_hi = frequency_of_k(k_max, cfg.fields, cfg.material, thickness)

    return f_lo, max(f_hi, f_lo)


@torch.no_grad()
def measure_dispersion(
    cfg: SimConfig,
    freqs=None,
    nx: int = 400,
    ny: int = 8,
    settle_frac: float = 0.5,
):
    """Measure ``k(f)`` from the solver and compare against :func:`omega_of_k`.

    Drives a long, narrow strip of film with a tone at one end, lets the wave
    fill the strip, then reads the dominant spatial frequency of the standing
    profile. This validates the whole chain -- mesh, field terms, timestep,
    absorbing boundary -- against an independent analytic result, and it is the
    thing to run first when a task will not train.

    The strip is made periodic along y, so it behaves as a section of an
    infinite film rather than a waveguide with quantised width modes -- the
    analytic formula above describes the infinite film, and comparing against
    anything else would be comparing two different problems.

    :param nx: strip length in cells. Needs to hold several wavelengths.
    :param ny: strip width. Small, since the mode is uniform across it.
    :param settle_frac: fraction of the strip nearest the source that is
        discarded before the spatial FFT, to skip the antenna near field.
    :returns: list of dicts with ``freq``, ``k_measured``, ``k_analytic``,
        ``lambda_measured``, ``lambda_analytic``, ``rel_error``.
    """
    from dataclasses import replace as _replace

    from .config import MeshConfig
    from .model import SpinWaveNetwork
    from .probes import PointProbe
    from .signals import tone
    from .sources import LineSource

    freqs = list(freqs) if freqs is not None else [3.6e9, 4.0e9, 4.5e9, 5.0e9]
    thickness = cfg.mesh.dz * cfg.mesh.nz

    strip = MeshConfig(
        nx=nx, ny=ny, nz=1, dx=cfg.mesh.dx, dy=cfg.mesh.dy, dz=cfg.mesh.dz, pbc=(0, 1, 0)
    )
    material = _replace(cfg.material, abc_width=max(min(cfg.material.abc_width, nx // 8), 2))
    strip_cfg = cfg.replace(mesh=strip, material=material)

    source_x = material.abc_width + 2
    results = []
    for freq in freqs:
        src = LineSource(strip, strip_cfg.fields, source_x, 0, source_x, ny - 1)
        model = SpinWaveNetwork(strip_cfg, [src], [PointProbe(strip, nx // 2, ny // 2)])

        result = model.run(tone(strip_cfg, freq))
        m0 = model.equilibrium()

        # transverse wave amplitude along the strip, averaged across its width
        line = ((result.m - m0)[:, :, 0, 2]).mean(dim=1).detach()

        # The window has to sit between the antenna near field and the
        # wavefront. That matters more than it sounds: the group velocity of
        # these dipolar modes is only a few hundred m/s, so in a 10 ns rollout
        # the wave covers a few micrometres and no more. A fixed window placed
        # past the front measures nothing but numerical noise.
        envelope = line.abs()
        kernel = torch.ones(1, 1, 5, dtype=envelope.dtype, device=envelope.device) / 5.0
        smooth = torch.nn.functional.conv1d(
            envelope.reshape(1, 1, -1), kernel, padding=2
        ).reshape(-1)

        start = source_x + int(max(4, settle_frac * 8))
        front = (smooth > 0.15 * smooth.max()).nonzero()
        stop = nx - material.abc_width - 2
        if len(front):
            stop = min(stop, int(front.max()) - 2)

        if stop - start < 24:
            raise RuntimeError(
                f"at {freq / 1e9:.2f} GHz the wave only reached cell {stop} of {nx}; "
                f"increase cfg.solver.timesteps so it propagates further, or "
                f"shorten the strip"
            )

        segment = line[start:stop]
        segment = segment - segment.mean()
        segment = segment * torch.hann_window(
            len(segment), dtype=segment.dtype, device=segment.device
        )

        # Zero-pad so the k grid is fine enough to locate the peak precisely,
        # then refine with a parabolic fit over the three bins around it.
        n_fft = 8 * len(segment)
        spectrum = torch.fft.rfft(segment, n=n_fft).abs()
        k_axis = 2 * math.pi * torch.fft.rfftfreq(n_fft, d=strip.dx)
        peak = int(spectrum[1:].argmax()) + 1  # skip DC

        k_measured = float(k_axis[peak])
        if 0 < peak < len(spectrum) - 1:
            a, b, c = (float(spectrum[peak + o]) for o in (-1, 0, 1))
            denom = a - 2 * b + c
            if denom != 0:
                shift = 0.5 * (a - c) / denom
                dk = float(k_axis[1] - k_axis[0])
                k_measured += shift * dk

        k_analytic = k_of_frequency(freq, cfg.fields, cfg.material, thickness)
        results.append(
            {
                "freq": freq,
                "k_measured": k_measured,
                "k_analytic": k_analytic,
                "lambda_measured": (2 * math.pi / k_measured) if k_measured > 0 else float("inf"),
                "lambda_analytic": (2 * math.pi / k_analytic) if k_analytic > 0 else float("inf"),
                "rel_error": abs(k_measured - k_analytic) / k_analytic if k_analytic > 0 else float("nan"),
            }
        )
    return results
