"""Drive waveforms, and the vowel dataset.

Two groups of helpers live here:

* generic excitations -- tones, multi-tones, chirps -- with sensible envelopes;
* a vowel dataset, because vowel classification is the paper's headline task.

**On frequency scaling.** The paper scales speech waveforms "up to microwave
frequencies". A single multiplicative factor cannot work for a magnonic film:
speech formants span more than a decade (roughly 300 Hz to 3 kHz), while a
biased film only propagates above its ferromagnetic resonance and, at a fixed
mesh spacing, only up to the frequency whose wavelength the mesh still
resolves. Scaling F1 into the passband throws F3 into the unresolvable exchange
regime, and vice versa. :func:`vowel_dataset` therefore defaults to an *affine*
map that places the formant range inside the device's usable band while keeping
the relative spacing that distinguishes the vowels. Pass ``mapping="linear"``
for the paper-literal single-factor version, and check
:func:`usable_band` before you do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .config import SimConfig
from .dispersion import kittel_fmr, usable_band

__all__ = [
    "time_vector",
    "raised_cosine_envelope",
    "tone",
    "multitone",
    "chirp",
    "gated_tones",
    "FORMANTS",
    "synth_vowel",
    "vowel_dataset",
]

def time_vector(cfg: SimConfig) -> torch.Tensor:
    """Sample times of the drive signal, shape ``(T,)``."""
    T = cfg.solver.timesteps
    return torch.arange(T, dtype=torch.get_default_dtype()) * cfg.solver.dt


def raised_cosine_envelope(T: int, ramp_frac: float = 0.1) -> torch.Tensor:
    """Smooth turn-on/turn-off envelope, shape ``(T,)``.

    Switching a tone on abruptly injects a broadband transient that excites
    every mode the film supports, including ones far from the intended
    frequency. A raised-cosine ramp over ``ramp_frac`` of the record confines
    the excitation to a narrow band around the carrier.

    ``ramp_frac = 0`` returns a rectangular window.
    """
    env = torch.ones(T, dtype=torch.get_default_dtype())
    n_ramp = int(T * ramp_frac)
    if n_ramp < 1:
        return env
    ramp = 0.5 * (1.0 - torch.cos(torch.linspace(0.0, math.pi, n_ramp)))
    env[:n_ramp] = ramp
    env[T - n_ramp :] = ramp.flip(0)
    return env


def tone(cfg: SimConfig, freq: float, amplitude: float = 1.0, phase: float = 0.0, ramp_frac: float = 0.1) -> torch.Tensor:
    """Single-frequency drive, shape ``(T, 1)``."""
    t = time_vector(cfg)
    wave = amplitude * torch.sin(2 * math.pi * freq * t + phase)
    return (wave * raised_cosine_envelope(len(t), ramp_frac)).reshape(-1, 1)


def multitone(cfg: SimConfig, freqs, amplitudes=None, phases=None, ramp_frac: float = 0.1) -> torch.Tensor:
    """Sum of tones on a single channel, shape ``(T, 1)``."""
    t = time_vector(cfg)
    freqs = list(freqs)
    amplitudes = [1.0] * len(freqs) if amplitudes is None else list(amplitudes)
    phases = [0.0] * len(freqs) if phases is None else list(phases)

    wave = torch.zeros_like(t)
    for f, a, p in zip(freqs, amplitudes, phases):
        wave = wave + a * torch.sin(2 * math.pi * f * t + p)
    return (wave * raised_cosine_envelope(len(t), ramp_frac)).reshape(-1, 1)


def chirp(cfg: SimConfig, f0: float, f1: float, amplitude: float = 1.0, ramp_frac: float = 0.05) -> torch.Tensor:
    """Linear frequency sweep, shape ``(T, 1)``.

    Useful for measuring what the film actually transmits: drive with a chirp
    covering the band of interest and look at the probe spectrum.
    """
    t = time_vector(cfg)
    duration = float(t[-1]) if len(t) > 1 else 1.0
    rate = (f1 - f0) / max(duration, 1e-30)
    phase = 2 * math.pi * (f0 * t + 0.5 * rate * t**2)
    wave = amplitude * torch.sin(phase)
    return (wave * raised_cosine_envelope(len(t), ramp_frac)).reshape(-1, 1)


def gated_tones(cfg: SimConfig, freqs, gate_frac: float = 1.0, ramp_frac: float = 0.1) -> torch.Tensor:
    """One tone per source channel, shape ``(T, len(freqs))``.

    ``gate_frac < 1`` shortens each burst to that fraction of the record, which
    lets you separate the transit-time response from the steady state.
    """
    t = time_vector(cfg)
    T = len(t)
    env = raised_cosine_envelope(int(T * gate_frac), ramp_frac)
    full = torch.zeros(T, dtype=t.dtype)
    full[: len(env)] = env

    cols = [torch.sin(2 * math.pi * f * t) * full for f in freqs]
    return torch.stack(cols, dim=-1)


# ---------------------------------------------------------------------- vowels

# Formant frequencies (Hz) and bandwidths for the three vowels used in the
# paper, from the Hillenbrand et al. (1995) adult-male averages.
# Bandwidths are not tabulated by Hillenbrand; they follow the usual rule of
# thumb that a formant's bandwidth scales with its centre frequency, taken here
# as ~7% of F1 and ~9% of F2/F3. They set how sharply each resonance is defined
# and only weakly affect class separability.
FORMANTS = {
    # the original three, kept as the default task
    "iy": ((342.0, 2322.0, 3000.0), (90.0, 170.0, 240.0)),   # "heed"
    "ei": ((476.0, 2089.0, 2691.0), (110.0, 180.0, 250.0)),  # "hayed"  (= /ey/)
    "ae": ((588.0, 1952.0, 2601.0), (130.0, 190.0, 260.0)),  # "had"
    # the rest of the Hillenbrand adult-male set
    "ih": ((427.0, 2034.0, 2684.0), (100.0, 180.0, 250.0)),  # "hid"
    "eh": ((580.0, 1799.0, 2605.0), (130.0, 170.0, 250.0)),  # "head"
    "ah": ((623.0, 1200.0, 2550.0), (140.0, 130.0, 240.0)),  # "hud"
    "aa": ((768.0, 1333.0, 2522.0), (160.0, 140.0, 240.0)),  # "hod"
    "ao": ((652.0, 997.0, 2538.0), (140.0, 110.0, 240.0)),   # "hawed"
    "uh": ((469.0, 1122.0, 2434.0), (110.0, 120.0, 230.0)),  # "hood"
    "ow": ((497.0, 910.0, 2459.0), (110.0, 100.0, 230.0)),   # "hoed"
    "uw": ((378.0, 997.0, 2343.0), (90.0, 110.0, 220.0)),    # "who'd"
    "er": ((474.0, 1379.0, 1710.0), (110.0, 140.0, 170.0)),  # "heard"
}
"""Formant centres (Hz) and bandwidths for the Hillenbrand et al. (1995)
adult-male averages. The default task uses the first three, which are well
separated; adding more packs the formant space and makes the task harder in a
way that has nothing to do with the physics -- check
:func:`class_separations` before blaming a poor result on the film."""

_AUDIO_F_MIN = 300.0
_AUDIO_F_MAX = 3100.0


@dataclass
class VowelDataset:
    """Container for a synthesised vowel set.

    :ivar signals: ``(N, T, 1)`` drive waveforms, peak-normalised to 1.
    :ivar labels: ``(N,)`` integer class indices.
    :ivar classes: class names in label order.
    :ivar carrier_frequencies: ``(N, 3)`` microwave formant frequencies actually used.
    """

    signals: torch.Tensor
    labels: torch.Tensor
    classes: list
    carrier_frequencies: torch.Tensor

    def __len__(self):
        return self.signals.shape[0]

    def split(self, n_train_per_class: int, generator: torch.Generator | None = None):
        """Stratified train/test split. Returns ``(train, test)`` datasets."""
        train_idx, test_idx = [], []
        for c in range(len(self.classes)):
            idx = (self.labels == c).nonzero(as_tuple=True)[0]
            perm = torch.randperm(len(idx), generator=generator, device=idx.device)
            idx = idx[perm]
            train_idx.append(idx[:n_train_per_class])
            test_idx.append(idx[n_train_per_class:])
        train_idx = torch.cat(train_idx)
        test_idx = torch.cat(test_idx)
        return self._subset(train_idx), self._subset(test_idx)

    def _subset(self, idx):
        return VowelDataset(
            signals=self.signals[idx],
            labels=self.labels[idx],
            classes=self.classes,
            carrier_frequencies=self.carrier_frequencies[idx],
        )


def _map_frequency(f_audio, mapping: str, band, scale: float):
    if mapping == "linear":
        return f_audio * scale
    if mapping == "affine":
        f_lo, f_hi = band
        frac = (f_audio - _AUDIO_F_MIN) / (_AUDIO_F_MAX - _AUDIO_F_MIN)
        return f_lo + frac * (f_hi - f_lo)
    raise ValueError(f"unknown mapping {mapping!r}; use 'affine' or 'linear'")


def synth_vowel(
    cfg: SimConfig,
    vowel: str,
    band,
    f0_audio: float = 120.0,
    jitter: float = 0.0,
    mapping: str = "affine",
    scale: float = 2e6,
    generator: torch.Generator | None = None,
    ramp_frac: float = 0.08,
):
    """Synthesise one vowel token as a microwave-band waveform.

    Source-filter model: a harmonic comb at the (scaled) pitch ``f0_audio``,
    shaped by three Lorentzian formant resonances. That reproduces the property
    the classifier has to exploit -- vowel identity lives in the *envelope* of
    the spectrum, not in any single frequency -- while staying fully
    reproducible and dependency-free.

    :param jitter: fractional random perturbation of pitch and formant
        frequencies, standing in for speaker-to-speaker variation.
    :returns: ``(waveform (T,), formants_used (3,))``.
    """
    formants, bandwidths = FORMANTS[vowel]

    def rnd():
        return float(torch.randn(1, generator=generator, device="cpu").item())

    f0 = f0_audio * (1.0 + jitter * rnd())
    used = [f * (1.0 + jitter * rnd()) for f in formants]

    f_mw = [_map_frequency(f, mapping, band, scale) for f in used]
    bw_mw = [
        max(_map_frequency(f + b, mapping, band, scale) - _map_frequency(f, mapping, band, scale), 1e6)
        for f, b in zip(used, bandwidths)
    ]
    f0_mw = _map_frequency(_AUDIO_F_MIN + f0, mapping, band, scale) - _map_frequency(
        _AUDIO_F_MIN, mapping, band, scale
    )
    f0_mw = max(f0_mw, 1e6)

    t = time_vector(cfg)
    nyquist = cfg.nyquist
    lo = min(f_mw) - 3 * max(bw_mw)
    hi = min(max(f_mw) + 3 * max(bw_mw), 0.9 * nyquist)

    wave = torch.zeros_like(t)
    k = max(int(lo / f0_mw), 1)
    while k * f0_mw <= hi:
        fk = k * f0_mw
        # Lorentzian formant envelope; amplitudes fall with formant index the
        # way they do in real speech.
        amp = 0.0
        for i, (fc, bw) in enumerate(zip(f_mw, bw_mw)):
            amp += (1.0 / (i + 1)) / (1.0 + ((fk - fc) / (0.5 * bw)) ** 2)
        phase = 2 * math.pi * float(torch.rand(1, generator=generator, device="cpu").item())
        wave = wave + amp * torch.sin(2 * math.pi * fk * t + phase)
        k += 1

    wave = wave * raised_cosine_envelope(len(t), ramp_frac)
    peak = wave.abs().max()
    if peak > 0:
        wave = wave / peak
    return wave, torch.tensor(f_mw, dtype=t.dtype)


def class_separations(cfg, classes, band=None, mapping: str = "affine", scale: float = 2e6):
    """Smallest gap between any two classes' mapped formants, per formant index.

    The number to look at before scaling the task up. Every extra vowel has to
    fit into the same fixed device band, so separations shrink as classes are
    added, and once they fall below the rollout's spectral resolution
    (``1 / duration``) no amount of training or data will separate those
    classes. Returns ``(separations_Hz, resolution_Hz)`` where the first is a
    list of the minimum pairwise gap for F1, F2 and F3.
    """
    if band is None:
        band = usable_band(cfg)

    mapped = []
    for name in classes:
        formants, _ = FORMANTS[name]
        mapped.append([_map_frequency(f, mapping, band, scale) for f in formants])

    seps = []
    for i in range(3):
        col = sorted(m[i] for m in mapped)
        seps.append(min((b - a) for a, b in zip(col, col[1:])) if len(col) > 1 else float("inf"))
    return seps, 1.0 / cfg.duration


def vowel_dataset(
    cfg: SimConfig,
    classes=("ae", "ei", "iy"),
    n_per_class: int = 16,
    jitter: float = 0.06,
    mapping: str = "affine",
    scale: float = 2e6,
    band=None,
    seed: int = 0,
) -> VowelDataset:
    """Build a synthetic vowel classification set matched to the device band.

    :param n_per_class: tokens per vowel. The paper trains on 4 per vowel and
        tests on 45; the default here gives a usable train/test split out of
        the box without a long generation step.
    :param jitter: speaker variability. At 0 the task is trivial (three fixed
        spectra); the default 6 % makes formants overlap enough that the
        network has to generalise.
    :param band: ``(f_lo, f_hi)`` for the affine map. Defaults to
        :func:`usable_band`, so tokens land where the film actually propagates.
    """
    if band is None:
        band = usable_band(cfg)

    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    signals, labels, carriers = [], [], []
    for ci, vowel in enumerate(classes):
        for _ in range(n_per_class):
            wave, f_used = synth_vowel(
                cfg, vowel, band, jitter=jitter, mapping=mapping, scale=scale, generator=gen
            )
            signals.append(wave.reshape(-1, 1))
            labels.append(ci)
            carriers.append(f_used)

    return VowelDataset(
        signals=torch.stack(signals),
        labels=torch.tensor(labels, dtype=torch.long),
        classes=list(classes),
        carrier_frequencies=torch.stack(carriers),
    )
