"""Drive waveforms, the vowel dataset, and the objectives."""

from __future__ import annotations

import math

import pytest
import torch

import magnonic_nn as mnn
from magnonic_nn.signals import raised_cosine_envelope, synth_vowel


def _peak_frequency(wave, dt):
    spectrum = torch.fft.rfft(wave.reshape(-1) - wave.mean()).abs()
    freqs = torch.fft.rfftfreq(wave.numel(), d=dt)
    return float(freqs[int(spectrum.argmax())])


# ------------------------------------------------------------------- signals


def test_tone_shape_and_frequency(f32):
    cfg = mnn.get_preset("tiny")
    cfg.solver.timesteps = 512
    wave = mnn.tone(cfg, 4.0e9)

    assert wave.shape == (512, 1)
    assert _peak_frequency(wave, cfg.solver.dt) == pytest.approx(4.0e9, rel=0.05)


def test_envelope_ramps_from_zero_and_back(f32):
    env = raised_cosine_envelope(100, ramp_frac=0.2)
    assert float(env[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(env[-1]) == pytest.approx(0.0, abs=1e-6)
    assert float(env[50]) == pytest.approx(1.0)
    assert float(env.max()) <= 1.0 + 1e-12


def test_zero_ramp_gives_a_rectangular_window(f32):
    env = raised_cosine_envelope(50, ramp_frac=0.0)
    assert torch.allclose(env, torch.ones_like(env))


def test_multitone_contains_every_requested_frequency(f32):
    cfg = mnn.get_preset("tiny")
    cfg.solver.timesteps = 1024
    freqs = [3.5e9, 4.5e9]
    wave = mnn.multitone(cfg, freqs).reshape(-1)

    spectrum = torch.fft.rfft(wave - wave.mean()).abs()
    axis = torch.fft.rfftfreq(len(wave), d=cfg.solver.dt)
    for f in freqs:
        bin_index = int((axis - f).abs().argmin())
        window = spectrum[max(bin_index - 2, 0) : bin_index + 3]
        assert float(window.max()) > 0.2 * float(spectrum.max())


def test_gated_tones_gives_one_channel_per_frequency(f32):
    cfg = mnn.get_preset("tiny")
    cfg.solver.timesteps = 256
    sig = mnn.gated_tones(cfg, [3.5e9, 4.0e9, 4.5e9], gate_frac=0.5)

    assert sig.shape == (256, 3)
    assert float(sig[200:].abs().max()) == 0.0  # gated off in the second half


def test_chirp_sweeps_upwards(f32):
    cfg = mnn.get_preset("tiny")
    cfg.solver.timesteps = 1024
    wave = mnn.chirp(cfg, 3.0e9, 6.0e9).reshape(-1)

    first = _peak_frequency(wave[:400], cfg.solver.dt)
    last = _peak_frequency(wave[-400:], cfg.solver.dt)
    assert last > first


# -------------------------------------------------------------------- vowels


def test_vowel_dataset_shapes_and_labels(f32):
    cfg = mnn.get_preset("tiny")
    cfg.solver.timesteps = 256
    data = mnn.vowel_dataset(cfg, n_per_class=5, seed=0)

    assert data.signals.shape == (15, 256, 1)
    assert data.labels.shape == (15,)
    assert sorted(data.labels.tolist()) == sorted([0] * 5 + [1] * 5 + [2] * 5)
    assert data.carrier_frequencies.shape == (15, 3)


def test_vowel_carriers_land_in_the_usable_band(f32):
    """The affine map exists precisely so this holds; if it fails the film cannot
    carry the signal and no amount of training will help."""
    cfg = mnn.get_preset("focus")
    cfg.solver.timesteps = 256
    lo, hi = mnn.usable_band(cfg)
    data = mnn.vowel_dataset(cfg, n_per_class=6, seed=1)

    assert float(data.carrier_frequencies.min()) >= lo * 0.9
    assert float(data.carrier_frequencies.max()) <= hi * 1.1


def test_vowel_classes_are_spectrally_distinct(f32):
    """Zero jitter must give perfectly separable formant triples.

    If the classes were not separable in the input, a failure to classify would
    say nothing about the physics.
    """
    cfg = mnn.get_preset("focus")
    cfg.solver.timesteps = 256
    data = mnn.vowel_dataset(cfg, n_per_class=3, jitter=0.0, seed=2)

    centres = [data.carrier_frequencies[data.labels == c][0] for c in range(3)]
    for i in range(3):
        for j in range(i + 1, 3):
            assert float((centres[i] - centres[j]).abs().max()) > 1e7


def test_jitter_spreads_tokens_within_a_class(f32):
    cfg = mnn.get_preset("focus")
    cfg.solver.timesteps = 256
    tight = mnn.vowel_dataset(cfg, n_per_class=8, jitter=0.0, seed=3)
    loose = mnn.vowel_dataset(cfg, n_per_class=8, jitter=0.1, seed=3)

    # std must be taken down each formant column: across the whole (N, 3) block
    # it would measure the ~700 MHz spacing between F1, F2 and F3 instead of the
    # token-to-token spread, and never go near zero.
    assert float(tight.carrier_frequencies[tight.labels == 0].std(dim=0).max()) < 1e3
    assert float(loose.carrier_frequencies[loose.labels == 0].std(dim=0).min()) > 1e6


def test_dataset_is_reproducible(f32):
    cfg = mnn.get_preset("tiny")
    cfg.solver.timesteps = 128
    a = mnn.vowel_dataset(cfg, n_per_class=4, seed=7)
    b = mnn.vowel_dataset(cfg, n_per_class=4, seed=7)
    assert torch.equal(a.signals, b.signals)


def test_split_is_stratified(f32):
    cfg = mnn.get_preset("tiny")
    cfg.solver.timesteps = 128
    data = mnn.vowel_dataset(cfg, n_per_class=10, seed=0)
    train, test = data.split(3)

    assert len(train) == 9 and len(test) == 21
    for c in range(3):
        assert int((train.labels == c).sum()) == 3
        assert int((test.labels == c).sum()) == 7


def test_synth_vowel_is_peak_normalised(f32):
    cfg = mnn.get_preset("tiny")
    cfg.solver.timesteps = 256
    wave, _ = synth_vowel(cfg, "iy", band=mnn.usable_band(cfg))
    assert float(wave.abs().max()) == pytest.approx(1.0, rel=1e-6)


def test_linear_mapping_is_available_but_leaves_the_band(f32):
    """The paper-literal single-factor scaling is offered, and documented as
    unusable at these mesh sizes; this pins that claim down."""
    cfg = mnn.get_preset("focus")
    cfg.solver.timesteps = 256
    lo, hi = mnn.usable_band(cfg)
    data = mnn.vowel_dataset(cfg, n_per_class=2, mapping="linear", scale=2e6, seed=0)
    assert float(data.carrier_frequencies.min()) < lo


# -------------------------------------------------------------------- losses


def test_focus_loss_is_minimised_by_perfect_focus():
    concentrated = torch.tensor([[1e-6, 1e-6, 1.0, 1e-6]])
    spread = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    target = torch.tensor([2])
    assert float(mnn.focus_loss(concentrated, target)) < float(mnn.focus_loss(spread, target))


def test_focus_loss_is_scale_invariant():
    """Multiplying every probe by a constant must not change the loss --
    otherwise the optimiser can improve it by raising throughput alone."""
    u = torch.tensor([[0.1, 0.5, 0.2, 0.2]])
    target = torch.tensor([1])
    assert float(mnn.focus_loss(u, target)) == pytest.approx(
        float(mnn.focus_loss(1000 * u, target)), rel=1e-6
    )


def test_cross_entropy_prefers_the_target_probe():
    right = torch.tensor([[0.05, 0.9, 0.05]])
    wrong = torch.tensor([[0.9, 0.05, 0.05]])
    target = torch.tensor([1])
    assert float(mnn.intensity_cross_entropy(right, target)) < float(
        mnn.intensity_cross_entropy(wrong, target)
    )


def test_cross_entropy_is_scale_invariant():
    u = torch.tensor([[0.1, 0.7, 0.2]])
    target = torch.tensor([1])
    assert float(mnn.intensity_cross_entropy(u, target)) == pytest.approx(
        float(mnn.intensity_cross_entropy(50 * u, target)), rel=1e-6
    )


def test_contrast_in_dB():
    u = torch.tensor([[1.0, 10.0, 1.0]])
    assert float(mnn.contrast(u, torch.tensor([1]))[0]) == pytest.approx(10.0, rel=1e-6)


def test_margin_loss_saturates_once_the_gap_is_won():
    target = torch.tensor([0])
    clear = torch.tensor([[0.9, 0.05, 0.05]])
    assert float(mnn.margin_loss(clear, target, margin=0.2)) == pytest.approx(0.0)

    close = torch.tensor([[0.4, 0.35, 0.25]])
    assert float(mnn.margin_loss(close, target, margin=0.2)) > 0


def test_accuracy_and_confusion_matrix():
    u = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
    target = torch.tensor([0, 1, 1])
    assert mnn.accuracy(u, target) == pytest.approx(2 / 3)

    cm = mnn.confusion_matrix(u, target, 2)
    assert cm.tolist() == [[1, 0], [1, 1]]


def test_losses_accept_an_unbatched_vector():
    u = torch.tensor([0.1, 0.8, 0.1])
    assert math.isfinite(float(mnn.focus_loss(u, torch.tensor([1]))))
    assert math.isfinite(float(mnn.intensity_cross_entropy(u, torch.tensor([1]))))


def test_normalize_intensities_sums_to_one():
    u = torch.tensor([[1.0, 3.0, 4.0], [2.0, 2.0, 6.0]])
    p = mnn.normalize_intensities(u)
    assert torch.allclose(p.sum(dim=-1), torch.ones(2), atol=1e-6)
